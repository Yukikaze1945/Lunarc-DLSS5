"""True HDR stills: linear Rec.709 scRGB (80 nits/unit) -> BT.2020/PQ.

Never passes through an 8-bit display image, applies display tone mapping or
attaches an sRGB ICC profile. Image encoders run on the CPU after FP16 readback.
"""
import functools
import json
import os
import re
import subprocess
import tempfile

import numpy as np


REC709_TO_REC2020 = np.array([
    [.6274038959, .3292830384, .0433130657],
    [.0690972894, .9195403951, .0113623156],
    [.0163914389, .0880133079, .8955952532],
], dtype=np.float32)
CICP = dict(color_primaries=9, transfer_characteristics=16,
            matrix_coefficients=9, full_range_flag=1)


def _run(args, timeout=15):
    return subprocess.run(args, capture_output=True, timeout=timeout,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


@functools.lru_cache(maxsize=8)
def _avif_capability(ffmpeg, mtime, ffprobe):
    # Help queries only: no device initialization, frames or encoder sessions.
    try:
        help_result = _run([ffmpeg, '-hide_banner', '-h', 'encoder=libaom-av1'])
        mux_result = _run([ffmpeg, '-hide_banner', '-muxers'])
        help_text = (help_result.stdout + help_result.stderr).decode('utf-8', 'replace')
        mux_text = mux_result.stdout.decode('utf-8', 'replace')
        supported = (help_result.returncode == mux_result.returncode == 0 and
                     'yuv444p10le' in help_text and 'still-picture' in help_text and
                     re.search(r'^\s*E\s+avif\s', mux_text, re.MULTILINE) is not None)
        return dict(available=bool(supported and ffprobe), alpha='gray10le' in help_text,
                    reason='' if supported and ffprobe else '视频处理组件缺少 10-bit AVIF 编码或检查支持')
    except (OSError, subprocess.SubprocessError) as error:
        return dict(available=False, alpha=False, reason=f'无法检查 HDR AVIF 编码器：{error}')


def capabilities():
    from image_source import heif_plugin
    import pipeline
    ffmpeg, ffprobe = pipeline._bundle_ffmpeg(), pipeline._bundle_ffprobe()
    try:
        avif = _avif_capability(ffmpeg, os.stat(ffmpeg).st_mtime_ns, ffprobe) if ffmpeg else dict(
            available=False, alpha=False, reason='请先安装视频处理组件以保存 HDR AVIF')
    except OSError as error:
        avif = dict(available=False, alpha=False, reason=f'无法检查 HDR AVIF 编码器：{error}')
    try:
        heic = bool(heif_plugin().libheif_info().get('HEIF'))
        heic_reason = '' if heic else '当前 HEIC 组件没有可用的编码器'
    except (ValueError, OSError) as error:
        heic, heic_reason = False, str(error)
    from image_extra_codecs import capabilities as extra_capabilities
    heif = dict(available=heic, alpha=heic, reason=heic_reason)
    return dict(avif=avif, heic=heif, heif=dict(heif), **extra_capabilities())


def validate_request(settings, fmt, path, preserve_alpha):
    import native_effects
    from image_source import read_rgba
    from hdr_image_decode import probe_hdr
    if not native_effects.has_stage(settings, 'truehdr') and not probe_hdr(path):
        raise ValueError('HDR 图片保存需要在渲染链中开启 TrueHDR；普通 SDR 不能直接标记为 HDR')
    info = capabilities().get(fmt)
    if not info or not info['available']:
        raise ValueError(info['reason'] if info else 'HDR 图片请选择 AVIF、HEIC、HEIF、UHDR 或 JXL')
    if preserve_alpha and not info['alpha'] and np.any(read_rgba(path)[:, :, 3] != 255):
        raise ValueError('当前 HDR 图片编码器不支持透明度，请更换格式或关闭保留透明度')


def scrgb_to_pq16(rgb):
    """Full-range 16-bit RGB PQ codes for a 10-bit encoder; no exposure scaling.

    Work in rows to avoid several full-size FP32 intermediates on large stills.
    Negative/out-of-gamut values and >10000 nits are clipped and counted.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('HDR 保存需要三通道线性 RGB 图像')
    encoded = np.empty(rgb.shape, dtype='<u2')
    below = above = 0
    for row in range(0, rgb.shape[0], 128):
        linear = rgb[row:row+128].astype(np.float32)
        if not np.isfinite(linear).all():
            raise ValueError('HDR 图像包含非有限亮度值，未保存')
        wide = linear @ REC709_TO_REC2020.T
        below += int(np.count_nonzero(wide < 0))
        above += int(np.count_nonzero(wide > 125))  # 125 scRGB = 10000 nits.
        normalized = np.clip(wide * .008, 0, 1)
        power = np.power(normalized, 2610/16384)
        pq = np.power((3424/4096 + 2413/128*power)/(1 + 2392/128*power), 2523/32)
        encoded[row:row+128] = np.rint(pq * 65535).astype('<u2')
    return encoded, dict(negative_bt2020_components_clipped=below,
                         above_10000_nits_components_clipped=above,
                         tone_mapping=False, scrgb_nits_per_unit=80)


def _write_yuv444p10(path, pq):
    # BT.2020 non-constant luminance, full range. Planar words are LSB-aligned.
    # Avoid swscale's implicit/default RGB colour matrix and any 8-bit formats.
    with open(path, 'wb') as file:
        for plane in range(3):
            for row in range(0, pq.shape[0], 128):
                rgb = pq[row:row+128].astype(np.float32) / 65535
                y = .2627*rgb[:, :, 0] + .6780*rgb[:, :, 1] + .0593*rgb[:, :, 2]
                value = (y*1023 if plane == 0 else (rgb[:, :, 2]-y)*1023/1.8814 + 512
                         if plane == 1 else (rgb[:, :, 0]-y)*1023/1.4746 + 512)
                np.rint(np.clip(value, 0, 1023)).astype('<u2').tofile(file)


def avif_command(ffmpeg, color, alpha, size, output, quality):
    width, height = size
    args = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-f', 'rawvideo', '-pixel_format', 'yuv444p10le', '-video_size', f'{width}x{height}',
            '-framerate', '1', '-i', color]
    if alpha:
        args += ['-f', 'rawvideo', '-pixel_format', 'gray10le', '-video_size', f'{width}x{height}',
                 '-framerate', '1', '-i', alpha]
    args += ['-map', '0:v:0']
    if alpha:
        args += ['-map', '1:v:0']
    args += ['-c:v', 'libaom-av1', '-still-picture', '1', '-cpu-used', '6', '-threads', '8',
             '-b:v', '0', '-crf', str(round((100-quality)*63/99)),
             '-frames:v:0', '1', '-pix_fmt:v:0', 'yuv444p10le', '-profile:v:0', '1',
             # libaom also reads per-frame color properties. Rawvideo leaves
             # these unspecified; output codec options alone can be overwritten.
             # setparams tags already converted PQ/YUV; it never changes pixels.
             '-filter:v:0', 'setparams=range=full:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc',
             '-color_primaries:v:0', 'bt2020', '-color_trc:v:0', 'smpte2084',
             '-colorspace:v:0', 'bt2020nc', '-color_range:v:0', 'pc']
    if alpha:
        args += ['-frames:v:1', '1', '-pix_fmt:v:1', 'gray10le', '-profile:v:1', '0',
                 '-crf:v:1', '0', '-color_range:v:1', 'pc']
    return args + ['-f', 'avif', output]


def _save_avif(path, pq, alpha, quality):
    import pipeline
    with tempfile.TemporaryDirectory(prefix='.lunarc-hdr-', dir=os.path.dirname(path)) as folder:
        color_path = os.path.join(folder, 'color.yuv')
        _write_yuv444p10(color_path, pq)
        alpha_path = None
        if alpha is not None:
            alpha_path = os.path.join(folder, 'alpha.yuv')
            np.rint(alpha.astype(np.float32)*1023/255).astype('<u2').tofile(alpha_path)
        command = avif_command(pipeline._bundle_ffmpeg(), color_path, alpha_path,
                               (pq.shape[1], pq.shape[0]), path, quality)
        result = _run(command, timeout=600)
        if result.returncode:
            raise ValueError('HDR AVIF 编码失败：' + result.stderr.decode('utf-8', 'replace')[-3000:])
    # Check declarations before publishing; do not accept a silent 8-bit fallback.
    result = _run([pipeline._bundle_ffprobe(), '-v', 'error', '-select_streams', 'v:0',
                   '-show_streams', '-of', 'json', path])
    if result.returncode:
        raise ValueError('无法确认 HDR AVIF 的实际编码信息，未保存')
    streams = json.loads(result.stdout).get('streams', [])
    info = streams[0] if streams else {}
    expected = dict(codec_name='av1', pix_fmt='yuv444p10le', color_primaries='bt2020',
                    color_transfer='smpte2084', color_space='bt2020nc', color_range='pc',
                    width=pq.shape[1], height=pq.shape[0])
    mismatches = {key:dict(expected=value,actual=info.get(key))
                  for key,value in expected.items() if info.get(key) != value}
    if mismatches:
        names = dict(codec_name='编码器',pix_fmt='像素格式',color_primaries='原色',
                     color_transfer='传递曲线',color_space='色彩矩阵',color_range='范围',width='宽度',height='高度')
        details = '；'.join(f"{names[key]}应为 {values['expected']}，实际为 {values['actual'] or '未声明'}"
                           for key,values in mismatches.items())
        raise ValueError('HDR AVIF 编码信息与请求不符，未保存：'+details)


def _save_heic(path, pq, alpha, quality):
    from image_source import heif_plugin
    plugin = heif_plugin()
    if alpha is not None:
        image = np.empty((*pq.shape[:2], 4), dtype='<u2')
        image[:, :, :3], image[:, :, 3] = pq, alpha.astype('<u2') * 257
    else:
        image = pq
    # RGB(A);16 is full-range 16-bit input; bit_depth explicitly selects 10-bit output.
    heif = plugin.from_bytes('RGBA;16' if alpha is not None else 'RGB;16',
        (pq.shape[1], pq.shape[0]), image.tobytes(), stride=image.strides[0])
    heif.save(path, quality=-1 if quality == 100 else quality, bit_depth=10,
              chroma=444, save_nclx_profile=True, **CICP)
    saved = plugin.open_heif(path, convert_hdr_to_8bit=False)
    info = saved.info
    if (info.get('bit_depth') != 10 or any(
            info.get('nclx_profile', {}).get(key) != value for key, value in CICP.items()) or
            saved.size != (pq.shape[1], pq.shape[0])):
        raise ValueError('HDR HEIC 的位深、色彩声明或尺寸与请求不符，未保存')


def save_hdr_image(source_path, rgb, output_path, fmt, quality=95, preserve_alpha=True, overwrite=False, source_alpha=None):
    from image_source import read_rgba, validate_destination
    from PIL import Image
    validate_destination(source_path, output_path, fmt, quality, overwrite, color_mode='hdr10')
    h,w=rgb.shape[:2]
    alpha = None
    if preserve_alpha:
        source_alpha = read_rgba(source_path)[:, :, 3] if source_alpha is None else source_alpha
        if np.any(source_alpha != 255):
            alpha = np.asarray(Image.fromarray(source_alpha).resize(
                (w,h), Image.Resampling.BILINEAR))
    if alpha is not None and not capabilities()[fmt]['alpha']:
        raise ValueError('当前 HDR 图片编码器不支持透明度，请更换格式或关闭保留透明度')
    fd, temporary = tempfile.mkstemp(prefix='.lunarc-image-', dir=os.path.dirname(os.path.abspath(output_path)))
    os.close(fd)
    try:
        if fmt in ('uhdr','jxl'):
            from image_extra_codecs import save_uhdr,save_jxl
            metadata=(save_uhdr(temporary,rgb,quality) if fmt=='uhdr'
                      else save_jxl(temporary,rgb,alpha,quality,hdr=True))
        else:
            pq, report = scrgb_to_pq16(rgb)
            (_save_avif if fmt == 'avif' else _save_heic)(temporary, pq, alpha, quality)
            metadata=dict(bit_depth=10,color_primaries='bt2020',color_transfer='smpte2084',
                color_space='bt2020nc',color_range='pc',chroma='4:4:4',
                has_alpha=alpha is not None,hdr_mapped_to_sdr=False,conversion=report)
        if overwrite:
            os.replace(temporary, output_path)
        else:
            os.link(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dict(path=output_path,width=w,height=h,format=fmt,bytes=os.path.getsize(output_path),
                color_mode='hdr10',**metadata)
