"""Still-image decoding and one reusable RGB GPU upload; no video conversion."""
import ctypes as C
import io
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, features

from jxr_image_decode import EXTENSIONS as JXR_EXTENSIONS


EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff', '.avif', '.heic', '.heif','.jxl') + JXR_EXTENSIONS


def heif_plugin():
    try:
        import pillow_heif
    except ImportError as error:
        raise ValueError('当前运行环境缺少 pillow-heif 图片编码组件，请安装项目依赖后重启') from error
    pillow_heif.register_heif_opener()
    return pillow_heif


def export_formats():
    heic = False
    try:
        heic = bool(heif_plugin().libheif_info().get('HEIF'))
    except ValueError:
        pass
    from image_extra_codecs import capabilities
    return dict(png=True, jpeg=True, webp=features.check('webp'),
                avif=features.check('avif'), heic=heic, heif=heic,
                jxl=capabilities()['jxl']['available'],uhdr=False)


def is_image(path):
    return os.path.splitext(path)[1].lower() in EXTENSIONS


def read_pixels(path):
    """Processing master: RGBA8 SDR or signed linear scRGB FP16 HDR."""
    from hdr_image_decode import probe_hdr, decode_hdr
    info = probe_hdr(path)
    return decode_hdr(path, info) if info else _read_sdr_rgba(path)


def read_rgba(path):
    """8-bit display copy only; processing must use read_pixels."""
    pixels = read_pixels(path)
    if pixels.dtype == np.float16:
        from image_render import display_bgr
        rgb = display_bgr(pixels, True)[:, :, ::-1]
        alpha = np.rint(np.clip(pixels[:, :, 3].astype(np.float32), 0, 1)*255).astype(np.uint8)
        return np.ascontiguousarray(np.dstack((rgb, alpha)))
    return pixels


def _read_sdr_rgba(path):
    # Pillow can silently reduce a 16-bit RGB PNG to 8-bit. Check before converting.
    extension = os.path.splitext(path)[1].lower()
    if extension=='.jxl':
        from image_extra_codecs import codec_file,run
        info=run([codec_file('jxlinfo.exe'),'-v',path],timeout=30)
        if ('8-bit RGB' not in info or 'sRGB transfer function' not in info or
                'sRGB primaries' not in info):
            raise ValueError('当前图片入口支持 8-bit sRGB JXL；高位深或 HDR JXL 不会自动降精度')
        with tempfile.TemporaryDirectory(prefix='lunarc-jxl-read-') as folder:
            decoded=os.path.join(folder,'decoded.png')
            run([codec_file('djxl.exe'),path,decoded,'--num_threads=8'])
            return read_rgba(decoded)
    if extension in ('.heic','.heif'):
        source = heif_plugin().open_heif(path, convert_hdr_to_8bit=False)
        if source.info.get('bit_depth',8) > 8:
            raise ValueError('当前图片入口支持 8-bit 图片；高位深 HEIC 不会自动降为 8-bit')
    elif extension in JXR_EXTENSIONS:
        from jxr_image_decode import read_rgba8
        rgba, profile = read_rgba8(path)
        if profile:
            image = ImageCms.profileToProfile(Image.fromarray(rgba[:,:,:3]),
                ImageCms.ImageCmsProfile(io.BytesIO(profile)), ImageCms.createProfile('sRGB'),
                outputMode='RGB')
            rgba = np.dstack((np.asarray(image), rgba[:,:,3]))
        return np.ascontiguousarray(rgba)
    elif extension == '.avif':
        # Pillow's AVIF decoder always returns 8-bit; verify original precision first.
        import pipeline
        info = pipeline.probe_pixel_info(path)
        if not info.get('pix_fmt'):
            raise ValueError('无法确认 AVIF 输入位深')
        if (info.get('bit_depth',8)>8 or info.get('is_hdr') or
                info.get('color_transfer') in ('smpte2084','smpte-st-2084','arib-std-b67')):
            raise ValueError('此 AVIF 的色彩标记或位深尚不支持；HDR 输入需要完整的 BT.2020 / PQ 标记，不会自动降为 SDR')
    else:
        raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError('无法读取图片')
        if raw.dtype != np.uint8:
            raise ValueError('当前图片入口支持 8-bit 图片；16-bit 图片尚未接入，不会自动降为 8-bit')
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        alpha = image.convert('RGBA').getchannel('A')
        profile = opened.info.get('icc_profile')
        if profile:
            image = ImageCms.profileToProfile(image if image.mode in ('RGB','CMYK','L') else image.convert('RGB'),
                ImageCms.ImageCmsProfile(io.BytesIO(profile)), ImageCms.createProfile('sRGB'),
                outputMode='RGB')
        else:
            image = image.convert('RGB')
        image.putalpha(alpha)
        return np.asarray(image).copy()


def probe(path):
    from hdr_image_decode import probe_hdr
    hdr = probe_hdr(path)
    if hdr:
        w, h = hdr['width'], hdr['height']
    else:
        h, w = read_rgba(path).shape[:2]
    return dict(media_type='image', width=(w+7)//8*8, height=(h+7)//8*8,
        image_width=w, image_height=h, pix_fmt='rgba16f' if hdr else 'rgba', video_stream_count=1,
        color_range='pc', color_space='gbr', color_transfer='linear' if hdr else 'iec61966-2-1',
        color_primaries='bt709', r_frame_rate='1/1', audio_streams=0,
        subtitle_streams=0, nb_chapters=0, metadata=False, codec_name='image',
        **(dict(is_hdr=True, bit_depth=16, processing_transfer='linear',
                source_colour=hdr) if hdr else dict(is_hdr=False, bit_depth=8)))


class Capture:
    def __init__(self, path):
        self.rgba = read_rgba(path)
        self.position = 0

    def isOpened(self):
        return self.rgba is not None

    def get(self, key):
        h, w = self.rgba.shape[:2]
        return {cv2.CAP_PROP_FRAME_COUNT: 1, cv2.CAP_PROP_FPS: 1,
            cv2.CAP_PROP_FRAME_WIDTH: w, cv2.CAP_PROP_FRAME_HEIGHT: h}.get(key, 0)

    def set(self, key, value):
        if key == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def read(self):
        if self.position != 0 or self.rgba is None:
            return False, None
        self.position = 1
        return True, cv2.cvtColor(self.rgba, cv2.COLOR_RGBA2BGR)

    def release(self):
        self.rgba = None


class DeviceImage:
    def __init__(self, path):
        self.context, self.pointer = C.c_void_p(), C.c_uint64()
        self.cuda = C.WinDLL('nvcuda.dll')
        for name, args in {
            'cuInit': [C.c_uint], 'cuDeviceGet': [C.POINTER(C.c_int), C.c_int],
            'cuCtxCreate_v2': [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
            'cuCtxDestroy_v2': [C.c_void_p], 'cuMemAlloc_v2': [C.POINTER(C.c_uint64), C.c_size_t],
            'cuMemFree_v2': [C.c_uint64],
            'cuMemcpyHtoD_v2': [C.c_uint64, C.c_void_p, C.c_size_t],
        }.items():
            getattr(self.cuda, name).argtypes = args
        rgba = read_pixels(path)
        if rgba.dtype != np.uint8:
            raise ValueError('HDR 图片需要使用线性 FP16 图片渲染入口')
        h, w = rgba.shape[:2]
        self.width, self.height = (w+7)//8*8, (h+7)//8*8
        image = np.pad(rgba, ((0,self.height-h), (0,self.width-w), (0,0)), mode='edge')
        image[:, :, 3] = 255  # Models process colour; export restores source transparency.
        self.host_upload_bytes = image.nbytes
        self.pitch = self.width * 4
        try:
            self.check(self.cuda.cuInit(0))
            device = C.c_int()
            self.check(self.cuda.cuDeviceGet(C.byref(device), 0))
            self.check(self.cuda.cuCtxCreate_v2(C.byref(self.context), 0, device))
            self.check(self.cuda.cuMemAlloc_v2(C.byref(self.pointer), image.nbytes))
            self.check(self.cuda.cuMemcpyHtoD_v2(self.pointer, image.ctypes.data, image.nbytes))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'图片 GPU 上传失败，CUDA 错误 {code}')

    def frame(self, frame):
        frame.width, frame.height = self.width, self.height
        frame.cuda_context = self.context.value
        frame.time_base_num = frame.time_base_den = 1
        frame.pts = 0
        frame.image_rgba = (self.pointer.value, self.pitch)
        return 1

    def close(self):
        if self.pointer.value:
            self.cuda.cuMemFree_v2(self.pointer)
            self.pointer.value = 0
        if self.context.value:
            self.cuda.cuCtxDestroy_v2(self.context)
            self.context.value = None


def output_pixels(bgr, source, settings):
    """Trim source padding; fixed VSR output resolutions need no trimming."""
    w, h = output_extent(bgr.shape[1], bgr.shape[0], source, settings)
    return bgr[:h, :w]


def output_extent(width, height, source, settings):
    """Logical image extent; native display uses this without image readback."""
    import native_effects
    from vsr_config import configuration
    fixed=any(native_effects.stage_kind(stage)=='vsr' and
        not configuration(native_effects.stage_settings(settings,stage))['same_size'] and
        native_effects.stage_settings(settings,stage).get('vsr_output_mode','resolution')=='resolution'
        for stage in native_effects.stages(settings))
    if fixed:
        return width, height
    w = round(width * source['image_width'] / source['width'])
    h = round(height * source['image_height'] / source['height'])
    return max(1,w), max(1,h)


FORMATS = {'png': ('.png', 'PNG'), 'jpeg': ('.jpg', 'JPEG'), 'webp': ('.webp', 'WEBP'),
           'avif': ('.avif', 'AVIF'), 'heic': ('.heic', 'HEIF'), 'heif': ('.heif', 'HEIF'),
           'jxl': ('.jxl','JXL'),'uhdr': ('.jpg','UHDR')}


def validate_destination(path, output_path, fmt, quality, overwrite=False, color_mode='sdr'):
    if fmt not in FORMATS or not 1 <= quality <= 100:
        raise ValueError('图片格式或质量参数无效')
    if color_mode not in ('sdr', 'hdr10'):
        raise ValueError('图片输出色彩模式无效')
    if color_mode == 'hdr10':
        from image_hdr_export import capabilities
        info = capabilities().get(fmt)
        if not info or not info['available']:
            raise ValueError(info['reason'] if info else 'HDR 图片请选择 AVIF、HEIC、HEIF、UHDR 或 JXL')
    elif fmt == 'uhdr':
        raise ValueError('UHDR JPEG 需要 HDR 渲染结果；保存普通 SDR 请选 JPEG')
    elif fmt == 'jxl':
        from image_extra_codecs import capabilities
        info=capabilities()['jxl']
        if not info['available']:raise ValueError(info['reason'])
    elif fmt in ('heic', 'heif'):
        heif_plugin()
    elif fmt == 'avif' and not features.check('avif'):
        raise ValueError('当前 Pillow 缺少 AVIF 编码支持，请安装项目依赖后重启')
    if os.path.splitext(output_path)[1].lower() not in (FORMATS[fmt][0], '.jpeg' if fmt == 'jpeg' else FORMATS[fmt][0]):
        raise ValueError('文件扩展名与导出格式不一致')
    if os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(output_path)):
        raise ValueError('不能覆盖原图片')
    if not os.path.isdir(os.path.dirname(os.path.abspath(output_path))):
        raise ValueError('保存目录不存在')
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError('目标文件已存在；请更换文件名或开启覆盖')


def save_image(path, rgba, output_path, fmt, quality=95, preserve_alpha=True, overwrite=False, source_alpha=None):
    validate_destination(path, output_path, fmt, quality, overwrite)
    image = Image.fromarray(rgba)
    source_alpha = read_rgba(path)[:, :, 3] if source_alpha is None else source_alpha
    if preserve_alpha and fmt != 'jpeg' and np.any(source_alpha != 255):
        alpha = Image.fromarray(source_alpha).resize(image.size, Image.Resampling.BILINEAR)
        image.putalpha(alpha)
    elif fmt == 'jpeg' and np.any(source_alpha != 255):
        alpha = Image.fromarray(source_alpha).resize(image.size, Image.Resampling.BILINEAR)
        image = Image.composite(image, Image.new('RGB', image.size, 'white'), alpha)
    parent = os.path.dirname(os.path.abspath(output_path))
    fd, temporary = tempfile.mkstemp(prefix='.lunarc-image-', dir=parent)
    os.close(fd)
    try:
        opts = {'quality':quality} if fmt != 'png' else {}
        opts['icc_profile'] = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        if fmt == 'jpeg': opts['subsampling'] = 0
        if fmt == 'webp': opts.update(lossless=quality == 100, exact=True)
        if fmt == 'avif': opts.update(subsampling='4:4:4',speed=6,max_threads=8)
        if fmt in ('heic', 'heif'):
            opts.update(chroma=444,color_primaries=1,transfer_characteristic=13,full_range_flag=1)
            if quality == 100: opts.update(quality=-1,matrix_coefficients=0)
        if fmt=='jxl':
            from image_extra_codecs import save_jxl
            pixels=np.asarray(image)
            save_jxl(temporary,pixels[:,:,:3],pixels[:,:,3] if pixels.shape[2]==4 else None,quality)
        else:image.save(temporary, format=FORMATS[fmt][1], **opts)
        if overwrite:
            os.replace(temporary, output_path)
        else:
            # Atomic no-overwrite publication, including a file created during render.
            os.link(temporary, output_path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return dict(path=output_path, width=image.width, height=image.height, format=fmt,
                bytes=os.path.getsize(output_path), bit_depth=8, color_mode='sRGB SDR')
