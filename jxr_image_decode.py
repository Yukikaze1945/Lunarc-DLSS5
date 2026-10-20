"""JPEG XR stills through the built-in Windows Imaging Component (WIC).

WIC exposes JPEG XR as integer SDR or linear scRGB HDR. This module accepts
8-bit SDR and fixed/float scRGB HDR; 16/32-bit integer SDR is rejected rather
than silently reduced. HDR values stay linear at 80 cd/m² per unit so they
feed the same FP16 master as PQ AVIF input.
"""
import ctypes as C
from ctypes import wintypes as W
import functools
import os
import uuid

import numpy as np


EXTENSIONS = ('.jxr', '.wdp', '.hdp')

_GENERIC_READ = 0x80000000
_CLSCTX_INPROC_SERVER = 1
_COINIT_MULTITHREADED = 0
_RPC_E_CHANGED_MODE = 0x80010106
_WIC_COLOR_CONTEXT_PROFILE = 1


class _GUID(C.Structure):
    _fields_ = [('Data1', C.c_ulong), ('Data2', C.c_ushort), ('Data3', C.c_ushort),
                ('Data4', C.c_ubyte * 8)]

    @classmethod
    def parse(cls, text):
        value = uuid.UUID(text)
        result = cls()
        result.Data1 = value.time_low
        result.Data2 = value.time_mid
        result.Data3 = value.time_hi_version
        result.Data4[:] = value.bytes[8:]
        return result

    def __str__(self):
        return str(uuid.UUID(bytes_le=bytes(self)))

    def __eq__(self, other):
        return isinstance(other, _GUID) and bytes(self) == bytes(other)

    def __hash__(self):
        return hash(bytes(self))


def _guid(text):
    return _GUID.parse(text)


_CLSID_WIC_IMAGING_FACTORY = _guid('cacaf262-9370-4615-a13b-9f5539da4c0a')
_IID_IWIC_IMAGING_FACTORY = _guid('ec5ec8a9-c395-4314-9c77-54d7a935ff70')
_PIXEL_RGBA8 = _guid('f5c7ad2d-6a8d-43dd-a7a8-a29935261ae9')
_PIXEL_RGBA128F = _guid('6fddc324-4e03-4bfe-b185-3d77768dc919')


def _wic(tail):
    return _guid('6fddc324-4e03-4bfe-b185-3d77768dc9' + tail)


# GUID -> (kind, source bit depth, WIC pixel format name)
_FORMATS = {
    # 8-bit integer SDR
    _wic('08'): ('sdr8', 8, '8bppGray'),
    _guid('e6cd0116-eeba-4161-aa85-27dd9fb3a895'): ('sdr8', 8, '8bppAlpha'),
    _wic('0c'): ('sdr8', 8, '24bppBGR'),
    _wic('0d'): ('sdr8', 8, '24bppRGB'),
    _wic('0e'): ('sdr8', 8, '32bppBGR'),
    _wic('0f'): ('sdr8', 8, '32bppBGRA'),
    _wic('10'): ('sdr8', 8, '32bppPBGRA'),
    _guid('d98c6b95-3efe-47d6-bb25-eb1748ab0cf1'): ('sdr8', 8, '32bppRGB'),
    _guid('f5c7ad2d-6a8d-43dd-a7a8-a29935261ae9'): ('sdr8', 8, '32bppRGBA'),
    _guid('3cc4a650-a527-4d37-a916-3142c7ebedba'): ('sdr8', 8, '32bppPRGBA'),
    # 16/32-bit integer SDR: high precision, not HDR
    _wic('0b'): ('sdr16', 16, '16bppGray'),
    _wic('15'): ('sdr16', 16, '48bppRGB'),
    _guid('e605a384-b468-46ce-bb2e-36f180e64313'): ('sdr16', 16, '48bppBGR'),
    _guid('a1182111-186d-4d42-bc6a-9c8303a8dff9'): ('sdr16', 16, '64bppRGB'),
    _wic('16'): ('sdr16', 16, '64bppRGBA'),
    _guid('1562ff7c-d352-46f9-979e-42976b792246'): ('sdr16', 16, '64bppBGRA'),
    _wic('17'): ('sdr16', 16, '64bppPRGBA'),
    _guid('8c518e8e-a4ec-468b-ae70-c9a35a9c5530'): ('sdr16', 16, '64bppPBGRA'),
    # Fixed point / float: linear scRGB HDR
    _wic('11'): ('hdr', 32, '32bppGrayFloat'),
    _wic('3e'): ('hdr', 16, '16bppGrayHalf'),
    _wic('3f'): ('hdr', 32, '32bppGrayFixedPoint'),
    _wic('12'): ('hdr', 16, '48bppRGBFixedPoint'),
    _guid('49ca140e-cab6-493b-9ddf-60187c37532a'): ('hdr', 16, '48bppBGRFixedPoint'),
    _wic('1d'): ('hdr', 16, '64bppRGBAFixedPoint'),
    _guid('356de33c-54d2-4a23-bb04-9b7bf9b1d42d'): ('hdr', 16, '64bppBGRAFixedPoint'),
    _wic('40'): ('hdr', 16, '64bppRGBFixedPoint'),
    _wic('18'): ('hdr', 32, '96bppRGBFixedPoint'),
    _wic('1e'): ('hdr', 32, '128bppRGBAFixedPoint'),
    _wic('41'): ('hdr', 32, '128bppRGBFixedPoint'),
    _wic('3b'): ('hdr', 16, '48bppRGBHalf'),
    _wic('3a'): ('hdr', 16, '64bppRGBAHalf'),
    _wic('42'): ('hdr', 16, '64bppRGBHalf'),
    _guid('e3fed78f-e8db-4acf-84c1-e97f6136b327'): ('hdr', 32, '96bppRGBFloat'),
    _wic('19'): ('hdr', 32, '128bppRGBAFloat'),
    _wic('1a'): ('hdr', 32, '128bppPRGBAFloat'),
    _wic('1b'): ('hdr', 32, '128bppRGBFloat'),
    _wic('3d'): ('hdr', 32, '32bppRGBE'),
}


@functools.lru_cache(maxsize=1)
def _libraries():
    if os.name != 'nt':
        raise ValueError('JXR 图片解码依赖 Windows Imaging Component，仅支持 Windows')
    ole32 = C.WinDLL('ole32')
    ole32.CoInitializeEx.argtypes = [C.c_void_p, C.c_ulong]
    ole32.CoInitializeEx.restype = C.c_long
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    ole32.CoCreateInstance.argtypes = [C.POINTER(_GUID), C.c_void_p, C.c_ulong,
                                       C.POINTER(_GUID), C.POINTER(C.c_void_p)]
    ole32.CoCreateInstance.restype = C.c_long
    return ole32


def _method(pointer, index, *argtypes):
    table = C.cast(pointer, C.POINTER(C.POINTER(C.c_void_p)))[0]
    function = C.WINFUNCTYPE(C.c_long, C.c_void_p, *argtypes)(table[index])
    return lambda *args: function(pointer, *args)


def _release(pointer):
    if pointer:
        _method(pointer, 2)()


def _check(result, message):
    if result < 0:
        raise OSError(f'{message}（WIC 0x{result & 0xffffffff:08X}）')


def _initialize(ole32):
    result = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if (result & 0xffffffff) == _RPC_E_CHANGED_MODE:
        return False
    _check(result, '无法初始化 Windows Imaging Component')
    return True


def _create_factory(ole32):
    factory = C.c_void_p()
    _check(ole32.CoCreateInstance(C.byref(_CLSID_WIC_IMAGING_FACTORY), None,
                                 _CLSCTX_INPROC_SERVER, C.byref(_IID_IWIC_IMAGING_FACTORY),
                                 C.byref(factory)), '无法创建 WIC 解码器')
    return factory


def _open_frame(factory, path):
    decoder = C.c_void_p()
    frame = C.c_void_p()
    try:
        create_decoder = _method(factory, 3, W.LPCWSTR, C.POINTER(_GUID), W.DWORD,
                                 C.c_int, C.POINTER(C.c_void_p))
        _check(create_decoder(path, None, _GENERIC_READ, 0, C.byref(decoder)),
               '无法打开 JXR 文件')
        get_frame = _method(decoder, 13, W.UINT, C.POINTER(C.c_void_p))
        _check(get_frame(0, C.byref(frame)), 'JXR 文件没有可读取的图像帧')
        return decoder, frame
    except BaseException:
        _release(frame)
        _release(decoder)
        raise


def _frame_info(frame):
    get_size = _method(frame, 3, C.POINTER(W.UINT), C.POINTER(W.UINT))
    width, height = W.UINT(), W.UINT()
    _check(get_size(C.byref(width), C.byref(height)), '无法读取 JXR 尺寸')
    get_format = _method(frame, 4, C.POINTER(_GUID))
    pixel_format = _GUID()
    _check(get_format(C.byref(pixel_format)), '无法读取 JXR 像素格式')
    if not width.value or not height.value:
        raise ValueError('JXR 图像尺寸无效')
    return width.value, height.value, pixel_format


def _color_profiles(factory, frame):
    get_contexts = _method(frame, 9, W.UINT, C.POINTER(C.c_void_p),
                           C.POINTER(W.UINT))
    count = W.UINT()
    _check(get_contexts(0, None, C.byref(count)), '无法读取 JXR 色彩配置')
    if not count.value:
        return []
    create_context = _method(factory, 15, C.POINTER(C.c_void_p))
    contexts = (C.c_void_p * count.value)()
    created = []
    try:
        for index in range(count.value):
            _check(create_context(C.byref(contexts[index])), '无法创建 WIC 色彩配置')
            created.append(contexts[index])
        actual = W.UINT()
        _check(get_contexts(count.value, C.cast(contexts, C.POINTER(C.c_void_p)),
                            C.byref(actual)), '无法读取 JXR 色彩配置')
        profiles = []
        for context in contexts[:actual.value]:
            get_type = _method(context, 6, C.POINTER(C.c_int))
            context_type = C.c_int()
            if get_type(C.byref(context_type)) < 0:
                continue
            if context_type.value != _WIC_COLOR_CONTEXT_PROFILE:
                continue
            get_profile = _method(context, 7, W.UINT, C.POINTER(C.c_ubyte),
                                  C.POINTER(W.UINT))
            size = W.UINT()
            if get_profile(0, None, C.byref(size)) < 0 or size.value < 132:
                continue
            buffer = (C.c_ubyte * size.value)()
            if get_profile(size.value, buffer, C.byref(size)) < 0:
                continue
            profiles.append(bytes(buffer[:size.value]))
        return profiles
    finally:
        for context in created:
            _release(context)


def _describe(pixel_format):
    kind, depth, name = _FORMATS.get(pixel_format, ('unknown', 0, str(pixel_format)))
    return kind, depth, name


def _validate(kind, name, purpose):
    if purpose == 'sdr':
        if kind == 'sdr8':
            return
        if kind == 'hdr':
            raise ValueError('JXR 是 scRGB HDR 图像，不能进入 8-bit SDR 入口')
        if kind == 'sdr16':
            raise ValueError('当前图片入口支持 8-bit JXR 与 scRGB HDR JXR；'
                             '16-bit / 32-bit 整数 JXR 不会自动降为 8-bit')
        raise ValueError(f'当前 JXR 入口只支持 RGB 或灰度图像（像素格式 {name}）')
    if purpose == 'hdr' and kind != 'hdr':
        raise ValueError('JXR 不是 JPEG XR 固定点或浮点 HDR 像素格式')


def _create_converter(factory, frame, target):
    converter = C.c_void_p()
    _check(_method(factory, 10, C.POINTER(C.c_void_p))(C.byref(converter)),
           '无法创建 JXR 像素转换器')
    try:
        initialize = _method(converter, 8, C.c_void_p, C.POINTER(_GUID), C.c_int,
                             C.c_void_p, C.c_double, C.c_int)
        _check(initialize(frame, C.byref(target), 0, None, 0.0, 0),
               '无法转换 JXR 像素格式')
    except BaseException:
        _release(converter)
        raise
    return converter


def _copy_pixels(converter, width, height, target):
    bytes_per_pixel = 16 if target == _PIXEL_RGBA128F else 4
    stride = width * bytes_per_pixel
    buffer = (C.c_ubyte * (stride * height))()
    copy_pixels = _method(converter, 7, C.c_void_p, W.UINT, W.UINT,
                          C.POINTER(C.c_ubyte))
    _check(copy_pixels(None, stride, C.sizeof(buffer), buffer), '无法读取 JXR 像素')
    if target == _PIXEL_RGBA128F:
        return np.frombuffer(buffer, dtype='<f4').reshape(height, width, 4).copy()
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4).copy()


def _read(path, target, purpose):
    path = os.fspath(path)
    ole32 = _libraries()
    initialized = False
    factory = decoder = frame = converter = C.c_void_p()
    try:
        initialized = _initialize(ole32)
        factory = _create_factory(ole32)
        decoder, frame = _open_frame(factory, path)
        width, height, pixel_format = _frame_info(frame)
        kind, depth, name = _describe(pixel_format)
        profiles = _color_profiles(factory, frame)
        _validate(kind, name, purpose)
        result = dict(width=width, height=height, kind=kind, bit_depth=depth,
                      pixel_format=name, pixel_guid=str(pixel_format),
                      profiles=profiles)
        if target is not None:
            converter = _create_converter(factory, frame, target)
            result['pixels'] = _copy_pixels(converter, width, height, target)
        return result
    except OSError as error:
        raise ValueError(f'无法读取 JXR 图片：{error}') from error
    finally:
        _release(converter)
        _release(frame)
        _release(decoder)
        _release(factory)
        if initialized:
            ole32.CoUninitialize()


def _icc_description(profile):
    if len(profile) < 132:
        return ''
    count = int.from_bytes(profile[128:132], 'big')
    if not 1 <= count <= 4096:
        return ''
    for index in range(count):
        entry = 132 + index * 12
        if entry + 12 > len(profile):
            break
        signature = profile[entry:entry + 4]
        if signature not in (b'desc', b'dmnd', b'dmdd'):
            continue
        offset = int.from_bytes(profile[entry + 4:entry + 8], 'big')
        size = int.from_bytes(profile[entry + 8:entry + 12], 'big')
        if size < 12 or offset + size > len(profile):
            continue
        tag = profile[offset:offset + size]
        if tag[:4] == b'desc':
            length = int.from_bytes(tag[8:12], 'big')
            if 12 + length <= len(tag):
                return tag[12:12 + length].split(b'\0')[0].decode('ascii', 'replace')
        elif tag[:4] == b'mluc' and len(tag) >= 28:
            records = int.from_bytes(tag[8:12], 'big')
            if records:
                length = int.from_bytes(tag[20:24], 'big')
                text_offset = int.from_bytes(tag[24:28], 'big')
                if text_offset + length <= len(tag):
                    return tag[text_offset:text_offset + length].decode(
                        'utf-16-be', 'replace').rstrip('\0')
    return ''


def _hdr_profiles_supported(profiles):
    """Untagged JPEG XR fixed/float data is scRGB; a conflicting profile is not."""
    for profile in profiles:
        description = _icc_description(profile)
        if 'scrgb' not in description.lower():
            return False, description
    return True, ''


def probe_hdr(path):
    """Return HDR metadata for fixed/float JXR, otherwise ``None``."""
    if os.path.splitext(os.fspath(path))[1].lower() not in EXTENSIONS:
        return None
    info = _read(path, None, 'info')
    if info['kind'] != 'hdr':
        return None
    supported, description = _hdr_profiles_supported(info['profiles'])
    if not supported:
        raise ValueError('HDR JXR 含有非 scRGB 的 ICC 色彩配置'
                         f'（{description or "无法识别"}），当前入口不支持该色彩空间')
    return dict(codec='jxr', width=info['width'], height=info['height'],
                bit_depth=info['bit_depth'], pix_fmt='rgba16f', is_hdr=True,
                color_space='rgb', color_range='pc', color_transfer='linear',
                color_primaries='bt709', jxr_pixel_format=info['pixel_format'],
                jxr_pixel_guid=info['pixel_guid'],
                jxr_hdr_encoding='scRGB fixed/float',
                embedded_profile_count=len(info['profiles']))


def decode_hdr(path, info=None):
    """Decode fixed/float JXR into linear scRGB FP16 RGBA (1 = 80 cd/m²)."""
    result = _read(path, _PIXEL_RGBA128F, 'hdr')
    supported, description = _hdr_profiles_supported(result['profiles'])
    if not supported:
        raise ValueError('HDR JXR 含有非 scRGB 的 ICC 色彩配置'
                         f'（{description or "无法识别"}），当前入口不支持该色彩空间')
    pixels = result['pixels']
    if not np.isfinite(pixels).all():
        raise ValueError('JXR HDR 解码产生非有限值')
    alpha = pixels[:, :, 3]
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError('JXR HDR 解码的 alpha 超出 [0, 1]')
    if np.any(np.abs(pixels) > np.finfo(np.float16).max):
        raise ValueError('JXR HDR 分量超出 float16 范围')
    with np.errstate(over='ignore'):
        converted = pixels.astype(np.float16)
    if not np.isfinite(converted).all():
        raise ValueError('JXR HDR 分量无法表示为 float16')
    return converted


def read_rgba8(path):
    """Decode 8-bit SDR JXR to RGBA8 plus the first embedded ICC profile."""
    result = _read(path, _PIXEL_RGBA8, 'sdr')
    return result['pixels'], (result['profiles'][0] if result['profiles'] else None)
