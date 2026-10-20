"""JXR still input through Windows Imaging Component; no GPU."""
import base64
import os
import struct

import cv2
import numpy as np
import pytest

import hdr_image_decode
import image_source


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='JXR input uses Windows WIC')

SDR_JXR = (
    'SUm8AQgAAAALAAG8AQAQAAAAkgAAAAK8BAABAAAAAAAAAAS8BAABAAAAAAAAAIC8BAABAAAABAAAAIG8BAABAAAAAwAAAIK8'
    'CwABAAAAAADAQoO8CwABAAAAAADAQsC8BAABAAAAogAAAMG8BAABAAAAYwEAAMK8BAABAAAABQIAAMO8BAABAAAAjAAAAAAA'
    'AAAkw91vA07+S7GFPXd2jckPV01QSE9UTwARRMBxAAMAAmAAoAAKAACgAAAAAQAAAAkARP8ABEKAAAEAAAEBbX0pl6IAAAEC'
    '/KOFD40UKHUTJBvxvO7JSyxpkvhZgEsFbzas2ntpy4AIIaPCFEEHHDhFEBGjRtQYoiQiqXgTIAAAAQMjP8Ze9RAhBLEYRCMI'
    '/GEkRBIAYRMMIqGEQjCYcXGEc+EKwhthJxIIyIASjDCYimExxBCQlhJxIIyIASjsJOJBGRACUYYTEUwmOIISEDCYimExxBCQ'
    'm4ghI8NGEjhBAxlGEjxhMWMJBCFGE34wiPwmBwkDQQWYIghCwghtMIOQAVBBZgiCELoILMEhCFhBDaYQcgAgghuYQcgAsERj'
    'CWgwhQwnbGEkhhGIwksIKY4wmAGEzQwj0YUERhKngjyCbagmTmEskITjBhHmYTKMAFYxQTJzCWSEJxqCZOYSyQhOMGEeZhMo'
    'wAVjAYR5mEyjABWMAFdNUEhPVE8AEUTAAQADAAIAgCAIAAABAAAABgAf/wAEQoAAAQAAAQHp0AAAAQLxKIozURBEPxYUKE8k'
    'y0USmTllSQAAAAEDJ5gkKMIOIJyhAhhPQMIIIJHjGGOIEMJUiCSwwHlwiewngSCLiAEzAYT7UYUNMmEvAQwn2owoaZMJeAhh'
    'PtRhQ0yYS8BA'
)
INT16_JXR = (
    'SUm8AQgAAAALAAG8AQAQAAAAkgAAAAK8BAABAAAAAAAAAAS8BAABAAAAAAAAAIC8BAABAAAABAAAAIG8BAABAAAAAwAAAIK8'
    'CwABAAAAAADAQoO8CwABAAAAAADAQsC8BAABAAAAogAAAMG8BAABAAAANAIAAMK8BAABAAAA1gIAAMO8BAABAAAAzAAAAAAA'
    'AAAkw91vA07+S7GFPXd2jckWV01QSE9UTwARRMByAAMAAmAAAKAACgAAoAAAAAEAAAAPAKb/AARCgAABAAABAWFKLPhQoqYW'
    'u3ogAAABAvwtt4wrfBBX8DC0CjCrSEFZYIKywYV4YwphiCpOEFLoQUuhBQc4KDkE0m+Eg52y+B9OCofEFlfGE2IgpbiCwtDC'
    'Y0YTGiCgkIKrcYSmGEjBhIwYRh4RhhC0wfeWWWm2lsF8LCDHy4wsQogu2kQYmXGFc6MK50QWyEQX/6MKqAYU+BhT4GFETwoi'
    'YTdwxREhFVvBmAAAAQMjP8Ze/hNUIMAICDWgCMMjIRhoDfGGk/kQaBpBhlEsMMrzhhkWCMNidEQWJkYZ2v+GNfMMfG2GnyhI'
    'M0MiAGpkMMNi2amGx8cQQ0NKWGnyhIM0MiAGpkOw0+UJBmhkQA1MhhhsWzUw2PjiCGhpQMNi2amGx8cQQ0NKYLLEIMW3jCwB'
    'DCgYMNIhwgwsGMLYKMNIO4w2NoxhoQCEF4KDDcB0GGS6vhsIGYaDdaDDpswwAcQQxm2EGO+aYYoXIAX9dQYdNmGADiCGM26D'
    'DpswwAchDGbYQY75phihcgBf1wQY75zDFC5AC/rsGSsYw1rBDDGjQw3eWxhpOJGGZGIw0v7CDmV7jDYKwGGzo0MM+0xhwUJx'
    'hqwX4M+GINuG2g2Vk5hrRyQhuRjBhnw5mGy0owAdjwxQbKycw1o5IQ3IxqDZWTmGtHJCG5GMGGfDmYbLSjAB2PDAYZ8OZhst'
    'KMAHY8MAV01QSE9UTwARRMACAAMAAgAAgCAIAAABAAAACAA//wAEQoAAAQAAAQGFdO+gAAABAvC6nBhR5EFuiGFxMDChiILQ'
    '2ILQ2MLK8MJvhBYcBBWpiCtTEFRLwVEsFD7Zo7bTZ2DXMAAAAQMnmDQ5OMMSyiDcxNEGADhhvW0DDDCgg0jnjC8WhhebYgwA'
    'cMNVjYg0ySDC+nPC3i4ZQVWG8UEkGYFxADZroDDffWow4bmmTDXr4CGG++tRhw3NMmGvXwEMN99ajDhuaZMNevgI'
)
HDR_JXR = (
    'SUm8AQgAAAALAAG8AQAQAAAAkgAAAAK8BAABAAAAAAAAAAS8BAABAAAAAAAAAIC8BAABAAAAAgAAAIG8BAABAAAAAgAAAIK8'
    'CwABAAAAAADAQoO8CwABAAAAAADAQsC8BAABAAAAogAAAMG8BAABAAAAKwIAAMK8BAABAAAAzQIAAMO8BAABAAAA3AAAAAAA'
    'AAAkw91vA07+S7GFPXd2jckZV01QSE9UTwARRMB3AAEAAWAADQSgAAoAAKAAAAABAAAADwCg/wAEQoAAAQAAAQFhW8O4Jl/0'
    'FX56gAAAAQL4KzEQV54wmEEFUsIKxYYSpGEqRBSdEFO0YSMGEcDCOBhEfhEeEEsQvN4dbw7xwXyGMLVoMJTCC7jhhZbRhJEY'
    'SREFs4GFbIMI6GEVjCKxhC/CF4PBBIqvrHCnLhZFRBbPhhUijCwOiCzvDCmgMKaAwqkhBXUjCiiMJ/BhP4MJovCaLCTseMAz'
    'ixr7q4AAAAEDIz/GXvwZCKCDKvQYZDXCDFZggyLMGGK8hhivIQYC+EGFvhhg5cYX1kML6yGFy+eFy+YWYKgzlFSDLmUgBjP4'
    'INBktIM2ZSAGQ/1BnKKkGXMpADGf1BnKKkGXMpADGfwQaDJaQZsykAMh/gg0GS0gzZlIAZD/YN1kgYa5dBBtbMDDFDww1EyD'
    'DCcog1Y/DDCcowz1+GF+YjC7ZRhdsowtrXhbWsLCLQb0hcg28yyAGtf4MNTWNMNJmJBDM/6g3pC5Bt5lkANa/1BvSFyDbzLQ'
    'BrX+DDU1jTDSZiQQzP+DDU1jmGkzEghmf9w2f9xBksKMNgZcQY5lGGyN8MNIfww1pMCDBfww1pMDDSmWMM/mBhn8wMMpEfDK'
    'RHDD7Fhm6VMMsykIYv+CDY4yyDWZkQA0P/WGbpUwyzKQhi/6wzdKmGWZSEMX/BBscZZBrMyQBof+CDY4yyDWZkgDQ/8AV01Q'
    'SE9UTwARRMAHAAEAAQANBIAgCAAAAQAAAAgAPP8ABEKAAAEAAAEBhOyAAAAAAQLgtd4guWcYTRiCzPCC3PDCYIYTBCCuCEFj'
    'CGEnhhIYYSGGEWeEWYQmACAAAAIAIAAAAAEDJ3g0xvCDYBeGGUnwg0P/CDU/8MMg/hhkH8IMs/hBoD+GGH/hhgf4YYH+GF3P'
    'eF3PYWr6g1s/JBp/+IAZ3/BBs7/Ug1//EANH/1BrZ+SDT/8QAzv+oNbPyQaf/iAGd/wQbO/1INf/xADR/8EGzv9SDX/8QA0f'
    '/A=='
)
FIXED_HDR_JXR = (
    'SUm8AQgAAAAJAAG8AQAQAAAAegAAAAK8BAABAAAAAAAAAAS8BAABAAAAAAAAAIC8BAABAAAAAgAAAIG8BAABAAAAAgAAAIK8'
    'CwABAAAAAADAQoO8CwABAAAAAADAQsC8BAABAAAAigAAAMG8BAABAAAAiQEAAAAAAAAkw91vA07+S7GFPXd2jckSV01QSE9U'
    'TwARRMBzAAEAAXAAAAAAAQAAAA0AV/8ABEKAAAEAAAEBjRAQGKJkQdgAAAABAhRokpMSHJzIiys/wQ8Qw8IISIQcOECEQbbV'
    'Q6TbTZzbTvggBjRATOiOEkiyTen3rKaSj4EQQOjGEDtGUAgFKyl0Tr9u9IAAAAEDFHygoRcgIEnMDBkZ/jL34KAIgiQYTYiC'
    'dAQQ0YTGjCY0QTAiCAjCUwwkYMJGDCMPCMMIXQUgUgocEAJwggmkpBMKQAklQUgUgocEAJwqCkCkFDggBOEEE0lIJhSAEkgg'
    'mkpBMKQAklgnXGE0wgmuGEJGEwwwggglYMIIMJJjBDGGMM3m9QT1yCb4gBLYMJkqYS2IISJoJ65BN8QAltQT1yCb5AEtgwmS'
    'phLYghIkMJkswlsQQkTwnjEE74wm3EE2gwkWMJZDCOxBLAYR2MIrGEPGEPGEE8IJjWE1kwmLQhJYIKC2QTwkAJjLCayYTFoQ'
    'ktYTWTCYtCElggoLZBPCgCYwIKC2QTwoAmMA'
)


def fixture(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(base64.b64decode(payload))
    return str(path)


@pytest.mark.parametrize('extension', ['jxr', 'wdp', 'hdp'])
def test_8bit_jxr_decodes_to_rgba8_with_alpha(tmp_path, extension):
    path = fixture(tmp_path, f'source.{extension}', SDR_JXR)
    expected = np.array([
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [128, 64, 32, 128]],
        [[255, 255, 255, 255], [128, 128, 128, 255], [64, 64, 64, 255], [10, 20, 30, 64]],
        [[0, 0, 0, 0], [255, 128, 0, 255], [0, 128, 255, 255], [200, 100, 50, 200]],
    ], np.uint8)
    assert image_source.is_image(path)
    assert np.array_equal(image_source.read_rgba(path), expected)
    assert image_source.read_pixels(path).dtype == np.uint8
    probe = image_source.probe(path)
    assert (probe['image_width'], probe['image_height']) == (4, 3)
    assert (probe['width'], probe['height']) == (8, 8)
    assert probe['pix_fmt'] == 'rgba' and not probe['is_hdr']
    capture = image_source.Capture(path)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 1
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 4
    opened, frame = capture.read()
    assert opened and frame.shape == (3, 4, 3)
    assert not capture.read()[0]
    capture.release()


def test_high_depth_integer_jxr_is_not_silently_reduced(tmp_path):
    path = fixture(tmp_path, 'source.jxr', INT16_JXR)
    assert image_source.is_image(path)
    assert hdr_image_decode.probe_hdr(path) is None
    with pytest.raises(ValueError, match='不会自动降为 8-bit'):
        image_source.read_pixels(path)


def test_scrgb_float_jxr_keeps_linear_hdr_and_alpha(tmp_path):
    path = fixture(tmp_path, 'source.jxr', HDR_JXR)
    info = hdr_image_decode.probe_hdr(path)
    assert info['is_hdr'] and info['jxr_pixel_format'] == '128bppRGBAFloat'
    assert (info['width'], info['height']) == (2, 2)
    pixels = image_source.read_pixels(path)
    expected = np.array([
        [[2.0, 0.5, -0.25, 1.0], [0.0, 4.0, 0.0, 0.5]],
        [[0.25, 0.25, 8.0, 1.0], [0.1, 0.2, 0.3, 0.0]],
    ], np.float16)
    assert pixels.dtype == np.float16 and pixels.shape == (2, 2, 4)
    np.testing.assert_allclose(pixels.astype(np.float32), expected.astype(np.float32),
                               rtol=0, atol=1e-3)
    probe = image_source.probe(path)
    assert probe['is_hdr'] and probe['pix_fmt'] == 'rgba16f'
    assert probe['color_transfer'] == 'linear' and probe['bit_depth'] == 16
    assert probe['source_colour']['jxr_hdr_encoding'] == 'scRGB fixed/float'
    display = image_source.read_rgba(path)
    assert display.dtype == np.uint8 and display.shape == (2, 2, 4)
    assert display[:, :, 3].tolist() == [[255, 128], [255, 0]]


def test_scrgb_fixed_point_jxr_is_linear_hdr(tmp_path):
    path = fixture(tmp_path, 'source.jxr', FIXED_HDR_JXR)
    info = hdr_image_decode.probe_hdr(path)
    assert info['is_hdr'] and info['jxr_pixel_format'] == '48bppRGBFixedPoint'
    assert info['bit_depth'] == 16
    pixels = image_source.read_pixels(path)
    expected = np.array([
        [[1.0, 0.5, -0.25, 1.0], [2.0, 4.0, 0.0, 1.0]],
        [[0.25, 0.25, 4.0, 1.0], [0.1, 0.2, 0.3, 1.0]],
    ], np.float32)
    assert pixels.dtype == np.float16 and pixels.shape == (2, 2, 4)
    np.testing.assert_allclose(pixels.astype(np.float32), expected, rtol=0, atol=0.02)
    assert np.all(pixels[:, :, 3] == 1)


def test_hdr_jxr_rejects_non_scrgb_icc_profile():
    import jxr_image_decode as jxr

    def profile(description):
        tag = (b'desc' + b'\0' * 4 + struct.pack('>I', len(description) + 1)
               + description.encode('ascii') + b'\0')
        offset = 132 + 12
        return (b'\0' * 128 + struct.pack('>I', 1) + b'desc'
                + struct.pack('>II', offset, len(tag)) + tag)

    assert jxr._icc_description(profile('scRGB IEC 61966-2-2')).startswith('scRGB')
    assert jxr._hdr_profiles_supported([profile('scRGB')]) == (True, '')
    supported, description = jxr._hdr_profiles_supported([profile('sRGB IEC61966-2.1')])
    assert not supported and 'sRGB' in description


def test_corrupt_jxr_is_rejected(tmp_path):
    path = tmp_path / 'broken.jxr'
    path.write_bytes(b'not jpeg xr')
    assert image_source.is_image(str(path))
    with pytest.raises(ValueError, match='无法读取 JXR'):
        image_source.read_pixels(str(path))
