"""Software ProRes 4:2:2 10-bit decoder with a persistent CUDA P210 upload."""
from __future__ import annotations

import ctypes
from fractions import Fraction

import numpy as np


SOFTWARE_FORMAT_P210 = 3


def _value(value):
    if value is None:
        return 0
    return int(getattr(value, "value", value))


def is_prores(source):
    """Return true only for a ProRes stream in the supported planar format."""
    if not isinstance(source, dict):
        return False
    return (str(source.get("codec_name", "")).lower() == "prores" and
            str(source.get("pix_fmt", "")).lower() == "yuv422p10le")


def _plane_words(plane, rows, columns):
    stride = int(getattr(plane, "line_size", 0))
    if stride < columns * 2 or stride % 2:
        raise ValueError("ProRes plane stride is smaller than its 10-bit row")
    raw = np.frombuffer(bytes(plane), dtype="<u2")
    row_words = stride // 2
    if raw.size < rows * row_words:
        raise ValueError("ProRes plane buffer is shorter than its declared extent")
    return raw[:rows * row_words].reshape(rows, row_words)[:, :columns]


def pack_p210(frame):
    """Pack a decoded yuv422p10le VideoFrame into one uploadable RGBA8 view.

    PyAV's decoded 10-bit components are right-aligned uint16 values. P210 stores
    the same codes MSB-aligned. The returned byte view is Y bytes followed by
    interleaved UV bytes, with 2*w bytes per row in each half.
    """
    if str(frame.format.name) != "yuv422p10le":
        raise ValueError(f"expected yuv422p10le, got {frame.format.name}")
    width, height = int(frame.width), int(frame.height)
    if width <= 0 or height <= 0 or width % 2:
        raise ValueError("ProRes P210 requires positive dimensions and even width")
    if len(frame.planes) != 3:
        raise ValueError("yuv422p10le must contain Y, U and V planes")
    y = _plane_words(frame.planes[0], height, width)
    u = _plane_words(frame.planes[1], height, width // 2)
    v = _plane_words(frame.planes[2], height, width // 2)
    if np.any(y > 1023) or np.any(u > 1023) or np.any(v > 1023):
        raise ValueError("decoded yuv422p10le component exceeds 10 bits")
    words = np.empty((height * 2, width), dtype="<u2")
    words[:height] = y << 6
    words[height:] = np.stack((u, v), axis=-1).reshape(height, width) << 6
    return words.view(np.uint8).reshape(height, width, 4)


def _fraction(value):
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _rational(value):
    """Return numerator/denominator without losing FFmpeg metadata precision."""
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return int(value.numerator), int(value.denominator)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError("unrecognized mastering-display rational")


class ProResDecoder:
    """Decode ProRes frames and retain one CUDA P210 upload until next/close."""

    def __init__(self, path, source):
        if not is_prores(source):
            raise ValueError("source is not ProRes yuv422p10le")
        import av
        self.path = str(path)
        self.source = source
        self.container = av.open(self.path)
        streams = [stream for stream in self.container.streams if stream.type == "video"]
        if not streams:
            self.container.close()
            raise ValueError("ProRes source has no video stream")
        self.stream = streams[0]
        self._iterator = self._frames()
        self._upload = None
        self._shape = None
        self._cancelled = False
        self._closed = False
        self._decoded = 0
        self._host_upload_bytes = 0
        self._upload_count = 0
        self._static_metadata = None

    def _frames(self):
        for packet in self.container.demux(self.stream):
            for decoded in packet.decode():
                yield decoded

    def _copy_upload(self, data):
        from image_render import PixelUpload
        if self._upload is None:
            self._upload = PixelUpload(data)
            self._host_upload_bytes += data.nbytes
            self._upload_count += 1
            return
        if (data.shape[1], data.shape[0]) != (self._upload.width, self._upload.height):
            raise ValueError("ProRes frame dimensions changed during decode")
        contiguous = np.ascontiguousarray(data)
        with self._upload.current():
            PixelUpload.check(self._upload.cuda.cuMemcpyHtoD_v2(
                self._upload.pointer, contiguous.ctypes.data, contiguous.nbytes))
        self._host_upload_bytes += contiguous.nbytes
        self._upload_count += 1

    def _frame_colour_tag(self, decoded, frame_name, source_name, names):
        value = getattr(decoded, frame_name, None)
        if value is None:
            value = getattr(getattr(self.stream, "codec_context", None), frame_name, None)
        if value is not None:
            return _value(value)
        tag = str(self.source.get(source_name, "")).lower()
        return names.get(tag, 0)

    def next(self, output):
        if self._closed:
            raise RuntimeError("ProRes decoder is closed")
        if self._cancelled:
            return 0
        try:
            decoded = next(self._iterator)
        except StopIteration:
            return 0
        data = pack_p210(decoded)
        shape = (int(decoded.width), int(decoded.height))
        if self._shape is not None and shape != self._shape:
            raise ValueError("ProRes frame dimensions changed during decode")
        if decoded.pts is None:
            raise ValueError("ProRes frame has no presentation timestamp")
        time_base = _fraction(decoded.time_base or self.stream.time_base)
        if time_base is None or time_base.numerator <= 0 or time_base.denominator <= 0:
            raise ValueError("ProRes frame has no valid time base")
        self._copy_upload(data)
        self._shape = shape
        output.width, output.height = shape
        output.software_format = SOFTWARE_FORMAT_P210
        output.y_device_ptr = self._upload.pointer.value
        output.uv_device_ptr = self._upload.pointer.value + 2 * shape[0] * shape[1]
        output.y_pitch_bytes = output.uv_pitch_bytes = shape[0] * 2
        output.cuda_context = self._upload.context.value
        output.pts = int(decoded.pts)
        output.time_base_num, output.time_base_den = time_base.numerator, time_base.denominator
        output.color_range = self._frame_colour_tag(decoded, "color_range", "color_range",
            {"tv": 1, "mpeg": 1, "limited": 1, "pc": 2, "jpeg": 2, "full": 2})
        output.color_primaries = self._frame_colour_tag(decoded, "color_primaries", "color_primaries",
            {"bt709": 1, "bt2020": 9})
        output.color_transfer = self._frame_colour_tag(decoded, "color_trc", "color_transfer",
            {"bt709": 1, "smpte2084": 16, "arib-std-b67": 18})
        output.color_space = self._frame_colour_tag(decoded, "colorspace", "color_space",
            {"bt709": 1, "bt2020ncl": 9, "bt2020nc": 9})
        self._decoded += 1
        return 1

    def seek_us(self, relative_us):
        if self._closed:
            raise RuntimeError("ProRes decoder is closed")
        if int(relative_us) < 0:
            raise ValueError("seek position must be non-negative")
        time_base = _fraction(self.stream.time_base)
        if time_base is None or time_base <= 0:
            raise ValueError("ProRes stream has no valid time base")
        ticks = int(round(Fraction(int(relative_us) * time_base.denominator,
                                  1_000_000 * time_base.numerator)))
        start = int(getattr(self.stream, "start_time", 0) or 0)
        self.container.seek(start + ticks, stream=self.stream, backward=True, any_frame=False)
        self._iterator = self._frames()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def metadata(self):
        """Return source mastering/CLL metadata without decoding or advancing frames."""
        if self._closed:
            raise RuntimeError("ProRes decoder is closed")
        if self._static_metadata is None:
            import gpu_hdr_transfer_export as htx
            result = htx.HdrMetadata(ctypes.sizeof(htx.HdrMetadata))
            side_data = self.source.get("side_data_list") or []
            for side in side_data:
                kind = str(side.get("side_data_type", "")).lower()
                if "mastering" in kind:
                    if not self._parse_mastering(side, result):
                        raise ValueError("ProRes mastering display metadata is declared but unreadable")
                elif "content light" in kind:
                    if "max_content" not in side or "max_average" not in side:
                        raise ValueError("ProRes content light metadata is declared but unreadable")
                    result.has_content_light = 1
                    result.max_cll = int(side["max_content"])
                    result.max_fall = int(side["max_average"])
            self._static_metadata = result
        return self._static_metadata

    @staticmethod
    def _parse_mastering(data, result):
        """Parse ffprobe mastering-display fields, retaining decimal precision."""
        try:
            def frac(key):
                return _rational(Fraction(str(data[key])))
            for i, color in enumerate(("red", "green", "blue")):
                result.primary_x_num[i], result.primary_x_den[i] = frac(color + "_x")
                result.primary_y_num[i], result.primary_y_den[i] = frac(color + "_y")
            result.white_x_num, result.white_x_den = frac("white_point_x")
            result.white_y_num, result.white_y_den = frac("white_point_y")
            result.min_luminance_num, result.min_luminance_den = frac("min_luminance")
            result.max_luminance_num, result.max_luminance_den = frac("max_luminance")
        except (KeyError, ValueError, ZeroDivisionError, TypeError):
            return False
        result.has_mastering = result.mastering_has_primaries = result.mastering_has_luminance = 1
        return True

    def counters(self):
        return {"frames_decoded": self._decoded,
                "host_upload_bytes": self._host_upload_bytes,
                "host_upload_count": self._upload_count}

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._upload is not None:
            self._upload.close()
            self._upload = None
        self.container.close()

