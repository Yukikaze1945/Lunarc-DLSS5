"""Frame-index selections on the existing presentation timeline, end exclusive."""
import ctypes as C
import math


def validate_range(value, total=None):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('选区须包含起止帧')
    start, end = value.get('start_frame'), value.get('end_frame')
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start
            or (total is not None and end > total)):
        raise ValueError('选区帧范围无效，请选择素材内的起止帧')
    return dict(start_frame=start, end_frame=end)


class SelectedDecoder:
    """Seek to CFR selection, discard GOP preroll, rebase selected display PTS.

    Nonuniform PTS are rejected for selections: nominal frame indices cannot
    describe a VFR interval exactly. Full-source exports retain their old policy.
    """
    def __init__(self, chain, selection, fps, cancel=None):
        self.chain, self.selection, self.cancel = chain, selection, cancel
        self.fps = float(fps[0]) / float(fps[1]) if isinstance(fps, (tuple, list)) else float(fps)
        self.origin = self.first = self.previous = None
        self.emitted = 0
        self.source_first_seconds = None

    def next(self, frame):
        if self.selection is None:
            return self.chain.next_frame(frame)
        if self.origin is None:
            code = self.chain.next_frame(frame)
            if code <= 0:
                return code
            self.origin = int(frame.pts) * frame.time_base_num / frame.time_base_den
            if self.selection['start_frame']:
                seek_us = round(self.selection['start_frame'] / self.fps * 1e6)
                if hasattr(self.chain, 'seek_decoder'):
                    code = int(self.chain.seek_decoder(seek_us))
                else:
                    lib = self.chain.b['nvdec']
                    lib.nvdec_bridge_seek_us.argtypes = [C.c_void_p, C.c_int64]
                    lib.nvdec_bridge_seek_us.restype = C.c_int32
                    code = int(lib.nvdec_bridge_seek_us(self.chain.decoder, seek_us))
                if code:
                    raise RuntimeError(f'选区定位失败 ({code})')
                code = self.chain.next_frame(frame)
        else:
            code = self.chain.next_frame(frame)
        while code > 0:
            if self.cancel and self.cancel():
                import pipeline
                raise pipeline.JobCancelled('用户已取消任务')
            seconds = int(frame.pts) * frame.time_base_num / frame.time_base_den - self.origin
            target = self.selection['start_frame'] / self.fps
            if seconds + .5 / self.fps >= target:
                expected = (self.selection['start_frame'] + self.emitted) / self.fps
                tolerance = max(1.01 * frame.time_base_num / frame.time_base_den, 1e-6)
                if abs(seconds - expected) > tolerance:
                    raise ValueError('选区含非连续或可变帧率时间戳，请先转为恒定帧率')
                if self.first is None:
                    self.first = int(frame.pts)
                    self.source_first_seconds = seconds
                self.previous = int(frame.pts)
                self.emitted += 1
                frame.pts = int(frame.pts) - self.first
                return code
            code = self.chain.next_frame(frame)
        return code

    def mux_settings(self, settings, ffprobe):
        if self.selection is None:
            return settings
        import json
        import subprocess
        metadata = json.loads(subprocess.check_output([ffprobe, '-v', 'error',
            '-show_entries', 'format=start_time', '-of', 'json', self.chain.video], timeout=10))
        start = float((metadata.get('format') or {}).get('start_time') or 0)
        # FFmpeg input -ss is relative to the container origin, whereas the UI
        # frame indices and NVDEC seek are relative to the first video frame.
        offset = self.origin + self.source_first_seconds - start
        if not math.isfinite(offset) or offset < -1e-6:
            raise ValueError('素材时间轴无法对齐选区音轨')
        return dict(settings or {}, _source_range_start_seconds=max(0., offset))
