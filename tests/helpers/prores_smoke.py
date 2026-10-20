"""Four small GPU frames, no model inference: ProRes input -> FP16 -> ProRes."""
import ctypes
import json
from pathlib import Path
import sys
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import av
import numpy as np
import hdr_gpu_engine
import native_effects
import pipeline
from prores_export import ProResOutput


def run(directory, modes=('sdr','hdr')):
    directory.mkdir(parents=True, exist_ok=False)
    staged = ROOT / 'native/build/hdr_colour/gpu_hdr_scrgb_bridge.dll'
    hdr_gpu_engine._library = hdr_gpu_engine.validate_library(ctypes.CDLL(str(staged)))
    evidence = []
    for name in modes:
        hdr = name == 'hdr'
        sdr8 = name == 'sdr8'
        width, height = (128, 128) if sdr8 else (64, 32)
        source, output = directory / f'{name}.mov', directory / f'{name}_output.mov'
        with av.open(str(source), 'w', format='mov') as container:
            stream = container.add_stream('libx264' if sdr8 else 'prores_ks', rate=Fraction(24))
            cc = stream.codec_context
            cc.width, cc.height, cc.pix_fmt = width, height, 'yuv420p' if sdr8 else 'yuv422p10le'
            cc.options = {} if sdr8 else {'profile': '3'}
            cc.color_range = 1
            cc.color_primaries = cc.colorspace = 9 if hdr else 1
            cc.color_trc = 16 if hdr else 1
            for i in range(2):
                frame = av.VideoFrame(width, height, cc.pix_fmt)
                for p, plane in enumerate(frame.planes):
                    values = np.full((plane.height, plane.width), 400 if p == 0 else 512, '<u2')
                    if p:
                        values[::2] += 12  # Vertical chroma detail must survive the P210 boundary.
                    if sdr8:
                        plane.update(np.full((plane.height,plane.line_size), 100 if p==0 else 128,np.uint8).tobytes())
                    else:
                        ProResOutput._put_plane(plane, values)
                frame.pts, frame.time_base = i, Fraction(1, 24)
                frame.color_range, frame.colorspace = 1, 9 if hdr else 1
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        settings = dict(render_pipeline=[], _pipeline_bypass=True, working_scale=1,
            reconstruction_mode='single', guidance_mode=0, dlss_backend='gpu_flow_d3d12',
            export_backend='gpu_transfer', experimental_hdr_scrgb=hdr, residual_pyramid=True,
            hdr_paper_white_scrgb=8, temporal_controls={'mode': 'off', 'scene_detection': False})
        tx = dict(codec='prores_ks', profile='hq', pixel_format='yuv422p10le', container='.mov',
            color_preset='preserve', audio_mode='remove', subtitle_mode='drop',
            copy_attachments=False, copy_metadata=False, copy_chapters=False)
        live = native_effects.make_live(width, height, settings, wide=not sdr8)
        diagnostics = {}
        try:
            pipeline.export_gpu_transfer_stream(str(source), live, 2, 24, 24, False,
                str(output), transcode_settings=tx, job_settings=settings, flow_provider=None, diagnostics_out=diagnostics)
        finally:
            live.close()
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            frames = list(container.decode(stream))
            assert stream.codec_context.profile == 'HQ'
            assert stream.codec_context.format.name == 'yuv422p10le'
            assert len(frames) == 2
            assert stream.codec_context.color_trc == (16 if hdr else 1)
            assert [f.pts * f.time_base for f in frames] == [Fraction(0), Fraction(1, 24)]
            chroma = np.frombuffer(bytes(frames[0].planes[1]), '<u2').reshape(height, -1)[:, :width//2]
            if not sdr8:
                assert abs(float(chroma[::2].mean()) - float(chroma[1::2].mean())) > 5
        evidence.append(dict(mode=name, frames=2, codec='prores', profile='HQ', pixel_format='yuv422p10le', diagnostics=diagnostics))
    (directory / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('PASS:', ', '.join(modes), 'GPU render / ProRes HQ output;', len(modes)*2, 'small frames')


if __name__ == '__main__':
    run(Path(sys.argv[1]), tuple(sys.argv[2:]) or ('sdr','hdr'))
