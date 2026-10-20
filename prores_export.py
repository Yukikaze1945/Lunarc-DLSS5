"""Explicit CPU ProRes codec boundary around the existing GPU render chain.

Wide output is GPU-converted directly to P210, never via 4:2:0. Frame pixels
cross to host for the software encoder and are accounted separately from NVENC.
"""
import ctypes as C
from fractions import Fraction

import numpy as np

CODEC = 'prores_ks'


def encoder_plan(settings, source=None):
    settings = dict(settings or {})
    unsupported = []
    if settings.get('container', '.mov') not in ('.mov','mov'):
        unsupported.append('ProRes 422 HQ 使用 MOV 封装')
    if str(settings.get('profile','auto')).lower() not in ('auto','hq','3'):
        unsupported.append('ProRes 422 HQ 的 profile 固定为 HQ（3）')
    if settings.get('pixel_format','auto') not in ('auto','yuv422p10le'):
        unsupported.append('ProRes 422 HQ 使用 yuv422p10le，不能使用 P010 或 8-bit 输出')
    if str(settings.get('extra_args') or '').strip():
        unsupported.append('ProRes 固定编码入口不接受额外 FFmpeg 参数')
    try:
        import av
        av.codec.Codec(CODEC,'w')
    except Exception as error:
        unsupported.append(f'当前视频组件缺少 ProRes 软件编码器：{error}')
    return dict(codec=CODEC,profile='hq',pixel_format='yuv422p10le',
        unsupported=unsupported,av_options=[],tune_adjustment=None,
        struct=dict(codec=CODEC.encode(),profile=b'hq',preset=None,tune=None,
            multipass=None,rate_control=None,constant_quality=0,bitrate=0,gop=1,color_matrix=1))


class ProResOutput:
    def __init__(self, chain):
        self.chain=chain
        self.container=self.stream=self.bridge=self.exchange=None
        self.frames=self.readback_bytes=0
        self.first_pts=None
        self.finished=False
        self.output_hdr=None

    def _open(self, frame, size, hdr):
        import av
        from vfx_framegen import encoder_timing
        chain=self.chain
        rate,tb=encoder_timing({} if getattr(chain,'ordered_framegen',None) else
            getattr(chain,'framegen_settings',{}),getattr(chain,'_encoder_fps',chain.fps),
            (frame.time_base_num,frame.time_base_den))
        self.time_base=Fraction(*tb)
        self.size=tuple(size)
        self.container=av.open(chain.encoder_output,'w',format='mov')
        self.stream=self.container.add_stream(CODEC,rate=Fraction(*rate))
        self.stream.width,self.stream.height=size
        self.stream.pix_fmt='yuv422p10le'
        cc=self.stream.codec_context
        cc.time_base=self.time_base
        cc.thread_count=2
        cc.options={'profile':'3','vendor':'apl0'}
        cc.color_range=1
        cc.color_primaries=9 if hdr else 1
        cc.color_trc=16 if hdr else 1
        cc.colorspace=9 if hdr else 1
        self.output_hdr=hdr

    @staticmethod
    def _put_plane(plane, values):
        padded=np.zeros((plane.height,plane.line_size//2),dtype='<u2')
        padded[:values.shape[0],:values.shape[1]]=values
        plane.update(padded.tobytes())

    def _wide_frame(self, descriptor, frame, hdr):
        import av
        from native_effects import Exchange
        from hdr_gpu_engine import GpuHdrBridge,GpuSdr10Bridge
        from gpu_preview_engine import _Copy2D
        color=dict(descriptor)
        size=(color['width'],color['height'])
        if self.exchange is None:
            self.exchange=Exchange(self.chain.live.d3d12_device(),frame.cuda_context,size,wide=True)
        if 'device_ptr' not in color:color=self.exchange.color(color)
        texture=self.exchange.texture(color['device_ptr'],color['pitch_bytes'])
        if self.bridge is None:
            cls=GpuHdrBridge if hdr else GpuSdr10Bridge
            self.bridge=cls(d3d12_device=self.chain.live.d3d12_device(),cuda_context=frame.cuda_context,
                full_width=size[0],full_height=size[1],working_width=size[0],working_height=size[1],
                paper_white_scrgb=8.0 if hdr else 1.0,purpose='encode422')
        self.bridge.import_master(color)
        self.bridge.set_composite(texture)
        planes=self.bridge.encode(clip_ceiling_scrgb=125.0 if hdr else 1.0)
        arrays=[]
        cuda=C.WinDLL('nvcuda.dll')
        cuda.cuCtxPushCurrent_v2.argtypes=[C.c_void_p]
        cuda.cuCtxPopCurrent_v2.argtypes=[C.POINTER(C.c_void_p)]
        cuda.cuMemcpy2D_v2.argtypes=[C.POINTER(_Copy2D)]
        from image_render import PixelUpload
        PixelUpload.check(cuda.cuCtxPushCurrent_v2(frame.cuda_context))
        try:
            for plane in (planes.y,planes.uv):
                pixels=np.empty((size[1],size[0]),dtype='<u2')
                copy=_Copy2D(srcMemoryType=2,srcDevice=plane.device_ptr,srcPitch=plane.pitch_bytes,
                    dstMemoryType=1,dstHost=pixels.ctypes.data,dstPitch=pixels.strides[0],
                    WidthInBytes=pixels.strides[0],Height=size[1])
                PixelUpload.check(cuda.cuMemcpy2D_v2(C.byref(copy)))
                arrays.append(pixels>>6)
                self.readback_bytes+=pixels.nbytes
        finally:
            popped=C.c_void_p();PixelUpload.check(cuda.cuCtxPopCurrent_v2(C.byref(popped)))
        output=av.VideoFrame(*size,'yuv422p10le')
        for plane,values in zip(output.planes,(arrays[0],arrays[1][:,0::2],arrays[1][:,1::2])):
            self._put_plane(plane,values)
        return output

    def encode(self, color, frame, pts=None):
        import av
        chain=self.chain
        color=color.as_descriptor() if hasattr(color,'as_descriptor') else dict(color)
        color.setdefault('width',int(frame.width));color.setdefault('height',int(frame.height))
        color.setdefault('cuda_context',int(frame.cuda_context))
        hdr=bool(color.get('output_hdr',not getattr(chain,'sdr10',True)))
        wide=color.get('dxgi_format')==10 or color.get('format')=='rgba16f_cuda'
        if self.stream is None:self._open(frame,(color['width'],color['height']),hdr)
        if (color['width'],color['height']) != self.size:
            raise ValueError('ProRes 导出期间画面尺寸发生变化')
        if hdr!=self.output_hdr:raise ValueError('ProRes 导出期间输出色彩域发生变化')
        if wide:
            output=self._wide_frame(color,frame,hdr)
        else:
            from image_render import download
            pixels,_=download(dict(color,output_hdr=False),frame)
            self.readback_bytes+=pixels.nbytes
            # SDR8 is still an 8-bit source, encoded into the real 10-bit 4:2:2 codec.
            output=av.VideoFrame.from_ndarray(pixels,format='rgba').reformat(
                format='yuv422p10le',dst_colorspace='ITU709',src_color_range='JPEG',dst_color_range='MPEG')
        timestamp=int(frame.pts if pts is None else pts)
        if self.first_pts is None:self.first_pts=timestamp
        output.pts=timestamp-self.first_pts;output.time_base=self.time_base
        output.color_range=1;output.colorspace=9 if hdr else 1
        for packet in self.stream.encode(output):self.container.mux(packet)
        self.frames+=1
        return None,dict(codec=CODEC,profile='HQ',pixel_format='yuv422p10le')

    def finish(self):
        if self.finished:return
        if self.container:
            for packet in self.stream.encode():self.container.mux(packet)
            self.container.close();self.container=None
        self.finished=True

    def report(self):
        return dict(codec=CODEC,profile='HQ',pixel_format='yuv422p10le',
            frames_submitted=self.frames,frame_host_readback_bytes=self.readback_bytes,
            encoder='CPU/PyAV/libavcodec',zero_copy=False,
            color_range='limited',color_primaries='bt2020' if self.output_hdr else 'bt709',
            color_transfer='smpte2084' if self.output_hdr else 'bt709',
            color_matrix='bt2020nc' if self.output_hdr else 'bt709',
            hdr_static_metadata='not authored; color primaries/transfer/matrix are encoded')

    def close(self):
        from gpu_cleanup import finish_cleanup
        errors=[]
        for name in ('bridge','exchange','container'):
            owner=getattr(self,name,None);setattr(self,name,None)
            if owner:
                try:owner.close()
                except Exception as error:errors.append((name,error))
        finish_cleanup(errors)
