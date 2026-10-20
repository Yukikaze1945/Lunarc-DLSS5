"""Final SDR-to-HDR conversion and the existing Main10/P010 encoder boundary."""
import ctypes as C
import os


class _ImportedMaster:
    """Use the existing HDR reconstruction on an already converted FP16 master."""
    def _open_stages(self, frame):
        from hdr_gpu_engine import GpuHdrBridge
        self.context = int(frame.cuda_context)
        self.bridge = GpuHdrBridge(d3d12_device=self.live.d3d12_device(), cuda_context=self.context,
            full_width=self.src[0], full_height=self.src[1], working_width=self.working[0],
            working_height=self.working[1], paper_white_scrgb=self.policy['paper_white_scrgb'])
        self.contract = self.bridge.contract
        if getattr(self, 'extra_layers', None):
            self._require_multiscale_group()
        for layer in getattr(self, 'extra_layers', []):
            layer['bridge'] = GpuHdrBridge(d3d12_device=self.live.d3d12_device(), cuda_context=self.context,
                full_width=self.src[0], full_height=self.src[1], working_width=layer['dims'][0],
                working_height=layer['dims'][1], paper_white_scrgb=self.policy['paper_white_scrgb'])
        if hasattr(self, 'layer_session_ids'):
            self.layer_session_ids = [self.live.session_id] + [layer['live'].session_id for layer in self.extra_layers]
            if len(set(self.layer_session_ids)) != len(self.layer_session_ids):
                raise ValueError('HDR NR stages require distinct Feature 18 sessions')
            self.layer_papers = [self.bridge.paper_white] + [layer['bridge'].paper_white for layer in self.extra_layers]
        if getattr(self, 'multi', {}).get('fusion') == 'data_consistency_experimental':
            self._open_fusion()

    def _assert_metadata_stable(self):
        pass  # The upstream decoder owns source metadata; this stage doesn't decode.

    def _primary_colour(self, frame, colour, stages, reset=False):
        master = self.bridge.import_master(self.imported_color)
        self._preview_source = master.as_descriptor()
        working = self.bridge.downscale()
        proxy, guidance = self.bridge.proxy()
        return master, working, proxy, guidance, {'origin': 'ordered_truehdr_fp16'}, self.bridge.proxy_report()


class HdrNRStage:
    stage, wide, output_hdr = 'nr', True, True

    def __init__(self, live, settings, size, provider):
        from gpu_transfer_export import multi_scale_config
        from native_effects import nr_effect_settings
        self.live, self.settings, self.target = live, dict(settings), tuple(size)
        self.owner = live.owner
        self.multi = multi_scale_config(nr_effect_settings(settings), *size)
        if tuple(self.multi['dims'][0]) != (self.owner._w, self.owner._h):
            raise ValueError('TrueHDR NR session extent differs from the selected stage extent')
        self.provider = provider.new_independent_layer() if int(settings.get('guidance_mode', 0)) else None
        if self.provider:
            # Motion allocations/context outlive the primary Feature which
            # borrows them. The passthrough closes these after its anchor.
            live.stage_providers.append(self.provider)
        self.core = self.exchange = None
        self.calls = 0
        self.needs_reset = True

    def _open(self, frame):
        import gpu_hdr_transfer_export as htx
        import gpu_transfer_export as gtx
        from hdr_gpu_engine import load, probe_multiscale
        from post_control import GpuPostControl, controls_from_settings
        from temporal_control import GpuTemporalControl, temporal_from_settings
        base = htx.HdrMultiScaleChain if len(self.multi['scales']) > 1 else htx.HdrChain
        core_type = type('ImportedHdrNRChain', (_ImportedMaster, base), {})
        dims = dict(src_w=self.target[0], src_h=self.target[1], live_w=self.owner._w,
                    live_h=self.owner._h, dst_w=self.target[0], dst_h=self.target[1])
        # These are processing-domain constants, not claims about the SDR source.
        policy = dict(colour_mode='hdr10', paper_white_scrgb=float(self.settings.get('hdr_paper_white_scrgb', 8)),
                      support_ceiling_scrgb=125., clip_ceiling_scrgb=self.settings.get('truehdr_peak_nits', 650)/80.)
        extras = htx._open_extra_layers(self.multi, self.owner, self.provider, bool(self.provider),
                                       *self.target, settings=self.settings)
        kwargs = dict(flow_provider=self.provider, residual_strength=self.settings.get('residual_strength', 1),
                      uses_flow=bool(self.provider), preview_consumer=self._consume)
        if extras:
            kwargs.update(multi=self.multi, extra_layers=extras,
                          fusion_plans=[gtx.TapPlans(*self.target, *d) for d in self.multi['dims']])
        try:
            self.core = core_type({}, {'colour_bridge': {'multiscale': probe_multiscale(load())}},
                None, None, self.owner, {}, (30, 1), policy, dims, **kwargs)
            self.core.post_control = GpuPostControl(controls_from_settings(self.settings), hdr=True)
            self.core.temporal_control = GpuTemporalControl(temporal_from_settings(self.settings), hdr=True)
        except Exception:
            for layer in reversed(extras):
                layer['live'].close()
                if layer['flow_provider']: layer['flow_provider'].close_independent_layer()
            raise

    def _consume(self, color, frame):
        self.result = dict(color, output_hdr=True)

    def process(self, candidate, frame, reset):
        from native_effects import Exchange, frame_at_size
        if not candidate.get('output_hdr') or candidate.get('dxgi_format') != 10:
            raise ValueError('NR after TrueHDR requires linear FP16 HDR, not an SDR proxy picture')
        if self.core is None:
            self._open(frame)
            self.exchange = Exchange(self.live.d3d12_device(), frame.cuda_context, self.target, wide=True)
        self.core.imported_color = self.exchange.color(candidate)
        self.core.process(frame_at_size(frame, self.target), self.provider, self.calls, reset or self.needs_reset)
        self.needs_reset = False
        self.calls += 1
        for name in ('frame_reports', 'fusion_reports'):
            values = getattr(self.core, name, [])
            if len(values) > 16: del values[:-16]
        for values in getattr(self.core, 'layer_reset_flags', []):
            if len(values) > 16: del values[:-16]
        # HdrChain's consumer exposes a CUDA plane. The next spatial stage or
        # outer composite needs a complete D3D12 producer descriptor as well.
        result = self.exchange.texture(self.result['device_ptr'], self.result['pitch_bytes'])
        ptr, pitch = self.exchange.pointer()
        result.update(output_hdr=True, format='rgba16f_cuda', device_ptr=ptr,
                      pitch_bytes=pitch, cuda_context=int(frame.cuda_context))
        return result

    def report(self):
        return dict(stage='nr', calls=self.calls, processing_format='rgba16f', output_hdr=True,
                    layers=self.core.layer_diagnostics() if self.core and hasattr(self.core, 'layer_diagnostics') else
                        dict(layer_count=1, layer_session_ids=[self.owner.session_id], layer_working_dims=[self.multi['dims'][0]]),
                    frame_report=self.core.frame_reports[-1] if self.core and self.core.frame_reports else None)

    def update(self, settings):
        from gpu_transfer_export import multi_scale_config
        from native_effects import nr_effect_settings
        from post_control import controls_from_settings
        from temporal_control import temporal_from_settings
        self.settings = dict(settings)
        self.needs_reset = True
        self.multi = multi_scale_config(nr_effect_settings(settings), *self.target)
        if self.core:
            self.core.strength = float(settings.get('residual_strength', 1))
            self.core.post_control.update_controls(controls_from_settings(settings))
            self.core.temporal_control.update_controls(temporal_from_settings(settings))
            if hasattr(self.core, 'multi'):
                self.core.multi = self.multi
                if self.core.fusion and not self.core.fusion.update_parameters(self.multi, self.core.strength):
                    self.core.fusion.close(); self.core.fusion = None

    def close(self):
        if self.exchange:
            self.exchange.close(); self.exchange = None
        if self.core:
            # The outer passthrough owns the primary Feature; only extra NR
            # sessions and stage-local bridges belong to this imported chain.
            self.core.live = None
            self.core.flow_provider = None
            self.core.close(); self.core = None
        self.provider = None


class Tail:
    def __init__(self, live, settings, *, ordered=False):
        self.live, self.settings = live, dict(settings)
        self.effect = self.resize = self.bridge = self.encoder = self.encode_exchange = None
        self.lib = None
        self.ordered = ordered
        self.prores_output = False

    def update(self, settings):
        if self.effect:
            self.effect.update(settings)
        self.settings = dict(settings)

    def process(self, candidate, frame, *, at_stage=False):
        candidate = candidate.as_descriptor() if hasattr(candidate,'as_descriptor') else candidate
        if candidate.get('output_hdr') and not at_stage:
            return candidate
        if self.ordered and not at_stage:
            raise RuntimeError('Ordered TrueHDR output lost its HDR colour descriptor')
        if self.effect is None:
            from truehdr_engine import TrueHDR
            self.effect = TrueHDR(self.live.d3d12_device(),frame.cuda_context,
                (candidate.get('width',frame.width),candidate.get('height',frame.height)),self.settings,
                wide=candidate.get('dxgi_format') == 10 or candidate.get('format') == 'rgba16f_cuda')
        return self.effect.process(candidate,frame)

    def encode(self, chain, candidate, frame, pts=None):
        import gpu_transfer_export as gtx
        import gpu_hdr_transfer_export as htx
        from hdr_gpu_engine import GpuHdrBridge
        from vfx_framegen import encoder_timing
        output = self.process(candidate,frame)
        size = (output['width'],output['height'])
        target = tuple(getattr(chain,'export_size',getattr(chain,'output_size',size)))
        if size != target:
            if self.resize is None:
                from export_resize import Resize
                self.resize = Resize(self.live.d3d12_device(),frame.cuda_context,size,target,
                                     chain.export_scale_algorithm,wide=True)
            output = self.resize.texture(output)
        output['output_hdr'] = True
        if getattr(chain, 'prores_output', None) is not None:
            self.prores_output = True
            return chain.prores_output.encode(output, frame, pts)
        if self.bridge is None:
            self.bridge = GpuHdrBridge(d3d12_device=self.live.d3d12_device(),cuda_context=frame.cuda_context,
                full_width=target[0],full_height=target[1],working_width=target[0],working_height=target[1],paper_white_scrgb=1.,purpose='encode')
        if self.encoder is None:
            self.lib = chain.b['nvenc']
            bound = gtx._bind_optional(self.lib,htx._HDR_NVENC)
            if 'nvenc_hdr_open' not in bound or 'nvenc_encode_p010' not in bound:
                raise RuntimeError('TrueHDR export requires the existing HDR10 NVENC bridge')
            struct = chain.plan['struct']
            options = htx.HdrOptions(struct_size=C.sizeof(htx.HdrOptions))
            for key in ('codec','preset','tune','multipass','rate_control','constant_quality','bitrate','gop'):
                setattr(options,key,struct.get(key))
            options.profile = struct['profile']
            options.extra_options = gtx.extra_options_blob([(k,v) for k,v in chain.plan['av_options']
                                                           if k not in htx.LOCKED_HDR_AV_OPTIONS]) or None
            rate,tb = encoder_timing({} if getattr(chain,'ordered_framegen',None) else getattr(chain,'framegen_settings',{}),
                                    getattr(chain,'_encoder_fps',chain.fps),(frame.time_base_num,frame.time_base_den))
            handle = C.c_void_p()
            rc = htx.open_p010_encoder(self.lib,False,frame.cuda_context,*target,*rate,*tb,os.fsencode(chain.encoder_output),
                                         C.byref(options),None,C.byref(handle))
            if rc: gtx._bridge_error(self.lib,'nvenc_last_error',rc)
            self.encoder = handle
        color = output
        if 'device_ptr' not in color:
            if self.encode_exchange is None:
                from native_effects import Exchange
                self.encode_exchange = Exchange(self.live.d3d12_device(),frame.cuda_context,target,wide=True)
            color = self.encode_exchange.color(output)
        self.bridge.import_master(color)
        texture = output
        if 'd3d12_resource' not in texture:
            if self.encode_exchange is None:
                from native_effects import Exchange
                self.encode_exchange = Exchange(self.live.d3d12_device(), frame.cuda_context, target, wide=True)
            texture = self.encode_exchange.texture(color['device_ptr'], color['pitch_bytes'])
        self.bridge.set_composite(texture)
        planes = self.bridge.encode(clip_ceiling_scrgb=self.settings.get('truehdr_peak_nits',650)/80.)
        rc = self.lib.nvenc_encode_p010(self.encoder,planes.y.device_ptr,planes.y.pitch_bytes,
                                       planes.uv.device_ptr,planes.uv.pitch_bytes,int(frame.pts if pts is None else pts))
        if rc: gtx._bridge_error(self.lib,'nvenc_last_error',rc)
        return planes,self.bridge.encode_report()

    def finish(self):
        if self.encoder:
            from gpu_transfer_export import _bridge_error
            code = self.lib.nvenc_finish(self.encoder)
            if code: _bridge_error(self.lib,'nvenc_last_error',code)

    def report(self):
        contract = counters = None
        if self.encoder:
            from gpu_hdr_transfer_export import HdrContract, HdrCounters, _fields
            block = HdrContract(struct_size=C.sizeof(HdrContract))
            if not self.lib.nvenc_hdr_get_contract(self.encoder,C.byref(block)):
                contract = _fields(block)
            counts = HdrCounters(struct_size=C.sizeof(HdrCounters))
            if not self.lib.nvenc_hdr_get_counters(self.encoder,C.byref(counts)):
                counters = _fields(counts)
        return dict(effect=self.effect.report() if self.effect else None,encoder_contract=contract,encoder_counters=counters,
                    output_contract=('ProRes 422 HQ / yuv422p10le / BT.2020 / PQ / limited range' if self.prores_output else
                        (f"{contract['codec']} / {contract['profile_name']} / P010 / BT.2020 / PQ / limited range" if contract else 'P010 / BT.2020 / PQ / limited range')),
                    static_metadata=('not authored by the ProRes software encoder' if self.prores_output else
                        'Source mastering/CLL are not reused after TrueHDR; no unmeasured output metadata invented'))

    def close(self):
        if self.encoder:
            self.lib.nvenc_close(self.encoder); self.encoder=None
        for name in ('bridge','encode_exchange','resize','effect'):
            item = getattr(self,name,None)
            if item: item.close(); setattr(self,name,None)
