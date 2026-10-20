/**
 * Finding 4, at the level the panel actually decides it: the preview builder is a pure
 * function of (export chain, settings), so its behaviour is testable without a GPU, a
 * browser or a running backend. Run with:
 *
 *   node --test ../tests/node/
 *
 * Node type-strips the .ts sources; the specifiers carry extensions because that is what
 * Node requires, and frontend/tsconfig.app.json only type-checks src/.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  buildExportPreview,
  buildGpuTransferPreview,
  buildHdrTransferPreview,
  buildLegacyCommand,
  composeJobOutputPath,
  sameOutputAsInput,
} from '../../frontend/src/services/exportPreview.ts'
import {
  HDR_CLIP_CEILING_SCRGB, HDR_DERIVED_LEGAL_PEAK_SCRGB, HDR_SUPPORT_CEILING_SCRGB,
  paperWhiteLabel,
} from '../../frontend/src/services/hdrExport.ts'
import {
  chooseReconstructionMode, chooseWorkingScale, chooseLayerScale,
} from '../../frontend/src/services/multiScaleSettings.ts'

const SETTINGS = {
  codec: 'av1_nvenc',
  container: '.mkv',
  suffix: '_AV1',
  rate_mode: 'cq',
  cq: 16,
  video_bitrate: '80M',
  qp: 20,
  peak_policy: 'auto',
  maxrate: '380M',
  bufsize: '1520M',
  preset: 'p7',
  tune: 'uhq',
  multipass: 'fullres',
  gpu: '0',
  decode_mode: 'auto',
  finish_action: 'none',
  gop_mode: 'auto',
  gop: 120,
  b_frames: 'auto',
  b_ref_mode: 'auto',
  lookahead: 'auto',
  lookahead_level: 'auto',
  aq_mode: 'auto',
  aq_strength: 0,
  weighted_prediction: false,
  temporal_filter: 'auto',
  split_encode: 'auto',
  profile: 'auto',
  level: 'auto',
  tier: 'auto',
  resolution_mode: 'original',
  custom_width: 0,
  custom_height: 0,
  custom_factor: 1,
  keep_aspect: true,
  scale_algorithm: 'lanczos',
  pixel_format: 'auto',
  color_preset: 'preserve',
  custom_color_space: '',
  custom_color_primaries: '',
  custom_color_transfer: '',
  custom_color_range: '',
  audio_mode: 'copy',
  audio_bitrate: '320k',
  subtitle_mode: 'copy',
  copy_attachments: true,
  copy_metadata: true,
  copy_chapters: true,
  overwrite_output: false,
  skip_existing: true,
  extra_args: '',
}

const legacyInput = {
  inputPath: 'C:\\Videos\\clip.mkv',
  outputPath: 'C:\\Videos\\clip_dlss.mkv',
  pipeWidth: 1920,
  pipeHeight: 1080,
  sourceFps: 120,
  ts: SETTINGS,
}

const gpuInput = {
  ts: SETTINGS,
  guidanceMode: 0,
  flowEngine: 'nvof',
  workingScale: 0.5,
  residualPyramid: true,
  sourceFps: 120,
  outputDir: 'C:\\Videos',
  fileName: 'clip_dlss',
  fallbackStem: 'clip',
}

const gpuLines = (overrides = {}) =>
  buildGpuTransferPreview({ ...gpuInput, ...overrides }).lines.join('\n')

test('offline VFG fps and final-frame hold policy appear in every precision path', () => {
  for (const sdr10 of [false, true]) {
    for (const renderPipeline of [['nr'], ['dlaa', 'nr', 'vsr']]) {
      const text = gpuLines({frameGeneration:'vfg', frameMultiplier:3, sdr10, renderPipeline})
      assert.match(text, /NVIDIA VFG 3× · 120 → 360 fps/)
      assert.match(text, /片尾补齐最后一帧/)
      assert.match(text, /实时预览保持原帧率/)
    }
  }
  const hdr = buildHdrTransferPreview({...gpuInput, requested:true, paperWhiteScrgb:8,
    sourceWidth:1920,sourceHeight:1080,renderPipeline:['nr','vsr'],
    frameGeneration:'vfg',frameMultiplier:4}).lines.join('\n')
  assert.match(hdr, /NVIDIA VFG 4× · 120 → 480 fps/)
  assert.match(hdr, /RGB10A2 \/ PQ \/ BT.2020/)
})

test('16x VFG is reflected in SDR8, SDR10 and HDR export plans',()=>{
  for(const sdr10 of [false,true]){
    assert.match(gpuLines({frameGeneration:'vfg',frameMultiplier:16,sdr10}),/NVIDIA VFG 16× · 120 → 1920 fps/)
  }
  const hdr=buildHdrTransferPreview({...gpuInput,requested:true,paperWhiteScrgb:8,
    sourceWidth:1920,sourceHeight:1080,renderPipeline:['vfg'],
    frameGeneration:'vfg',frameMultiplier:16}).lines.join('\n')
  assert.match(hdr,/NVIDIA VFG 16× · 120 → 1920 fps/)
})

test('SDR10 preview describes BT.709 FP16 processing and independent layer scales', () => {
  const preview = buildGpuTransferPreview({ ...gpuInput, sdr10: true,
    ts: { ...SETTINGS, codec: 'hevc_nvenc' }, reconstructionMode: 'dual', layerScales: [.5, .75] })
  assert.equal(preview.kind, 'gpu-transfer-chain')
  assert.equal(preview.copyable, false)
  assert.equal(preview.command, '')
  const text = preview.lines.join('\n')
  assert.match(text, /linear Rec.709 FP16/)
  assert.match(text, /HEVC Main10/)
  assert.match(text, /50% \/ 75%/)
  assert.doesNotMatch(text, /NV12|8-bit/)
})

// --- the two chains are shown as what they are --------------------------------

test('the legacy chain is previewed as the command it runs', () => {
  const preview = buildExportPreview('legacy', legacyInput, gpuInput)

  assert.equal(preview.kind, 'legacy-command')
  assert.equal(preview.copyable, true)
  assert.match(preview.command, /-f rawvideo -pix_fmt bgr24/)
  assert.match(preview.command, /-s 1920x1080 -r 120\.000000 -i -/)
  assert.match(preview.command, /-map 1:a\?/, 'audio comes from the file input')
  assert.doesNotMatch(preview.command, /-map 0:a/, 'never from the pipe')
  assert.match(preview.command, /"C:\\Videos\\clip_dlss\.mkv"$/)
})

test('auto is the legacy chain, so its preview is the command too', () => {
  for (const backend of ['auto', '', undefined, null]) {
    assert.equal(buildExportPreview(backend, legacyInput, gpuInput).kind, 'legacy-command')
  }
})

test('gpu transfer is previewed as a device chain, never as a command', () => {
  const preview = buildExportPreview('gpu_transfer', legacyInput, gpuInput)
  const text = preview.lines.join('\n')

  assert.equal(preview.kind, 'gpu-transfer-chain')
  assert.equal(preview.command, '', 'no runnable string is offered')
  assert.equal(preview.copyable, false)
  assert.doesNotMatch(text, /-f rawvideo -pix_fmt bgr24/, 'the pipe signature is gone')
  assert.ok(!preview.lines.some((line) => /^\s*ffmpeg(\.exe)?\s/.test(line)),
    'nothing in the preview starts as a shell command')
  assert.match(text, /NVDEC/)
  assert.match(text, /NVENC/)
  assert.match(text, /不经过 FFmpeg 的 rawvideo 管道/)
})

test('independent layer controls keep unique scales and preview the real fusion request', () => {
  const settings = { working_scale: 0.5, layer_scales: [0.5],
    layer_weights: [1], reconstruction_mode: 'single', export_backend: 'auto' }
  const dual = chooseReconstructionMode(settings, 'dual')
  assert.deepEqual(dual.layer_scales, [0.5, 0.5])
  assert.equal(dual.export_backend, 'gpu_transfer')
  const changed = chooseWorkingScale({ ...settings, ...dual }, 0.75)
  assert.deepEqual(changed.layer_scales, [0.75, 0.5])
  assert.deepEqual(chooseLayerScale({ ...settings, ...dual }, 1, 0.5),[0.5,0.5])
  const text = gpuLines({ reconstructionMode: 'dual', layerScales: [0.5, 0.75],
    layerWeights: [1, 0.8], fusionMode: 'data_consistency_experimental',
    fusionBeta: 0.1, fusionMaxIterations: 128,
    fusionRelativeTolerance: 0.001 })
  assert.match(text, /第 1 层：50% · 权重 1/)
  assert.match(text, /第 2 层：75% · 权重 0.8/)
  assert.match(text, /GPU data-consistency CG 融合/)
  assert.match(text, /未收敛即失败/)
})

test('switching the chain changes the preview immediately', () => {
  const before = buildExportPreview('auto', legacyInput, gpuInput)
  const after = buildExportPreview('gpu_transfer', legacyInput, gpuInput)

  assert.notEqual(before.kind, after.kind)
  assert.notEqual(after.copyable, before.copyable)
})

// --- the parameters the device chain is asked for -----------------------------

test('the device chain preview reports the requested encoder parameters', () => {
  const text = gpuLines()

  assert.match(text, /av1_nvenc · preset p7/)
  assert.match(text, /CQ 16/)
  assert.match(text, /源帧率 × 2 = 240 帧/, 'the auto GOP is the source rate, not a guess')
})

test('uhq on h264 is stated as a downgrade the backend always applies', () => {
  const text = gpuLines({ ts: { ...SETTINGS, codec: 'h264_nvenc', tune: 'uhq' } })

  assert.match(text, /请求 uhq → 实际必定下发 hq/)
  assert.doesNotMatch(text, /实际下发 uhq/)
})

test('uhq on av1 is stated as capability-dependent, not as always applied', () => {
  const text = gpuLines()

  assert.match(text, /请求 uhq/)
  assert.match(text, /取决于后端对 av1_nvenc 编码器能力的探测/)
  assert.match(text, /实际值以任务诊断日志为准/)
  assert.doesNotMatch(text, /实际下发 uhq/)
})

test('a pixel format the bridge cannot submit is named as a refusal', () => {
  const text = gpuLines({ ts: { ...SETTINGS, pixel_format: 'p010le' } })

  assert.match(text, /p010le/)
  assert.match(text, /后端会拒绝该任务/)
  assert.doesNotMatch(text, /自动回退到 Legacy 链路.*成功/, 'a refusal is not a fallback')
})

test('the residual and flow stages describe the mode the panel is in', () => {
  const guided = gpuLines({ guidanceMode: 1 })
  assert.match(guided, /NVOF 设备光流（flow_engine=nvof）/)

  const unguided = gpuLines({ guidanceMode: 0 })
  assert.match(unguided, /零运动纹理/)
  assert.doesNotMatch(unguided, /NVOF 设备光流/)

  const noResidual = gpuLines({ residualPyramid: false })
  assert.match(noResidual, /后端因此会拒绝该任务/)
})

test('a container that is not the bridge output names the copy pass', () => {
  const text = gpuLines({ ts: { ...SETTINGS, container: '.mp4' } })

  assert.match(text, /NVENC 桥只能写出单视频流的 \.mkv/)
  assert.match(text, /-c:v copy 重封装/)
  assert.match(text, /视频不做二次编码/)
})

test('the device chain preview claims no speed and no zero copy', () => {
  const text = gpuLines()

  for (const claim of ['更快', '加速', '性能提升', '零拷贝', '性能增益']) {
    if (claim === '性能增益') {
      assert.match(text, new RegExp(`不声明任何速度或质量增益`), 'stated as a denial')
      continue
    }
    assert.doesNotMatch(text, new RegExp(claim), `preview advertises ${claim}`)
  }
  assert.match(text, /NOT_CHARACTERIZED/)
})

// --- the same-file prevention, and its limits ---------------------------------

test('a destination that is the source is caught before the request leaves the page', () => {
  const input = 'C:\\Videos\\clip.mkv'

  const identical = sameOutputAsInput(input, 'C:\\Videos', 'clip', '.mkv', 'clip')
  const otherCase = sameOutputAsInput(input, 'c:\\videos', 'CLIP', '.mkv', 'clip')
  const slashed = sameOutputAsInput('C:/Videos/clip.mkv', 'C:\\Videos', 'clip', '.mkv', 'clip')
  const dotSegment = sameOutputAsInput(input, 'C:\\Videos\\.\\', 'clip', '.mkv', 'clip')
  const dotDot = sameOutputAsInput(input, 'C:\\Videos\\sub\\..', 'clip', '.mkv', 'clip')

  for (const refused of [identical, otherCase, slashed, dotSegment, dotDot]) {
    assert.equal(refused, true, 'these spellings are one file')
  }
})

test('a valid target is never refused', () => {
  const input = 'C:\\Videos\\clip.mkv'

  assert.equal(sameOutputAsInput(input, 'C:\\Videos', 'clip_dlss', '.mkv', 'clip'), false)
  assert.equal(sameOutputAsInput(input, 'C:\\Other', 'clip', '.mkv', 'clip'), false)
  assert.equal(sameOutputAsInput(input, 'C:\\Videos', 'clip', '.mp4', 'clip'), false,
    'a different container is a different file, which is a valid way out')
  assert.equal(sameOutputAsInput(input, '', 'clip', '.mkv', 'clip'), false,
    'no directory means the backend derives a default beside the source')
})

test('the composed job path is the one the backend is handed', () => {
  assert.equal(composeJobOutputPath('C:\\Videos\\', '', '.mkv', 'clip'),
    'C:\\Videos\\clip.mkv', 'the file name falls back to the source stem')
  assert.equal(composeJobOutputPath('C:\\Videos\\\\', '  name  ', '.mp4', 'clip'),
    'C:\\Videos\\name.mp4')
  assert.equal(composeJobOutputPath('', 'name', '.mkv', 'clip'), '')
})

test('the legacy builder mirrors the backend overwrite and tune rules', () => {
  const command = buildLegacyCommand(legacyInput)

  // The rules the backend applies, mirrored rather than reinvented: never truncate when
  // the job says not to, and drop the tuned flag only where the codec has no such tune.
  assert.match(command, /^ffmpeg -n /, 'skip_existing keeps -n')
  assert.match(command, /-tune uhq/, 'av1 keeps what was asked')

  const h264 = buildLegacyCommand({
    ...legacyInput,
    ts: { ...SETTINGS, codec: 'h264_nvenc', skip_existing: false, overwrite_output: true },
  })
  assert.match(h264, /^ffmpeg -y /)
  assert.match(h264, /-tune hq/)
  assert.doesNotMatch(h264, /-tune uhq/, 'h264 never gets uhq')
})

// --- the experimental HDR10 chain ---------------------------------------------

const hdrInput = {
  ...gpuInput,
  requested: true,
  paperWhiteScrgb: 8.0,
  sourceWidth: 3840,
  sourceHeight: 2160,
  ts: { ...SETTINGS, codec: 'hevc_nvenc', profile: 'main10', pixel_format: 'p010le',
        container: '.mkv' },
}

const hdrLines = (overrides = {}) =>
  buildHdrTransferPreview({ ...hdrInput, ...overrides }).lines.join('\n')

test('only the GPU-transfer chain previews HDR, and only when it was opted into', () => {
  const on = buildExportPreview('gpu_transfer', legacyInput, gpuInput, hdrInput)
  assert.equal(on.kind, 'hdr10-transfer-chain')

  // The same HDR request on the other two chains is not an HDR preview: those chains
  // carry 8-bit SDR pixels, and the backend refuses the pairing at the gate.
  assert.equal(buildExportPreview('legacy', legacyInput, gpuInput, hdrInput).kind,
    'legacy-command')
  assert.equal(buildExportPreview('auto', legacyInput, gpuInput, hdrInput).kind,
    'legacy-command')

  // Off is off: the SDR device chain is shown, not a wider one with HDR words in it.
  const off = buildExportPreview('gpu_transfer', legacyInput, gpuInput,
    { ...hdrInput, requested: false })
  assert.equal(off.kind, 'gpu-transfer-chain')
  assert.doesNotMatch(off.lines.join('\n'), /scRGB/)
})

test('ProRes GPU preview describes host transfer and CPU HQ MOV encoding',()=>{
  const preview=buildExportPreview('gpu_transfer',legacyInput,{...gpuInput,
    ts:{...SETTINGS,codec:'prores_ks',container:'.mov',profile:'hq',pixel_format:'yuv422p10le'},
    renderPipeline:['nr','vsr']}, {requested:true,ts:{...SETTINGS,codec:'prores_ks',container:'.mov',profile:'hq',pixel_format:'yuv422p10le'},
    guidanceMode:0,flowEngine:'nvof',workingScale:.5,residualPyramid:true,sourceFps:120,
    outputDir:'C:\\Videos',fileName:'clip_prores',fallbackStem:'clip',paperWhiteScrgb:8,
    sourceWidth:1920,sourceHeight:1080})
  const text=preview.lines.join('\n')
  assert.match(text,/GPU 渲染/);assert.match(text,/主机内存/)
  assert.match(text,/prores_ks -profile:v 3 -pix_fmt yuv422p10le -tag:v apch -f mov/)
  assert.doesNotMatch(text,/NVENC|zeroCopy|零拷贝/)
  assert.equal(preview.copyable,false);assert.equal(preview.command,'')
})

test('the HDR chain is a device chain with no command to copy', () => {
  const preview = buildExportPreview('gpu_transfer', legacyInput, gpuInput, hdrInput)

  assert.equal(preview.command, '', 'a command line would start a different pipeline')
  assert.equal(preview.copyable, false)
  assert.ok(!preview.lines.some((line) => /^\s*ffmpeg(\.exe)?\s/.test(line)))
  assert.doesNotMatch(preview.lines.join('\n'), /-f rawvideo -pix_fmt bgr24/)
})

test('the HDR preview carries the selected layer vector and fusion request', () => {
  const text = hdrLines({
    reconstructionMode: 'triple', layerScales: [0.4, 0.67, 0.85],
    layerWeights: [1, 0.8, 1.2], fusionMode: 'data_consistency_experimental',
    fusionBeta: 0.2, fusionMaxIterations: 64, fusionRelativeTolerance: 0.0001,
  })
  assert.match(text, /TRIPLE 独立 RGBA16F Feature 18 会话/)
  assert.match(text, /第1层 40% \/ 权重 1；第2层 67% \/ 权重 0.8；第3层 85% \/ 权重 1.2/)
  assert.match(text, /data-consistency CG 融合（β=0.2，最多 64 次，相对阈值 0.0001/)
  assert.match(text, /3 层 · multi_pass=1/)
  assert.match(text, /能力是否可用由后端实际探测决定/)
})

test('the HDR chain names the fixed policy the bridge is opened with', () => {
  const text = hdrLines()

  assert.match(text, new RegExp(paperWhiteLabel(8.0)))
  assert.match(text, new RegExp(`支持上限 ${HDR_SUPPORT_CEILING_SCRGB} scRGB`))
  assert.match(text, new RegExp(`钳位上限 ${HDR_CLIP_CEILING_SCRGB} scRGB`))
  assert.match(text, new RegExp(`合法峰值 ${HDR_DERIVED_LEGAL_PEAK_SCRGB} scRGB`))
  // The ceilings are policy, not a control and not something read off the source.
  assert.match(text, /上限来自固定策略，绝不从 MaxCLL 推导/)
  assert.match(text, /不逐帧/)
  // Paper white is a control the operator picks for the job; the two ceilings are not.
  assert.match(text, /纸白 .*——这是 UI 选择/)
  assert.match(text, /scRGB 不是 UI 控件/)
  assert.doesNotMatch(text, /纸白[^。]*不是 UI 控件/)
  // The peak is the 125 scRGB box times the matrix row sums. The support ceiling is the
  // bound that sits above the result, so it must not be described as an input to it.
  assert.match(text, /由钳位上限 125 scRGB（ST 2084 顶在本桥参考白下的 BT\.2020 线性盒上界）/)
  assert.match(text, /Rec\.2020→Rec\.709 转换矩阵的正行和最大值 ~1\.6602/)
  assert.match(text, /不参与该推导/)
  assert.doesNotMatch(text, /由此推导|由此[^。]*推导/)
})

test('the HDR chain follows the paper white the job carries', () => {
  assert.match(hdrLines({ paperWhiteScrgb: 20 }), /20 scRGB = 1600 cd\/m²/)
  assert.match(hdrLines({ paperWhiteScrgb: 8.0 }), /640 cd\/m²/)
})

test('the HDR chain states the guidance plane it actually runs', () => {
  const guided = hdrLines({ guidanceMode: 1 })

  // The order the code runs in: the colour bridge derives the RGBA8 plane, NVOF reads it
  // and answers with float2 flow, and Feature 18 takes that flow beside the FP16 proxy.
  assert.match(guided, /proxy\(\) 在同一次 dispatch 里从工作主帧/)
  assert.match(guided, /RGBA8 引导图像/)
  assert.match(guided, /NVOF（flow_engine=nvof）\n?读这张引导图像/)
  assert.match(guided, /float2 设备光流/)
  assert.match(guided, /Feature 18 消费的是这道光流加 FP16 代理/)
  assert.match(guided, /只用于运动估计，从不作为 HDR 主色传输/)
  assert.match(guided, /NVOF 的输出是运动，不是 RGBA8/)

  // The three statements this line used to make, each of which is false: NVOF does not
  // author the RGBA8 plane, the plane is not colour-free, and it is not unrelated to the
  // FP16 proxy — both come off the same working master in the same pass.
  assert.doesNotMatch(guided, /NVOF[^。]*产生一张 RGBA8/)
  assert.doesNotMatch(guided, /只承载运动，不承载颜色|不承载任何颜色|不含颜色/)
  assert.doesNotMatch(guided, /互不相通|互不相关|独立于图像平面/)

  const unguided = hdrLines({ guidanceMode: 0 })
  assert.match(unguided, /零运动纹理/)
  assert.doesNotMatch(unguided, /RGBA8 引导图像|float2 设备光流/)
})

test('the HDR chain lists the form refusals as refusals, not as work', () => {
  const withBlockers = hdrLines({ blockers: ['codec=av1_nvenc：HDR10 编码入口只有 hevc_nvenc'] })
  assert.match(withBlockers, /当前表单里后端会拒绝的项（这是拒绝清单，不是将要完成的工作）/)
  assert.match(withBlockers, /· codec=av1_nvenc/)

  // Nothing listed: nothing invented. An empty list must not grow a heading.
  assert.doesNotMatch(hdrLines({ blockers: [] }), /当前表单里后端会拒绝的项/)
})

test('the HDR chain keeps backend-only unknowns as unknowns', () => {
  const text = hdrLines({ backendFinalChecks: ['源素材的 color_primaries 由后端探测后判定'] })
  assert.match(text, /只有后端能判定的项（页面无法预知，这里不作保证）/)
  assert.match(text, /· 源素材的 color_primaries/)
  assert.doesNotMatch(hdrLines({ backendFinalChecks: [] }), /只有后端能判定的项/)
})

test('the HDR chain states host traffic as prior evidence, an unanswered total, and unmeasured scalars', () => {
  const text = hdrLines()

  // What the accepted production gate actually measured, kept as that run's observation.
  assert.match(text, /不声明任何总量/)
  assert.match(text, /七个图像类主机边界计数器实测皆为 0/)
  assert.match(text, /report_readback_bytes 亦实测为 0/)
  assert.match(text, /那一次运行的观测证据/)
  assert.match(text, /既不是本作业的测量结果/)
  // The aggregate nobody answers, and the structural fact behind it.
  assert.match(text, /unavailable\/None/)
  assert.match(text, /未绑定任何图像回读入口/)
  // Scalar diagnostics do cross the boundary; no ABI counter weighs them.
  assert.match(text, /每次 dispatch 256 字节/)
  assert.match(text, /原生 ABI 没有计量这些字节的计数器/)
  assert.match(text, /未测量——未测量不是零/)

  // The two statements this line used to make, both false: the image-plane counters were
  // not missing in the accepted run, and there is no scalar byte counter that could be
  // "present but unmeasured".
  assert.doesNotMatch(text, /readback 计数在本版本缺失/)
  assert.doesNotMatch(text, /计数器存在但未测量|present\/unmeasured/)
  // No zero-total claim, and no promise about what a future job will measure.
  assert.doesNotMatch(text, /主机流量为零|图像流量全部为零|总量为零|总流量归零|所有计数器皆为 0/)
  assert.doesNotMatch(text, /无 readback|没有 readback|readback 为零/)
  assert.doesNotMatch(text, /将会测量|保证测量|必然测得/)
})

test('the HDR chain claims no conformance and no gain', () => {
  const text = hdrLines()

  assert.match(text, /未经正式 HDR 一致性认证/)
  assert.match(text, /FP16 链路存在数值误差/)
  assert.match(text, /NOT_CHARACTERIZED/)
  assert.match(text, /不会自动回退到另一条链路或另一种位深/)
  for (const claim of ['更快', '加速', '性能提升', '零拷贝']) {
    assert.doesNotMatch(text, new RegExp(claim), `preview advertises ${claim}`)
  }
})
