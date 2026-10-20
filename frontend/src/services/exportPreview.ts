import type { TranscodeSettings, RenderStage, DLSSSettings } from '../types'
// The .ts extension is what lets `node --test` load this module directly: Node's ESM
// resolver needs the real file name, and the delivery tests read the preview builder
// straight from source rather than through a browser.
import {
  HDR_CLIP_CEILING_SCRGB, HDR_CONTAINER, HDR_DERIVED_LEGAL_PEAK_SCRGB,
  HDR_PROFILE, HDR_SUPPORT_CEILING_SCRGB, NITS_PER_SCRGB, paperWhiteLabel,
} from './hdrExport.ts'

// What the delivery panel shows as "this is how your file will be written".
//
// The two export chains are different kinds of thing, so one text box cannot describe
// both honestly: the legacy chain really is one FFmpeg command line the backend builds
// (`pipeline.build_transcode_command`), while the GPU Transfer chain encodes inside this
// process through the NVDEC / NVOF-DLSS / Lanczos / residual / NVENC bridges and only
// touches FFmpeg for an optional copy-and-mux pass at the end. Showing the legacy string
// under that backend would advertise a command that runs a pipeline the operator did not
// select, so this module returns a kind plus either a command or a stage list, and the
// panel renders what the kind says.
//
// The HDR10 chain is a third kind: it encodes in-process and writes its own Matroska, and
// it reaches for the SDR chain's existing copy-and-mux pass only when the request carries
// something beyond video - with -c:v copy, so the bitstream is never encoded twice. A
// request neither the entry nor that pass can serve is refused rather than reinterpreted,
// and its preview has to show the refusal as a refusal.
export type ExportPreviewKind = 'legacy-command' | 'gpu-transfer-chain' | 'hdr10-transfer-chain'

export interface ExportPreview {
  kind: ExportPreviewKind
  // Only ever non-empty for 'legacy-command'. A GPU Transfer preview is never
  // copyable-as-a-command, because pasting it into a shell would start the other chain.
  command: string
  // Human-readable stage/parameter lines, both kinds.
  lines: string[]
  copyable: boolean
}

export interface LegacyPreviewInput {
  inputPath: string
  outputPath: string
  pipeWidth: number
  pipeHeight: number
  // Source frame rate, already defaulted by the caller (fps <= 0 reads as 30 there).
  sourceFps: number
  ts: TranscodeSettings
}

// The path the preview prints. The job payload uses composeJobOutputPath instead, whose
// empty result means "derive it from the source at run time".
export function composePreviewOutputPath(
  outputDir: string, fileName: string, container: string
): string {
  return outputDir
    ? `${outputDir}\\${fileName || 'export'}${container}`
    : `output${container}`
}

// The destination the backend is actually handed for this form state.
export function composeJobOutputPath(
  outputDir: string, fileName: string, container: string, fallbackStem: string
): string {
  const trimmedDir = outputDir.trim().replace(/[\\/]+$/, '')
  const trimmedName = fileName.trim() || fallbackStem
  return trimmedDir ? `${trimmedDir}\\${trimmedName}${container}` : ''
}

function foldWindowsPath(path: string): string {
  const text = String(path || '').trim()
  if (!text) return ''
  let body = text.replace(/\//g, '\\')
  // Extended-length and UNC aliases name the same file through a different spelling.
  if (/^\\\\\?\\UNC\\/i.test(body)) body = '\\\\' + body.slice('\\\\?\\UNC\\'.length)
  else if (/^\\\\\?\\/.test(body)) body = body.slice('\\\\?\\'.length)
  const isUnc = body.startsWith('\\\\')
  const parts = body.split('\\')
  const stack: string[] = []
  for (const part of parts) {
    if (part === '' || part === '.') continue
    if (part === '..') {
      if (stack.length > 0 && stack[stack.length - 1] !== '..') stack.pop()
      continue
    }
    stack.push(part.replace(/[. ]+$/, ''))
  }
  // Case and separator spelling are not identity on Windows; trailing dots and spaces
  // are not either. This is the prevention layer only - it cannot see a link or a hard
  // link, which is why the backend re-checks with the file table before anything opens.
  const folded = stack.map((s) => s.toLowerCase()).join('\\')
  return (isUnc ? '\\\\' : '') + folded
}

export function sameOutputAsInput(
  inputPath: string, outputDir: string, fileName: string, container: string, fallbackStem: string
): boolean {
  const out = composeJobOutputPath(outputDir, fileName, container, fallbackStem)
  if (!out || !inputPath) return false
  const a = foldWindowsPath(inputPath)
  const b = foldWindowsPath(out)
  return Boolean(a) && a === b
}

// Kept as a client-side mirror of pipeline.build_transcode_command, not as its source of
// truth: the backend still builds the command that runs. e2e 7.1 compares the two strings
// token by token and fails if this drifts.
export function buildLegacyCommand({
  inputPath, outputPath, pipeWidth, pipeHeight, sourceFps, ts,
}: LegacyPreviewInput): string {
  const pipeFps = sourceFps.toFixed(6)
  const overwriteFlag = (ts.skip_existing || !ts.overwrite_output) ? '-n' : '-y'
  const parts: string[] = ['ffmpeg', overwriteFlag, '-loglevel', 'warning',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', `${pipeWidth}x${pipeHeight}`, '-r', pipeFps, '-i', '-']

  parts.push('-i', `"${inputPath}"`)
  parts.push('-map', '0:v:0')

  if (ts.audio_mode !== 'remove') parts.push('-map', '1:a?')
  if (ts.subtitle_mode !== 'drop') parts.push('-map', '1:s?')
  if (ts.copy_attachments && ts.container === '.mkv') parts.push('-map', '1:t?')

  parts.push('-map_metadata', ts.copy_metadata ? '1' : '-1')
  parts.push('-map_chapters', ts.copy_chapters ? '1' : '-1')

  parts.push('-c:v', ts.codec)
  parts.push('-preset', ts.preset)

  let tune = ts.tune
  if (tune === 'uhq' && ts.codec === 'h264_nvenc') tune = 'hq'
  if (tune !== 'auto') {
    parts.push('-tune', tune)
  }
  if (ts.preset !== 'p7' && ts.multipass !== 'auto') {
    parts.push('-multipass', ts.multipass)
  }

  if (ts.rate_mode === 'cq') {
    parts.push('-rc:v', 'vbr', '-cq:v', String(ts.cq), '-b:v', '0')
  } else if (ts.rate_mode === 'constqp') {
    parts.push('-rc:v', 'constqp', '-qp', String(ts.qp))
  } else if (ts.rate_mode === 'cbr') {
    parts.push('-rc:v', 'cbr', '-b:v', ts.video_bitrate, '-maxrate', ts.video_bitrate, '-bufsize', ts.video_bitrate)
  } else {
    parts.push('-rc:v', 'vbr', '-b:v', ts.video_bitrate)
  }

  if (ts.rate_mode === 'cq' || ts.rate_mode === 'vbr') {
    if (ts.peak_policy === 'custom') {
      parts.push('-maxrate', ts.maxrate, '-bufsize', ts.bufsize)
    }
  }

  parts.push('-g', String(ts.gop_mode === 'auto' ? Math.round(sourceFps * 2) : ts.gop))

  if (ts.b_frames !== 'auto') parts.push('-bf', ts.b_frames)
  if (ts.b_ref_mode !== 'auto') parts.push('-b_ref_mode', ts.b_ref_mode)
  if (ts.lookahead !== 'auto') parts.push('-rc-lookahead', ts.lookahead)
  if (ts.lookahead_level !== 'auto') parts.push('-lookahead_level', ts.lookahead_level)

  if (ts.aq_mode === 'spatial') {
    parts.push('-spatial-aq', '1', '-temporal-aq', '0', '-aq-strength', String(ts.aq_strength))
  } else if (ts.aq_mode === 'temporal') {
    parts.push('-spatial-aq', '0', '-temporal-aq', '1')
  } else {
    parts.push('-spatial-aq', '0', '-temporal-aq', '0')
  }

  if (ts.weighted_prediction) parts.push('-weighted_pred', '1')
  if (ts.temporal_filter !== 'auto') parts.push('-tf_level', ts.temporal_filter)
  if (ts.split_encode !== 'auto') parts.push('-split_encode_mode', ts.split_encode)
  if (ts.gpu !== '-1') parts.push('-gpu', ts.gpu)
  if (ts.profile !== 'auto') parts.push('-profile:v', ts.profile)
  if (ts.level !== 'auto') parts.push('-level', ts.level)
  if (ts.tier !== 'auto' && ts.codec !== 'h264_nvenc') parts.push('-tier', ts.tier)

  if (ts.pixel_format !== 'auto') parts.push('-pix_fmt', ts.pixel_format)
  else parts.push('-pix_fmt', 'yuv420p')

  const swsMap: Record<string, string> = { lanczos: 'lanczos', bicubic: 'bicubic', bilinear: 'bilinear', nearest: 'neighbor' }
  parts.push('-sws_flags', swsMap[ts.scale_algorithm] || 'lanczos')
  if (ts.color_preset === 'bt709_limited') parts.push('-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv')
  else if (ts.color_preset === 'bt709_full') parts.push('-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'pc')
  else if (ts.color_preset === 'bt2020_sdr') parts.push('-colorspace', 'bt2020nc', '-color_primaries', 'bt2020', '-color_trc', 'bt709', '-color_range', 'tv')
  else if (ts.color_preset === 'bt2020_pq') parts.push('-colorspace', 'bt2020nc', '-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-color_range', 'tv')
  else if (ts.color_preset === 'bt2020_hlg') parts.push('-colorspace', 'bt2020nc', '-color_primaries', 'bt2020', '-color_trc', 'arib-std-b67', '-color_range', 'tv')
  else if (ts.color_preset === 'hdr_to_sdr') parts.push('-vf', 'tonemap=libplacebo:desat=0:tonemapping=bt2390:peak=100:format=yuv420p')

  if (ts.audio_mode === 'copy') parts.push('-c:a', 'copy', '-shortest')
  else if (ts.audio_mode === 'aac') parts.push('-c:a', 'aac', '-b:a', ts.audio_bitrate, '-shortest')
  else if (ts.audio_mode === 'opus') parts.push('-c:a', 'libopus', '-b:a', ts.audio_bitrate, '-shortest')
  else if (ts.audio_mode === 'flac') parts.push('-c:a', 'flac', '-shortest')
  else if (ts.audio_mode === 'pcm') parts.push('-c:a', 'pcm_s24le', '-shortest')
  else if (ts.audio_mode === 'remove') parts.push('-an')

  if (ts.subtitle_mode === 'copy') parts.push('-c:s', 'copy')
  else if (ts.subtitle_mode === 'convert') {
    const subCodec = ts.container === '.webm' ? 'webvtt' : ts.container === '.mkv' ? 'srt' : 'mov_text'
    parts.push('-c:s', subCodec)
  } else if (ts.subtitle_mode === 'drop') parts.push('-sn')

  if (ts.copy_attachments && ts.container === '.mkv') parts.push('-c:t', 'copy')

  if (ts.codec === 'av1_nvenc' && (ts.container === '.mp4' || ts.container === '.mov')) {
    parts.push('-tag:v', 'av01')
  } else if (ts.codec === 'hevc_nvenc' && (ts.container === '.mp4' || ts.container === '.mov')) {
    parts.push('-tag:v', 'hvc1')
  }

  if (ts.container === '.mp4' || ts.container === '.mov') {
    parts.push('-movflags', '+faststart')
  }

  if (ts.extra_args.trim()) {
    parts.push(ts.extra_args.trim())
  }

  parts.push(`"${outputPath}"`)

  return parts.join(' ')
}

function frameGenerationLines(engine: 'off' | 'vfg' | 'vfg_dynamic' = 'off', multiplier = 2, fps: number, target: number | string = 60): string[] {
  return engine !== 'off' ? [
    `导出插帧：NVIDIA VFG ${engine === 'vfg_dynamic' ? '动态 · 目标 '+target : multiplier+'×'} · ${fps} → ${engine === 'vfg_dynamic' ? target : fps * multiplier} fps；在渲染链完成后执行。`,
    '保留时长与音轨同步；片尾补齐最后一帧，实时预览保持原帧率。',
  ] : []
}

export interface GpuTransferPreviewInput {
  renderPipeline?: RenderStage[]
  dlaaPreset?: number
  vsrQuality?: number
  vsrTarget?: string
  frameGeneration?: 'off' | 'vfg' | 'vfg_dynamic'
  frameMultiplier?: DLSSSettings['frame_multiplier']
  framegenTargetFps?: number | string
  sdr10?: boolean
  ts: TranscodeSettings
  guidanceMode: number
  flowEngine: string
  workingScale: number
  residualPyramid: boolean
  reconstructionMode?: 'single' | 'dual' | 'triple'
  layerScales?: number[]
  layerWeights?: number[]
  fusionMode?: 'catmull_baseline' | 'data_consistency_experimental'
  fusionBeta?: number
  fusionMaxIterations?: number
  fusionRelativeTolerance?: number
  sourceFps: number
  outputDir: string
  fileName: string
  fallbackStem: string
}

// Where the requested tune survives the capability check, and where it cannot.
function tuneLine(ts: TranscodeSettings): string {
  const requested = ts.tune
  if (requested === 'auto') return '调优 tune：auto（由编码器默认决定）'
  if (requested === 'uhq' && ts.codec === 'h264_nvenc') {
    return '调优 tune：请求 uhq → 实际必定下发 hq（h264_nvenc 没有 UHQ）'
  }
  if (requested === 'uhq') {
    return `调优 tune：请求 uhq；能否按 uhq 下发取决于后端对 ${ts.codec} 编码器能力的探测，`
      + '探测为不支持时实际下发 hq。本行只是请求值，实际值以任务诊断日志为准。'
  }
  return `调优 tune：${requested}`
}

export function buildGpuTransferPreview({
  ts, guidanceMode, flowEngine, workingScale, residualPyramid, sourceFps,
  reconstructionMode = 'single', layerScales = [], layerWeights = [],
  fusionMode = 'catmull_baseline', fusionBeta = 0.1,
  fusionMaxIterations = 128, fusionRelativeTolerance = 0.001,
  outputDir, fileName, fallbackStem, sdr10 = false,
  renderPipeline = ['nr'], dlaaPreset = 0, vsrQuality = 2, vsrTarget = '1080p',
  frameGeneration = 'off', frameMultiplier = 2, framegenTargetFps = 60,
}: GpuTransferPreviewInput): ExportPreview {
  const out = composeJobOutputPath(outputDir, fileName, ts.container, fallbackStem)
  const isMultiScale = reconstructionMode !== 'single'
  const activeScales = isMultiScale ? layerScales : [workingScale]
  const interpolation = frameGenerationLines(frameGeneration, frameMultiplier, sourceFps, framegenTargetFps)
  if (renderPipeline.some(stage => stage !== 'nr')) return {
    kind: 'gpu-transfer-chain', command: '', copyable: false, lines: [
      `渲染顺序：${renderPipeline.map(s => (({nr:'DLSSNR',dlaa:'DLAA',vsr:'RTX VSR',vfg:'NVIDIA VFG',truehdr:'TrueHDR',illustration:'IllustrationJaNai V3',animesharp:'AnimeSharp V4'} as Record<string,string>)[s.split(':')[0]])).join(' → ')}`,
      ...(sdr10 ? [`SDR 10-bit：P010 → linear Rec.709 FP16 → 所选阶段 → P010 → ${ts.codec === 'av1_nvenc' ? 'AV1 Main · 10-bit' : 'HEVC Main10'}。`,
        ...(renderPipeline.includes('vsr') ? ['VSR 通过 VFX 1.3 RGB10A2；不经过 8-bit 图像。'] : [])] : []),
      ...(renderPipeline.includes('dlaa') ? [`DLAA 模型提示：${({0:'默认',5:'E',6:'F',10:'J',11:'K',12:'L',13:'M'} as Record<number,string>)[dlaaPreset] ?? dlaaPreset}；实际模型由运行库决定。`] : []),
      ...(renderPipeline.includes('nr') ? [`DLSSNR：${reconstructionMode} · ${activeScales.map(s => `${Math.round(s * 100)}%`).join(' / ')}；VSR 提前时按放大后尺寸处理。`] : []),
      ...(renderPipeline.includes('vsr') ? [`RTX VSR：档位 ${vsrQuality} · ${vsrTarget === '4k' ? '4K' : '1080p'}`] : []),
      'NVDEC → 所选 GPU 阶段 → NVENC；音轨、字幕、章节、metadata 和附件复用 FFmpeg 重封装。',
      ...interpolation,
      `输出：${out || '由源素材推导'}`,
    ],
  }
  if (sdr10) return { kind: 'gpu-transfer-chain', command: '', copyable: false, lines: [
    'SDR 10-bit · GPU Transfer',
    `P010 → linear Rec.709 FP16 → 所选阶段 → P010 → ${ts.codec === 'av1_nvenc' ? 'AV1 Main · 10-bit' : 'HEVC Main10'}`,
    `模式：${reconstructionMode} · 比例：${activeScales.map(s => `${Math.round(s * 100)}%`).join(' / ')}`,
    '保留 BT.709 SDR；不转换为 PQ，不添加 HDR metadata。',
    '音轨、字幕、章节与附件复用现有 FFmpeg 封装。',
    ...interpolation,
    `输出：${out}`,
  ] }

  const lines: string[] = [
    ...interpolation,
    '导出链路：GPU Transfer（export_backend=gpu_transfer）— 帧全程留在显存，不经过 FFmpeg 的 rawvideo 管道。',
    '本链路不由一条 FFmpeg 命令行驱动，所以这里不给出可运行命令，复制按钮保持不可用：',
    '把 Legacy 链路的命令粘到终端会启动另一条管线，而不是刚选定的这一条。',
    '',
    '设备链路阶段：',
    '  1. NVDEC 在 GPU 上解码源素材 → 8-bit NV12 4:2:0（limited range）。',
    guidanceMode === 1
      ? `  2. NVOF 设备光流（flow_engine=${flowEngine}）：光流张量在显存内产生，不下载回系统内存。`
      : '  2. guidance_mode=0：不运行光流引擎，使用 DLSS 宿主常驻的零运动纹理入口。',
    isMultiScale
      ? `  3. ${reconstructionMode.toUpperCase()} 独立 DLSS 会话：每层从同一全分辨率源用 Lanczos-2 降采样，各自运行 NVOF / DLSS。`
      : `  3. DLSS 宿主在 GPU 内合成帧（工作分辨率缩放 ${workingScale}）。`,
    isMultiScale
      ? `  4. ${fusionMode === 'data_consistency_experimental'
        ? `GPU data-consistency CG 融合；β=${fusionBeta}，最多 ${fusionMaxIterations} 次，相对阈值 ${fusionRelativeTolerance}，未收敛即失败。`
        : '各层残差用于观察；最终只编码第 1 层 Catmull 基线，没有融合。'}`
      : residualPyramid
        ? '  4. GPU Lanczos 上采样 + 残差金字塔融合，两者都按调用方的 tap 计划驱动。'
        : '  4. 残差金字塔已关闭：链路中会残留一次 CPU 缩放，后端因此会拒绝该任务，而不是自行回退。',
    '  5. NVENC 桥在本进程内直接编码：编码参数按名字写入编码器选项表，不是命令行开关。',
    ...(isMultiScale ? [
      '',
      '独立层配置（实验）：',
      ...activeScales.map((scale, index) =>
        `  第 ${index + 1} 层：${Math.round(scale * 100)}% · 权重 ${layerWeights[index] ?? 1}`),
      '性能与画质尚未定性。',
    ] : []),
    '',
    `编码器（请求值）：${ts.codec} · preset ${ts.preset} · ${tuneLine(ts)}`,
    `码控（请求值）：${ts.rate_mode === 'cq'
      ? `CQ ${ts.cq}（等价于 -rc:v vbr -cq:v ${ts.cq} -b:v 0）`
      : ts.rate_mode === 'constqp'
        ? `恒定 QP ${ts.qp}`
        : ts.rate_mode === 'cbr'
          ? `CBR ${ts.video_bitrate}`
          : `VBR ${ts.video_bitrate}`}`,
    `GOP（请求值）：${ts.gop_mode === 'auto'
      ? `源帧率 × 2 = ${Math.round(sourceFps * 2)} 帧`
      : `${ts.gop} 帧`}`,
    `像素格式：桥只提交 NV12 8-bit 4:2:0；请求 ${ts.pixel_format === 'auto' ? 'yuv420p（等价）' : ts.pixel_format}${
      ts.pixel_format === 'auto' || ts.pixel_format === 'yuv420p'
        ? '。'
        : ' 不在其列，后端会拒绝该任务。'}`,
    '',
    '收尾封装：NVENC 桥只能写出单视频流的 .mkv。目标容器不是 .mkv，或请求了音频重编码、字幕转换、'
    + '附件/元数据/章节保留、MP4 faststart、额外参数时，链路会在编码完成后追加一次 -c:v copy 重封装'
    + '（视频不做二次编码）；音频/字幕/元数据是否需要带过来由后端读取源素材的流后判定。',
    `目标文件：${out || '（未指定目录，运行时由源素材同名推导）'}`,
    '准入：任一条件不满足时后端拒绝并写明原因，不会自动回退到 Legacy 链路。',
    '性能：该链路为 opt-in，本版本未做基准测试，不声明任何速度或质量增益（NOT_CHARACTERIZED）。',
  ]

  return { kind: 'gpu-transfer-chain', command: '', lines, copyable: false }
}

export interface HdrPreviewInput extends GpuTransferPreviewInput {
  requested: boolean
  paperWhiteScrgb: number
  sourceWidth: number
  sourceHeight: number
  // The clauses the page can already prove. They are shown as refusals, never as work
  // the chain will do.
  blockers?: string[]
  // What only the backend can decide, stated as undecided.
  backendFinalChecks?: string[]
  limitations?: string[]
}

export function buildHdrTransferPreview(input: HdrPreviewInput): ExportPreview {
  if (input.renderPipeline?.some(s => s !== 'nr')) {
    const order = input.renderPipeline
    const out = composeJobOutputPath(input.outputDir, input.fileName, input.ts.container, input.fallbackStem)
    return {kind: 'hdr10-transfer-chain', command: '', copyable: false, lines: [
      'HDR10 · GPU Transfer · 实验性',
      `渲染顺序：${order.map(s => (({nr:'DLSSNR',dlaa:'DLAA',vsr:'RTX VSR',vfg:'NVIDIA VFG',truehdr:'TrueHDR',illustration:'IllustrationJaNai V3',animesharp:'AnimeSharp V4'} as Record<string,string>)[s.split(':')[0]])).join(' → ')}`,
      `P010 / PQ / BT.2020 → 有符号线性 scRGB FP16 → 所选阶段 → PQ / P010 → ${input.ts.codec === 'av1_nvenc' ? 'AV1 Main · 10-bit' : 'HEVC Main10'}。`,
      ...(order.includes('nr') ? ['NR 使用现有 FP16 代理和线性残差重建；多层融合保持原有算法。'] : []),
      ...(order.includes('dlaa') ? ['DLAA 直接处理线性 FP16 主图像；深度和抖动为零。'] : []),
      ...(order.includes('vsr') ? ['VSR 使用 VFX 1.3 RGB10A2 / PQ / BT.2020；输出回到线性 FP16。'] : []),
      ...frameGenerationLines(input.frameGeneration, input.frameMultiplier, input.sourceFps, input.framegenTargetFps),
      '保留源静态 HDR metadata；音轨、字幕、章节与附件复用现有 FFmpeg 重封装。',
      ...(input.blockers ?? []).map(s => `无法导出：${s}`),
      `输出：${out || '由源素材推导'}`,
    ]}
  }
  const {
    ts, guidanceMode, flowEngine, workingScale, reconstructionMode = 'single',
    layerScales = [], layerWeights = [], fusionMode = 'catmull_baseline',
    fusionBeta = 0.1, fusionMaxIterations = 128, fusionRelativeTolerance = 0.001,
    sourceFps, outputDir, fileName,
    fallbackStem, paperWhiteScrgb, sourceWidth, sourceHeight,
    blockers = [], backendFinalChecks = [], limitations = [],
  } = input

  const out = composeJobOutputPath(outputDir, fileName, ts.container, fallbackStem)
  const layerCount = reconstructionMode === 'triple' ? 3 : reconstructionMode === 'dual' ? 2 : 1
  const activeScales = reconstructionMode === 'single'
    ? [workingScale]
    : Array.from({ length: layerCount }, (_, index) => layerScales[index] ?? workingScale)
  const activeWeights = Array.from({ length: layerCount }, (_, index) => layerWeights[index] ?? 1)
  const guidanceLine = guidanceMode === 1
    ? '  4. 运动引导：色彩桥的 proxy() 在同一次 dispatch 里从工作主帧同时得到有界 FP16 显示代理与一张 '
      + `RGBA8 引导图像——后者由主帧自身派生，不是另一路无关的数据。NVOF（flow_engine=${flowEngine}）`
      + '读这张引导图像，产出留在显存的 float2 设备光流；Feature 18 消费的是这道光流加 FP16 代理。'
      + '引导图像只用于运动估计，从不作为 HDR 主色传输；NVOF 的输出是运动，不是 RGBA8。'
    : '  4. guidance_mode=0：不运行光流引擎，使用 DLSS 宿主常驻的零运动纹理入口。'

  const lines: string[] = [
    ...frameGenerationLines(input.frameGeneration, input.frameMultiplier, input.sourceFps, input.framegenTargetFps),
    '导出链路：实验性 HDR10 GPU Transfer（experimental_hdr_scrgb=true · export_backend=gpu_transfer）'
    + '— 仅用于导出；预览播放器不是经过审计的 HDR 显示路径，不作为 HDR 色彩基准。',
    '本链路不由一条 FFmpeg 命令行驱动，所以这里不给出可运行命令，复制按钮保持不可用：',
    '把 Legacy 链路的命令粘到终端会启动另一条管线，而不是刚选定的这一条。',
    '',
    '设备链路阶段：',
    `  1. NVDEC 在 GPU 上解码 HDR10 源 → 10-bit P010 4:2:0（limited range，BT.2020 / ST 2084 PQ）。`,
    '  2. 色彩桥（gpu_hdr_scrgb_bridge）：PQ → 有符号线性 scRGB FP16；Lanczos-2 降采样得到工作主帧，'
    + '保持有符号、未钳位的 FP16（signed, unclamped working master）。',
    reconstructionMode === 'single'
      ? `  3. 由工作主帧派生有界 FP16 显示代理，交给 DLSS Feature 18 的 RGBA16F 会话推理（工作缩放 ${workingScale}）。`
      : `  3. ${reconstructionMode.toUpperCase()} 独立 RGBA16F Feature 18 会话；各层从 pristine FP16 主帧独立派生工作幅面，HDR 残差留在线性 scRGB。`,
    guidanceLine,
    reconstructionMode !== 'single' && fusionMode === 'data_consistency_experimental'
      ? `  5. GPU typed FP16 data-consistency CG 融合（β=${fusionBeta}，最多 ${fusionMaxIterations} 次，相对阈值 ${fusionRelativeTolerance}；未收敛或能力闸门失败即拒绝），再由融合结果编码。`
      : reconstructionMode !== 'single'
        ? '  5. Catmull 基线重建与 FP16 合成；其余层参与观测，不参与最终融合输出。'
        : '  5. 原始 HDR 主帧重建 + Catmull-Rom 残差/FP16 合成 → scRGB → PQ → P010。',
    `  6. NVENC HDR10 编码：${ts.codec} @ ${ts.codec === 'av1_nvenc' ? 'main' : HDR_PROFILE} / P010，单视频流写入 ${HDR_CONTAINER}；`
    + `静态母版显示与 CLL/FALL 以码流内 ${ts.codec === 'av1_nvenc' ? 'Metadata OBU' : 'SEI'} 保留。`,
    '  7. 收尾封装：请求携带音频、字幕、章节、附件、容器级 FORMAT 元数据或源视频轨自己的标题/语言/标志位时，'
    + '编码先结束并释放文件，随后复用 SDR 链路既有的 gtx.build_mux_command copy pass，'
    + `以 -c:v copy 把这些流带进同一个 ${HDR_CONTAINER}：`
    + '不追加第二次视频编码，也不另写一个封装器；视频数据包原样拷贝，不需要携带任何东西时这一步不发生，'
    + '交付件就是编码器自己写出的那一个。临时中间文件由本次运行在整个生命周期上负责删除，'
    + '逐帧失败、取消、finish/teardown 抛错、封装失败与校验失败都不留下它。',
    '',
    '策略取值（整段作业固定，不逐帧变化）：',
    `  纸白 ${paperWhiteLabel(paperWhiteScrgb)}（NITS_PER_SCRGB：每 1.0 scRGB = ${NITS_PER_SCRGB} cd/m²）`
      + '——这是 UI 选择，按整段作业选定一次，不逐帧变化。',
    `  支持上限 ${HDR_SUPPORT_CEILING_SCRGB} scRGB 与钳位上限 ${HDR_CLIP_CEILING_SCRGB} scRGB 不是 UI 控件；`
      + '上限来自固定策略，绝不从 MaxCLL 推导，也绝不逐帧自适应。',
    `  合法峰值 ${HDR_DERIVED_LEGAL_PEAK_SCRGB} scRGB 由钳位上限 125 scRGB（ST 2084 顶在本桥参考白下的 `
      + 'BT.2020 线性盒上界）与本桥 Rec.2020→Rec.709 转换矩阵的正行和最大值 ~1.6602 相乘得到；'
      + '支持上限只是留有余量地高于它，不参与该推导。',
    '',
    `输出幅面：${sourceWidth}x${sourceHeight}（源素材全幅；输出幅面缩放被拒绝）· ${reconstructionMode} ${layerCount} 层 · multi_pass=1。`
    + '工作幅面不在被拒之列：25%–100% 的整数百分比可用，与 CPU 方案和 SDR 后端共用 resolve_working_dims 推导。',
    `HDR 层配置：${activeScales.map((scale, index) => `第${index + 1}层 ${Math.round(scale * 100)}% / 权重 ${activeWeights[index]}`).join('；')}；融合 ${fusionMode}。`
      + '能力是否可用由后端实际探测决定；此预览不推断硬件支持。',
    `编码器（请求值）：${ts.codec} · profile ${ts.profile} · preset ${ts.preset} · ${tuneLine(ts)}`,
    `码控（请求值）：${ts.rate_mode === 'cq'
      ? `CQ ${ts.cq}（等价于 -rc:v vbr -cq:v ${ts.cq} -b:v 0）`
      : ts.rate_mode === 'constqp'
        ? `恒定 QP ${ts.qp}`
        : ts.rate_mode === 'cbr'
          ? `CBR ${ts.video_bitrate}`
          : `VBR ${ts.video_bitrate}`}`,
    `GOP（请求值）：${ts.gop_mode === 'auto'
      ? `源帧率 × 2 = ${Math.round(sourceFps * 2)} 帧`
      : `${ts.gop} 帧`}`,
    `色彩：color_preset=${ts.color_preset} — 写回源素材自己的 HDR10 声明；不做色调映射，`
    + '不写第二份色彩子句。',
    '非视频流：'
    + `audio_mode=${ts.audio_mode} · subtitle_mode=${ts.subtitle_mode} · `
    + `copy_attachments=${ts.copy_attachments} · copy_metadata=${ts.copy_metadata} · `
    + `copy_chapters=${ts.copy_chapters}——这些选择保持不动，由编码结束后的一次 -c:v copy 重封装携带`
    + `（gtx.build_mux_command，视频不二次编码）；源素材里确实没有该类流时那条流自然带不过来，`
    + '是否含有由后端读取源素材判定。章节以 ffprobe -show_chapters 的条目数为准：'
    + 'Matroska 可以不写 format.nb_chapters，把那个缺失当作“没有章节”就是一次静默丢弃。'
    + '源视频轨自己的标题、语言与标志位随 copy_metadata 一并携带（默认复制，无需另开设置）：'
    + '它们写在输出的视频流上，不 map 源视频，也不把源素材的 ENCODER / BPS / DURATION / 帧数'
    + '冒充成这一次编码的实测值。静态 HDR 元数据（母版显示 / CLL / FALL）由编码器写在码流内，'
    + '与容器级 FORMAT 元数据是否复制无关。',
    '容器：交付件只有 .mkv——编码入口写出 .mkv，重封装也只写回同一个 .mkv；其它容器仍被拒绝。',
    'extra_args：本阶段不支持——只要该字段有内容，作业在派发到 GPU 前就被拒绝，理由如文字写明；'
    + '后端 hdr_mux_refusals 与本页同一口径，没有逐选项白名单，也不会静默应用或静默忽略。'
    + '理由是一句可执行的拒绝而不是一个解析器：任何选项名列表都无法把一处无害的添加，与一句会把'
    + '母版显示（ST 2086）和内容光照（CTA-861.3）SEI 从交付件里删掉的子句区分开。'
    + 'SDR 链路的 extra_args 行为不变。',
    `目标文件：${out || '（未指定目录，运行时由源素材同名推导）'}`,
    '',
    '主机流量：不声明任何总量，也不为未来作业预先声明会被测到什么。在已被接受的产线闸门那次运行里，'
    + '解码器 / DLSS / 编码器的七个图像类主机边界计数器实测皆为 0，HDR 色彩桥的诊断图像计数器 '
    + 'report_readback_bytes 亦实测为 0；那是那一次运行的观测证据，既不是本作业的测量结果，'
    + '也不能被读成本作业的零流量证明。',
    'HDR 像素回读的完整字节总量没有计数器可答（unavailable/None），产品链路也未绑定任何图像回读入口。'
    + '标量诊断确实逐次穿越 GPU→主机（按源码常量每次 dispatch 256 字节），但原生 ABI 没有计量这些字节的'
    + '计数器，其总量为未测量——未测量不是零。',
    '本链路未经正式 HDR 一致性认证，不声称无损，也不声称任何画质或性能增益（NOT_CHARACTERIZED）；'
    + 'FP16 链路存在数值误差。',
    '准入：任一条件不满足时后端拒绝并写明原因，不会自动回退到另一条链路或另一种位深。',
  ]

  if (blockers.length > 0) {
    lines.push('', '当前表单里后端会拒绝的项（这是拒绝清单，不是将要完成的工作）：',
      ...blockers.map((reason) => `  · ${reason}`))
  }
  if (backendFinalChecks.length > 0) {
    lines.push('', '只有后端能判定的项（页面无法预知，这里不作保证）：',
      ...backendFinalChecks.map((check) => `  · ${check}`))
  }
  if (limitations.length > 0) {
    lines.push('', '实验性限制：', ...limitations.map((item) => `  · ${item}`))
  }

  return { kind: 'hdr10-transfer-chain', command: '', lines, copyable: false }
}

export function buildExportPreview(
  exportBackend: string, legacy: LegacyPreviewInput, gpu: GpuTransferPreviewInput,
  hdr?: HdrPreviewInput
): ExportPreview {
  if (String(exportBackend || 'auto').toLowerCase() === 'gpu_transfer') {
    if (gpu.ts.codec === 'prores_ks') {
      const out = composeJobOutputPath(gpu.outputDir, gpu.fileName, gpu.ts.container, gpu.fallbackStem)
      const stages = gpu.renderPipeline ?? ['nr']
      const hdr10 = Boolean(hdr?.requested)
      const trueHdr = stages.includes('truehdr')
      const sourceMode = hdr10 ? 'HDR10 / PQ' : trueHdr ? 'TrueHDR / 线性 FP16' : gpu.sdr10 ? 'SDR 10-bit' : 'SDR 8-bit'
      const stageNames: Record<string, string> = {nr:'DLSSNR',dlaa:'DLAA',vsr:'RTX VSR',vfg:'NVIDIA VFG',truehdr:'TrueHDR',illustration:'IllustrationJaNai V3',animesharp:'AnimeSharp V4'}
      return {kind:'gpu-transfer-chain',command:'',copyable:false,lines:[
        `ProRes 422 HQ · ${sourceMode} → GPU 渲染 → CPU 编码 · MOV`,
        `渲染顺序：${stages.map(s => stageNames[s.split(':')[0]] ?? s).join(' → ')}`,
        'GPU Transfer 执行所选图像阶段；完成后将渲染帧从 GPU 传到主机内存。',
        'ProRes 422 输入使用 CPU 软件解码；输出使用 CPU 软件编码。',
        '编码器预览：ffmpeg -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le -tag:v apch -f mov',
        '固定输出：ProRes 422 HQ（10-bit 4:2:2）· MOV（.mov）；不显示硬件编码预设或码控参数。',
        ...frameGenerationLines(gpu.frameGeneration, gpu.frameMultiplier, gpu.sourceFps, gpu.framegenTargetFps),
        `目标文件：${out || '由源素材推导'}`,
      ]}
    }
    // The opt-in only changes what an export job asks for, and only on the chain that
    // can serve it: with the flag off - or with the chain on auto/legacy, where the
    // service refuses the request outright - this stays the SDR preview.
    if (hdr && hdr.requested) return buildHdrTransferPreview(hdr)
    return buildGpuTransferPreview(gpu)
  }
  const command = buildLegacyCommand(legacy)
  return {
    kind: 'legacy-command',
    command,
    lines: [command],
    copyable: true,
  }
}
