import type { DLSSSettings, TranscodeSettings, VideoInfo } from '../types'
// Extension required: node --test loads this module directly, and its resolver wants
// the real file name where the bundler would take either.
import { WORKING_PERCENT_MAX, WORKING_PERCENT_MIN, workingPercent } from './workingScale.ts'

// The client-side mirror of the part of gpu_hdr_transfer_export.negotiate_hdr that a
// form can answer for itself, and a list of the parts it cannot.
//
// The backend is authoritative: every clause here exists so the panel can say "this job
// will be refused, and by which clause" before the request leaves the page. Nothing in
// this module can approve a job - a configuration that reads clean here still goes
// through negotiate_hdr, which alone can see the source's own colour declaration, the
// installed bridge entries and the open Feature 18 session. Refusal is the contract on
// both sides: an unsupported request is refused, never run as something else.

export const HDR_SETTING = 'experimental_hdr_scrgb'
export const HDR_PAPER_WHITE_SETTING = 'hdr_paper_white_scrgb'

// The fixed policy the chain opens itself with. Stated here so the panel quotes the
// same numbers as the backend; none of them is a control, and none is adjustable.
export const HDR_PAPER_WHITE_SCRGB = 8.0
export const NITS_PER_SCRGB = 80.0
export const HDR_SUPPORT_CEILING_SCRGB = 250.0
export const HDR_CLIP_CEILING_SCRGB = 125.0
export const HDR_DERIVED_LEGAL_PEAK_SCRGB = 207.53

export const HDR_CODEC = 'hevc_nvenc'
export const HDR_PROFILE = 'main10'
export const P010_CODECS: readonly string[] = ['hevc_nvenc', 'av1_nvenc']
export const p010Profile = (codec: string) => codec === 'av1_nvenc' ? 'main' : 'main10'
export const HDR_CONTAINER = '.mkv'

// plan_hdr_encoder's own vocabularies, clause for clause.
export const HDR_ALLOWED_COLOR_PRESETS: readonly string[] =
  ['preserve', 'none', '', 'None', 'bt2020_pq']
export const HDR_ALLOWED_PIXEL_FORMATS: readonly string[] =
  ['auto', 'p010le', 'yuv420p10le']
export const HDR_ALLOWED_PROFILES: readonly string[] =
  ['auto', '', 'None', 'none', 'main10', 'Main 10', 'main 10', 'Main10']
export const HDR_RATE_MODES: readonly string[] = ['cq', 'vbr', 'cbr', 'constqp']
export const HDR_AQ_MODES: readonly string[] = ['spatial', 'temporal', 'off', 'none']

// gpu_transfer_export.MUX_AUDIO_MODES / MUX_SUBTITLE_MODES: the vocabulary of the copy
// pass this chain reuses. It is the shared helper's list rather than a second one
// invented here, which is why a mode outside it is refused at the gate instead of
// arriving after the GPU work is already spent.
export const HDR_MUX_AUDIO_MODES: readonly string[] =
  ['copy', 'aac', 'opus', 'flac', 'pcm', 'remove']
export const HDR_MUX_SUBTITLE_MODES: readonly string[] = ['copy', 'convert', 'drop']

// The source shape the decode bridge reads: negotiate_hdr's HDR_SOURCE_* constants.
export const HDR_SOURCE_PIX_FMT = 'yuv420p10le'
export const HDR_SOURCE_TRANSFER = 'smpte2084'
export const HDR_HLG_TRANSFERS: readonly string[] = ['arib-std-b67', 'hlg']

export function paperWhiteNits(paperWhiteScrgb: number): number {
  return Math.round(paperWhiteScrgb * NITS_PER_SCRGB * 100) / 100
}

export function paperWhiteLabel(paperWhiteScrgb: number): string {
  const nits = paperWhiteNits(paperWhiteScrgb)
  return `${paperWhiteScrgb} scRGB = ${Number.isInteger(nits) ? nits : nits.toFixed(2)} cd/m²`
}

// hdr_requested: explicit only. An absent key, a false value or anything unreadable is
// "not requested", which leaves the job on the SDR chain.
export function isHdrRequested(settings: Record<string, unknown> | DLSSSettings | null): boolean {
  const value = (settings as Record<string, unknown> | null)?.[HDR_SETTING]
  if (typeof value === 'string') {
    const text = value.trim().toLowerCase()
    return text === 'true' || text === '1' || text === 'yes' || text === 'on'
  }
  return value === true || value === 1
}

// hdr_flag_reason: the flag is a boolean, and a value that is neither clearly on nor
// clearly off is refused rather than guessed at.
export function hdrFlagReason(value: unknown): string | null {
  if (value === undefined || value === null || typeof value === 'boolean') return null
  if (typeof value === 'number') {
    if (value === 0 || value === 1) return null
  }
  if (typeof value === 'string') {
    const text = value.trim().toLowerCase()
    if (['true', 'false', '1', '0', 'yes', 'no', 'on', 'off', ''].includes(text)) return null
  }
  return `${HDR_SETTING}=${String(value)} 既不是明确的开也不是明确的关：实验性 HDR 链路用布尔值选择，`
    + '任何猜测都不构成一个作业'
}

// paper_white_reason: a float, finite, at least 1.0 scRGB. Empty, NaN and Infinity all
// land in the first clause, which is what the backend's own float() conversion does.
export function hdrPaperWhiteReason(value: unknown): string | null {
  // js.get(HDR_PAPER_WHITE_SETTING, HDR_PAPER_WHITE_SCRGB): an absent key is the fixed
  // default, so a settings blob written before this switch existed is not a refusal.
  // A key present as null is not absent — float(None) is what the backend evaluates.
  if (value === undefined) return null
  const blank = typeof value === 'string' && value.trim() === ''
  const number = blank ? Number.NaN : Number(value)
  if (value === null || value === '' || !Number.isFinite(number)) {
    return `${HDR_PAPER_WHITE_SETTING}=${String(value)} 不是数字：它是打开 HDR 色彩桥时使用的漫反射白`
      + `（${paperWhiteLabel(HDR_PAPER_WHITE_SCRGB)}），按整段作业选定，不逐帧变化`
  }
  if (number < 1.0) {
    return `${HDR_PAPER_WHITE_SETTING}=${number} 不是有限且至少 1.0 scRGB`
  }
  return null
}

// plan_hdr_encoder and negotiate_hdr read each of these with a default when the key is
// absent (ts.get("profile", "auto")). Mirroring that is what keeps a missing key from
// being previewed as a refusal the gate would never raise - and, the other way round,
// keeps the mirror's trims in the same places the Python trims.
const named = (value: unknown, fallback: string): string =>
  value === undefined || value === null ? fallback : String(value)

// os.path.splitext(out_path)[1] for the path the panel composes: the extension of the
// destination, not of the directory it sits in.
function outExtension(path: string): string {
  const name = String(path || '').trim().split(/[\\/]/).pop() ?? ''
  const at = name.lastIndexOf('.')
  return at <= 0 ? '' : name.slice(at).toLowerCase()
}

// hdr_mux_refusals' extra_args clause: this stage of the HDR10 chain carries no
// caller-written FFmpeg arguments at all, so any non-empty value is refused whatever it
// contains. The mirror has no parser, and that is the point - the only question is
// whether the field has something in it. A denylist of option names is what used to be
// here, and it was the bug: it accepted '-bsf:v filter_units=remove_types=39|40', which
// strips the mastering-display and content-light SEIs out of the deliverable while every
// gate still said yes. Reading an argument list the way FFmpeg does is FFmpeg's own
// job, and a second answer to it written here would only be a second chance to be wrong.
export function hdrExtraArgsReason(value: unknown): string | null {
  // A field the caller never filled in is not an argument list. JSON null is how an empty
  // form control arrives, which is why '' and null are the same answer and 'None' is not
  // a token to run.
  const extra = value === null || value === undefined ? '' : String(value).trim()
  if (!extra) return null
  return `extra_args=${extra} 不被本阶段接受：HDR10 链路的 copy pass 已经带着交付件的视频编解码器、流映射、`
    + '色彩声明、比特流过滤器与覆盖决定，FFmpeg 以靠后的子句为准，而任何选项名列表都无法把一处无害的添加，'
    + '与一句会把母版显示（ST 2086）和内容光照（CTA-861.3）SEI 从交付件里删掉的子句区分开。'
    + "因此本阶段拒绝任何非空的 extra_args；映射由表单的 audio_mode、subtitle_mode、copy_attachments、"
    + 'copy_metadata 与 copy_chapters 表达。SDR 链路的 extra_args 不受此限制'
}

export interface HdrFormInput {
  settings: DLSSSettings
  ts: TranscodeSettings
  video: VideoInfo | null
  // The destination the panel would hand the queue; its extension is what the gate
  // checks, not the container the select happens to show.
  outPath: string
  // The output extent the panel shows. The chain feeds the encoder the full-size
  // composite, so anything other than the source extent is a refusal.
  outputWidth: number
  outputHeight: number
}

// What the page can prove from its own form state and from VideoInfo, in the order the
// gate states them. Each of these is a clause of negotiate_hdr/plan_hdr_encoder.
// The clauses the preview page can decide from its own controls plus VideoInfo: the
// flag, the paper white, the chain/host/layer/pass/flow choices, the working scale and
// the source shape. Each one is a clause of negotiate_hdr.
export function hdrSettingsBlockers(settings: DLSSSettings, video: VideoInfo | null): string[] {
  const js = settings as unknown as Record<string, unknown>
  const reasons: string[] = []

  for (const reason of [hdrFlagReason(js[HDR_SETTING]),
                        hdrPaperWhiteReason(js[HDR_PAPER_WHITE_SETTING])]) {
    if (reason) reasons.push(reason)
  }
  if (!isHdrRequested(js)) return reasons

  if (String(js.export_backend) !== 'gpu_transfer') {
    reasons.push(`export_backend=${String(js.export_backend)}：实验性 HDR 链路是 GPU Transfer 的一种模式，`
      + '永不回退到承载 8-bit SDR 像素的 FFmpeg 管道')
  }
  // The gate compares str(js.get("dlss_backend")) — so a payload that omits the key is
  // refused too, and the mirror refuses it the same way instead of assuming a default
  // the job snapshot would not supply.
  if (String(js.dlss_backend) !== 'gpu_flow_d3d12') {
    reasons.push(`dlss_backend=${String(js.dlss_backend)}：HDR 链路运行在 dlss_backend=gpu_flow_d3d12，`
      + '即本项目唯一的 GPU DLSS 宿主')
  }
  // int(js.get("multi_pass", 1) or 1) != 1: the gate reads a blank as one pass.
  if ((Number(js.multi_pass ?? 1) || 1) !== 1) {
    reasons.push('multi_pass≠1：生产管线尚未实现多 pass，HDR 链路也不添加它')
  }

  const guidance = Number(js.guidance_mode ?? 0)
  if (guidance !== 0 && guidance !== 1) {
    reasons.push(`guidance_mode=${String(js.guidance_mode)} 既不是 0（关闭）也不是 1（光流）`)
  } else if (guidance === 1 && !['nvof', 'fidelityfx', 'raft', 'raft_large'].includes(String(js.flow_engine ?? 'nvof'))) {
    reasons.push(`flow_engine=${String(js.flow_engine)} 没有设备通路：引导图像无法直接交给设备光流，`
      + '就得把图像下载回主机才能算运动')
  }

  const percent = workingPercent(Number(js.working_scale))
  if (percent === null) {
    reasons.push(`working_scale=${String(js.working_scale)} 不是 ${WORKING_PERCENT_MIN}–`
      + `${WORKING_PERCENT_MAX} 之间的整数百分比：`
      + 'HDR 链路用与 CPU 方案和 SDR 后端相同的 resolve_working_dims 规则推导工作幅面，不另立一套缩放表')
  }

  // VideoInfo carries pix_fmt, bit_depth, is_hdr and color_transfer. Those four are
  // enough to name a refusal; they are not enough to promise a pass, which is what
  // hdrBackendFinalChecks is for.
  if (video && (video.width % 2 || video.height % 2)) {
    reasons.push(`源尺寸 ${video.width}x${video.height} 必须两者皆为偶数（全幅 4:2:0）`)
  }
  const proresSource = (video as (VideoInfo & {codec_name?: string}) | null)?.codec_name === 'prores' &&
    video?.pix_fmt === 'yuv422p10le' && Number(video.bit_depth || 0) >= 10
  if (video?.pix_fmt && video.pix_fmt !== HDR_SOURCE_PIX_FMT && !proresSource) {
    reasons.push(`pix_fmt=${video.pix_fmt}：HDR 链路需要 10-bit 4:2:0 源（${HDR_SOURCE_PIX_FMT}）`)
  }
  if (video && typeof video.bit_depth === 'number' && video.bit_depth < 10) {
    reasons.push(`bit_depth=${video.bit_depth}：NVDEC 交出的 P010 是 10-bit，低于它的源无法被这条链路读取`)
  }
  const transfer = String(video?.color_transfer ?? '')
  if (HDR_HLG_TRANSFERS.includes(transfer)) {
    reasons.push(`color_transfer=${transfer}：HLG 是另一套参考与另一条逆传递函数；本链路承载 ST 2084（PQ），`
      + '拒绝为 HLG 猜测一个显示尺寸')
  } else if (transfer && transfer !== HDR_SOURCE_TRANSFER) {
    reasons.push(`color_transfer=${transfer} 不是 smpte2084：色彩桥的解码逆传递函数是 PQ，`
      + 'SDR 标记或线性源会被当作它不是的东西来读取')
  }

  return reasons
}

// What the delivery page adds on top of that: the encoding request, and the mapping the
// reused copy pass carries. Still a refusal list, never a promise of acceptance.
export function hdrBlockers({
  settings, ts, video, outPath, outputWidth, outputHeight,
}: HdrFormInput): string[] {
  const reasons = hdrSettingsBlockers(settings, video)
  if (!isHdrRequested(settings as unknown as Record<string, unknown>)) return reasons

  if (named(ts && ts.codec, '') === 'prores_ks') {
    const outputContainer = outPath ? outExtension(outPath) : named(ts && ts.container, '').toLowerCase()
    if (outputContainer !== '.mov') {
      reasons.push('ProRes 422 HQ 仅写入 MOV 容器（.mov）')
    }
    if (named(ts && ts.profile, '').toLowerCase() !== 'hq') {
      reasons.push('ProRes 输出固定使用 HQ profile（profile 3）')
    }
    if (named(ts && ts.pixel_format, '').toLowerCase() !== 'yuv422p10le') {
      reasons.push('ProRes 422 HQ 固定使用 yuv422p10le（10-bit 4:2:2）')
    }
    return reasons
  }

  const codec = named(ts && ts.codec, HDR_CODEC)
  if (!P010_CODECS.includes(codec)) {
    reasons.push(`codec=${codec}：HDR10 支持 HEVC 或 AV1，不回退到别的编码器、`
      + '不降到 8-bit、也不降到 NV12')
  }
  const profile = named(ts && ts.profile, 'auto').trim()
  if (!(codec === 'av1_nvenc' ? ['auto', '', 'None', 'none', 'main'] : HDR_ALLOWED_PROFILES).includes(profile)) {
    reasons.push(`profile=${profile}：${codec} 使用 ${p010Profile(codec)}，10-bit`)
  }
  const pixelFormat = named(ts && ts.pixel_format, 'auto').trim().toLowerCase()
  if (!HDR_ALLOWED_PIXEL_FORMATS.includes(pixelFormat)) {
    reasons.push(`pixel_format=${pixelFormat}：HDR 链路提交 P010（10-bit 4:2:0）平面，`
      + '拒绝 8-bit 或 4:2:2/4:4:4 请求而不是照样编一遍')
  }
  const colorPreset = named(ts && ts.color_preset, 'preserve')
  if (!HDR_ALLOWED_COLOR_PRESETS.includes(colorPreset)) {
    reasons.push(`color_preset=${colorPreset}：本链路写回源素材自己的 HDR10 声明，`
      + 'hdr_to_sdr 与任何色调映射都会被拒绝——工作主帧是有符号、未钳位的，正是为了不被映射两次')
  }
  for (const key of ['custom_color_space', 'custom_color_primaries',
                     'custom_color_transfer', 'custom_color_range'] as const) {
    const value = String(ts[key] ?? '')
    if (value && value !== 'auto' && value !== 'preserve') {
      reasons.push(`${key}=${value}：由封装器写出的色彩子句会是第二份、且与编码器坚持的那份相矛盾的声明`)
    }
  }

  const container = outPath ? outExtension(outPath) : named(ts && ts.container, HDR_CONTAINER).toLowerCase()
  if (container !== HDR_CONTAINER) {
    reasons.push(`输出容器 ${container || '(none)'}：HDR10 交付件只有 ${HDR_CONTAINER}——编码入口写出 ${HDR_CONTAINER}，`
      + '复用的 copy pass 也只把非视频流带进同一个 .mkv。别的容器需要另一条封装分支，本轮没有实现，'
      + '因此被拒绝而不是被交付成另一个名字')
  }
  const extraArgsReason = hdrExtraArgsReason(ts && ts.extra_args)
  if (extraArgsReason) reasons.push(extraArgsReason)
  const audioMode = named(ts && ts.audio_mode, 'copy')
  if (!HDR_MUX_AUDIO_MODES.includes(audioMode)) {
    reasons.push(`audio_mode=${audioMode}：复用的容器 copy pass 只表达 ${HDR_MUX_AUDIO_MODES.join(' / ')}，`
      + '而这条链路本身不运行任何音频处理来实现别的取值')
  }
  const subtitleMode = named(ts && ts.subtitle_mode, 'copy')
  if (!HDR_MUX_SUBTITLE_MODES.includes(subtitleMode)) {
    reasons.push(`subtitle_mode=${subtitleMode}：复用的容器 copy pass 只表达 `
      + `${HDR_MUX_SUBTITLE_MODES.join(' / ')}。无法识别的取值会把字幕编码器的选择留给封装器，`
      + '那是一次静默转换而不是所请求的复制')
  }
  const rateMode = named(ts && ts.rate_mode, 'cq')
  if (!HDR_RATE_MODES.includes(rateMode)) {
    reasons.push(`rate_mode=${rateMode}：编码入口不接受该码控模式`)
  }
  const aqMode = named(ts && ts.aq_mode, 'spatial')
  if (!HDR_AQ_MODES.includes(aqMode)) {
    reasons.push(`aq_mode=${aqMode}：编码入口不接受该自适应量化模式`)
  }

  if ([outputWidth, outputHeight].some(v => !Number.isInteger(v) || v < 2 || v > 16384 || v % 2 !== 0)) {
    reasons.push(`输出分辨率必须为 2–16384 范围内的偶数宽高`)
  }

  return reasons
}

// What only the backend can decide, because it needs the source probe, the installed
// bridges or the open session. Listed as what it is: not a promise, not a green check.
export function hdrBackendFinalChecks({ ts, video }: HdrFormInput): string[] {
  const checks = [
    '加性解码/色彩桥/编码器入口与 Feature 18 的 RGBA16F 会话、D3D12 设备与能力上报：只有后端读取到才成立。',
    '源素材的 color_primaries / color_range / color_space 与母版显示、CLL/FALL 实际取值：由后端探测源素材后判定，'
      + '页面上的 VideoInfo 不含这些字段，这里不做保证。',
  ]
  const optionalStreams: string[] = []
  if (String(ts.audio_mode) !== 'remove') {
    optionalStreams.push('音轨（audio_mode 非 remove）')
  }
  if (String(ts.subtitle_mode) !== 'drop') {
    optionalStreams.push('字幕（subtitle_mode 非 drop）')
  }
  if (ts.copy_metadata) optionalStreams.push('容器级 FORMAT 元数据（copy_metadata=true）')
  if (ts.copy_chapters) optionalStreams.push('章节（copy_chapters=true）')
  if (optionalStreams.length > 0) {
    checks.push(`源素材若真的含有 ${optionalStreams.join('、')}，链路会在编码结束并释放封装后运行复用的 `
      + 'gtx.build_mux_command copy pass（-c:v copy，视频不重编码）把它们带进同一个 .mkv；'
      + '是否含有由后端读取源素材判定，页面不据此承诺某条流存在，也不承诺源素材里没有的流会凭空出现。')
  }
  if (!video) {
    checks.push('尚未导入素材：以上判定全部取决于后端对实际文件的探测。')
  }
  return checks
}

// The limits, stated once, in the words the contract uses. A panel that lists features
// without these lines is advertising a capability that was never built.
export const HDR_LIMITATIONS: readonly string[] = [
  '只服务 HDR10 / ST 2084（PQ）源：HLG、输出幅面缩放、色调映射一律明确拒绝，不回退成另一件事。'
    + 'Single、Dual、Triple 及其融合选择按现有多层配置运行，并由后端实际能力闸门决定是否可用；'
    + 'multi_pass 必须为 1。被拒的是输出幅面，不是工作幅面：工作缩放 25%–100% 的整数百分比'
    + '由 resolve_working_dims 推导工作尺寸。',
  '交付件只有 .mkv：编码入口写出单视频流 .mkv；请求携带音频/字幕/章节/附件/容器级 FORMAT 元数据或源视频轨的'
    + '标题/语言/标志（由 copy_metadata 一并决定，无需另开设置）时，链路复用既有 gtx.build_mux_command 的 '
    + 'copy pass，在编码结束并释放文件后以 -c:v copy 重封装——没有第二次编码，也没有另一个封装器，'
    + '视频数据包本身不动。非 .mkv 交付仍被拒绝。本阶段不支持任何 extra_args：只要该字段有内容，'
    + '作业就在派发到 GPU 前被明确拒绝，后端与页面同一口径，不做逐选项白名单，也不会静默应用或静默忽略。',
  '该开关只改变导出作业。预览播放器不是经过审计的 HDR 显示路径，不能作为 HDR 色彩基准。',
  '本版本未做正式 HDR 一致性认证，不声称无损、画质或性能增益；FP16 链路存在数值误差。',
]

export interface HdrPreset {
  settingsPatch: Partial<DLSSSettings>
  transcodePatch: Partial<TranscodeSettings>
  // What the click selects, shown before it is clicked.
  summary: string[]
}

// The one-click configuration of a job the gate can take. Explicit click only: nothing
// here runs when a file loads, and the summary names every choice it makes - including
// the mapping it leaves alone, because an operator who selected audio and subtitles
// would otherwise watch a readiness patch turn them off.
export function buildHdrExportPreset({ settings, ts }: HdrFormInput): HdrPreset {
  const selectedCodec = ts?.codec && P010_CODECS.includes(ts.codec) ? ts.codec : HDR_CODEC
  const selectedProfile = p010Profile(selectedCodec)
  const js = settings as unknown as Record<string, unknown>
  const settingsPatch: Partial<DLSSSettings> = {
    experimental_hdr_scrgb: true,
    export_backend: 'gpu_transfer',
    dlss_backend: 'gpu_flow_d3d12',
    multi_pass: 1,
  }
  const transcodePatch: Partial<TranscodeSettings> = {
    codec: selectedCodec,
    profile: selectedProfile,
    pixel_format: 'p010le',
    container: HDR_CONTAINER,
    color_preset: 'preserve',
    custom_color_space: '',
    custom_color_primaries: '',
    custom_color_transfer: '',
    custom_color_range: '',
    resolution_mode: 'original',
  }

  // One line per choice, and every key the click writes is named by the line that
  // writes it: the summary is the disclosure for a patch that changes the form, so a key
  // that changed without being named here would be a change the operator never saw.
  const currentWhite = js[HDR_PAPER_WHITE_SETTING]
  const whiteReason = hdrPaperWhiteReason(currentWhite)
  // An unusable paper white is put back on the fixed default rather than left in the
  // patch: a job the click cannot accept is not a job the click should claim to ready.
  const paperWhite = (whiteReason || currentWhite === undefined)
    ? HDR_PAPER_WHITE_SCRGB : Number(currentWhite)
  const summary = [
    '链路：export_backend=gpu_transfer · dlss_backend=gpu_flow_d3d12（本项目唯一的 GPU DLSS 宿主）',
    `重建：保留当前 reconstruction_mode=${String(js.reconstruction_mode ?? 'single')}、layer_scales、layer_weights、fusion_mode 与求解器参数；multi_pass=1（多层能力由后端实际闸门判定）`,
    `编码器：codec=${selectedCodec} · profile=${selectedProfile} · pixel_format=p010le（P010，10-bit 4:2:0）`,
    `容器：container=${HDR_CONTAINER}；编码入口写出单视频流 .mkv，请求携带非视频流时由复用的 `
      + 'gtx.build_mux_command copy pass 在编码结束后以 -c:v copy 重封装为同一个 .mkv（不再编一次）',
    '映射：audio_mode、subtitle_mode、copy_attachments、copy_metadata、copy_chapters '
      + '保持当前选择，交给那次 copy pass 携带；本点击不改写它们',
    '输出分辨率：resolution_mode=original，采用渲染链输出；也可在导出设置中选择其他分辨率',
    '色彩：color_preset=preserve（写回源素材自己的 HDR10 声明，不做色调映射）；'
      + '清空与之冲突的 custom_color_space、custom_color_primaries、custom_color_transfer、'
      + 'custom_color_range',
    `HDR 静态元数据：母版显示（ST 2086）与 CLL/FALL 写入码流${selectedCodec === 'av1_nvenc' ? ' Metadata OBU' : ' SEI'}，`
      + '与容器级 FORMAT 元数据是否被 copy_metadata 携带无关，也不随该开关改变',
    `实验性 HDR：${HDR_SETTING}=true · ${HDR_PAPER_WHITE_SETTING}=${paperWhite}`
      + `（${paperWhiteLabel(paperWhite)}，整段作业固定，不逐帧变化）`,
  ]

  if (whiteReason) {
    settingsPatch[HDR_PAPER_WHITE_SETTING] = HDR_PAPER_WHITE_SCRGB
    summary.push(`纸白当前不可用（${whiteReason}）：改回固定默认 ${HDR_PAPER_WHITE_SETTING}=${HDR_PAPER_WHITE_SCRGB}`)
  }

  if (workingPercent(Number(js.working_scale)) === null) {
    settingsPatch.working_scale = 1.0
    summary.push('工作分辨率缩放当前不是链路可用的整数百分比：working_scale=1.0（100%，源素材全幅）')
  }
  if (Number(js.guidance_mode ?? 0) === 1 && !['nvof', 'fidelityfx', 'raft', 'raft_large'].includes(String(js.flow_engine ?? 'nvof'))) {
    settingsPatch.flow_engine = 'nvof'
    summary.push('引导已开启（guidance_mode=1）：flow_engine=nvof，否则引导图像需要下载回主机才能计算光流')
  }
  if (!HDR_RATE_MODES.includes(named(ts && ts.rate_mode, 'cq'))) {
    transcodePatch.rate_mode = 'cq'
    summary.push(`码控模式 ${String(ts && ts.rate_mode)} 不被该入口接受：rate_mode=cq`)
  }
  if (!HDR_AQ_MODES.includes(named(ts && ts.aq_mode, 'spatial'))) {
    transcodePatch.aq_mode = 'spatial'
    summary.push(`自适应量化 ${String(ts && ts.aq_mode)} 不被该入口接受：aq_mode=spatial`)
  }
  if (!HDR_MUX_AUDIO_MODES.includes(named(ts && ts.audio_mode, 'copy'))) {
    transcodePatch.audio_mode = 'copy'
    summary.push(`音轨模式 ${String(ts && ts.audio_mode)} 不在 copy pass 表达的取值里：`
      + 'audio_mode=copy（复制源音轨，不重编码）')
  }
  if (!HDR_MUX_SUBTITLE_MODES.includes(named(ts && ts.subtitle_mode, 'copy'))) {
    transcodePatch.subtitle_mode = 'copy'
    summary.push(`字幕模式 ${String(ts && ts.subtitle_mode)} 不在 copy pass 表达的取值里：`
      + 'subtitle_mode=copy（复制源字幕轨，不转换编码）')
  }
  // This stage carries no caller-written FFmpeg arguments, so a job that names any is a
  // job the gate refuses. The click clears the field and says so, rather than leaving the
  // operator a readiness patch that still cannot dispatch.
  if (hdrExtraArgsReason(ts && ts.extra_args)) {
    transcodePatch.extra_args = ''
    summary.push(`本阶段不接受任何 extra_args，当前值是 ${String(ts && ts.extra_args)}：`
      + "清空为 extra_args=''（映射由表单的 audio_mode、subtitle_mode、copy_attachments、"
      + 'copy_metadata 与 copy_chapters 表达；被清空的内容不会被静默应用，也不会被静默忽略）')
  }

  return { settingsPatch, transcodePatch, summary }
}
