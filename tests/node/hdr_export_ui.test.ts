/**
 * The HDR10 opt-in at the level the page actually decides it: these are the client's
 * mirror of the gate's clauses and the one-click preset, both pure functions of the
 * form, so they run without a GPU, a browser or a live backend. What is asserted is
 * propagation - which clause a given form raises, and whether the configuration the
 * click writes down leaves any of them standing.
 *
 *   node --test ../tests/node/hdr_export_ui.test.ts
 *
 * The backend stays authoritative for everything a form cannot see; the last test here
 * is the one that says so.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  HDR_CLIP_CEILING_SCRGB, HDR_CODEC, HDR_CONTAINER, HDR_DERIVED_LEGAL_PEAK_SCRGB,
  HDR_LIMITATIONS, HDR_MUX_AUDIO_MODES, HDR_MUX_SUBTITLE_MODES,
  HDR_PAPER_WHITE_SCRGB, HDR_PAPER_WHITE_SETTING, HDR_PROFILE,
  HDR_SETTING, HDR_SUPPORT_CEILING_SCRGB, buildHdrExportPreset, hdrBackendFinalChecks,
  hdrBlockers, hdrExtraArgsReason, hdrFlagReason, hdrPaperWhiteReason, hdrSettingsBlockers,
  isHdrRequested, paperWhiteLabel,
} from '../../frontend/src/services/hdrExport.ts'

// The shipped defaults, copied rather than invented: App.tsx's DLSS defaults and
// DeliverPage.tsx's transcode defaults. A fixture that drifted from them would test a
// form the operator cannot reach.
const DLSS = {
  guidance_mode: 1,
  working_scale: 1.0,
  flow_engine: 'nvof',
  export_backend: 'auto',
  reconstruction_mode: 'single',
  multi_pass: 1,
  dlss_backend: 'gpu_flow_d3d12',
  experimental_hdr_scrgb: false,
  hdr_paper_white_scrgb: 8.0,
}

const TX = {
  codec: 'av1_nvenc',
  container: '.mkv',
  rate_mode: 'cq',
  cq: 16,
  video_bitrate: '80M',
  qp: 20,
  peak_policy: 'auto',
  preset: 'p7',
  tune: 'uhq',
  multipass: 'fullres',
  gop_mode: 'auto',
  gop: 240,
  aq_mode: 'spatial',
  aq_strength: 8,
  profile: 'auto',
  level: 'auto',
  tier: 'auto',
  resolution_mode: 'original',
  pixel_format: 'auto',
  color_preset: 'preserve',
  custom_color_space: 'bt709',
  custom_color_primaries: 'bt709',
  custom_color_transfer: 'bt709',
  custom_color_range: 'tv',
  audio_mode: 'copy',
  audio_bitrate: '320k',
  subtitle_mode: 'copy',
  copy_attachments: true,
  copy_metadata: true,
  copy_chapters: true,
  extra_args: '',
}

const HDR_VIDEO = {
  filename: 'clip.mkv', width: 3840, height: 2160, fps: 59.94, total_frames: 600,
  pix_fmt: 'yuv420p10le', bit_depth: 10, is_hdr: true, color_transfer: 'smpte2084',
}

// The reason strings are for a person to read; the assertion is on which clause fired.
// Each reason starts with the setting it refuses, except the three the gate states as
// sentences, which are named here once.
const SENTENCE_CLAUSES = [
  ['输出容器', 'container'],
  ['源尺寸', 'source_dims'],
  ['编码器收到的是全幅合成', 'output_extent'],
]

const clauses = (reasons: string[]): string[] => reasons.map((reason) => {
  const identifier = reason.match(/^[a-z_][a-z0-9_]*/)
  if (identifier) return identifier[0]
  for (const [prefix, name] of SENTENCE_CLAUSES) {
    if (reason.startsWith(prefix)) return name
  }
  return `unclassified: ${reason}`
}).sort()

const form = (dlssPatch = {}, tsPatch = {}, overrides = {}) => ({
  settings: { ...DLSS, ...dlssPatch },
  ts: { ...TX, ...tsPatch },
  video: HDR_VIDEO,
  outPath: 'C:\\Videos\\clip_dlss.mkv',
  outputWidth: 3840,
  outputHeight: 2160,
  ...overrides,
})

const ON = { experimental_hdr_scrgb: true, export_backend: 'gpu_transfer' }

test('HDR permits final export resizing after VSR', () => {
  const input = form({...ON,render_pipeline:['nr','vsr'],vsr_target:'4k'},
    {codec:'hevc_nvenc'},{video:{...HDR_VIDEO,width:1920,height:1080}})
  assert.ok(!hdrBlockers(input).some(s=>s.includes('输出不能缩放')))
  assert.ok(!hdrBlockers({...input,outputWidth:2560,outputHeight:1440}).some(s=>s.includes('输出分辨率')))
  assert.ok(hdrBlockers({...input,outputWidth:2561,outputHeight:1440}).some(s=>s.includes('输出分辨率')))
})

test('HDR ProRes output bypasses NVENC shape checks but retains source-domain checks',()=>{
  const input=form({...ON},{codec:'prores_ks',container:'.mov',profile:'hq',pixel_format:'yuv422p10le'},
    {video:{...HDR_VIDEO,codec_name:'prores',pix_fmt:'yuv422p10le'},outPath:'C:\\Videos\\clip.mov'})
  const reasons=hdrBlockers(input)
  assert.ok(!reasons.some(reason=>/codec=|profile=|pixel_format=|输出容器/.test(reason)),reasons.join('\n'))
  const badSource=hdrBlockers({...input,video:{...input.video,color_transfer:'arib-std-b67'}})
  assert.ok(badSource.some(reason=>reason.includes('color_transfer=arib-std-b67')))
})

// --- the opt-in is a request, not a guess --------------------------------------

test('the switch defaults to off and the backend default is what the panel quotes', () => {
  assert.equal(isHdrRequested(DLSS), false, 'the shipped defaults do not opt in')
  assert.equal(HDR_PAPER_WHITE_SCRGB, 8.0)
  assert.equal(HDR_PAPER_WHITE_SETTING, 'hdr_paper_white_scrgb')
  assert.equal(HDR_SETTING, 'experimental_hdr_scrgb')
  assert.equal(paperWhiteLabel(HDR_PAPER_WHITE_SCRGB), '8 scRGB = 640 cd/m²',
    'the label the chain opens with: 640 nits at 80 nits per scRGB unit')
})

test('the flag reads the forms the backend reads, and nothing else', () => {
  const on = [true, 'true', 'True', '1', 1, 'yes', 'on', ' TRUE ']
  const off = [false, 'false', '0', 0, '', 'no', 'off', null, undefined, 2, 'banana']

  for (const value of on) {
    assert.equal(isHdrRequested({ [HDR_SETTING]: value }), true, `${value} is on`)
  }
  for (const value of off) {
    assert.equal(isHdrRequested({ [HDR_SETTING]: value }), false, `${value} is off`)
  }
})

test('a flag that is neither on nor off is refused instead of guessed', () => {
  for (const value of [true, false, null, undefined, 0, 1, 'yes', 'off', '', 'TRUE']) {
    assert.equal(hdrFlagReason(value), null, `${value} is readable`)
  }
  for (const value of [2, 3.5, 'banana', Number.NaN]) {
    const reason = hdrFlagReason(value)
    assert.ok(reason, `${value} must be refused`)
    assert.ok(reason.startsWith(`${HDR_SETTING}=`), reason)
  }
})

test('paper white refuses exactly what the backend refuses', () => {
  // float(value) in the backend: a usable number, at least 1.0, finite. An absent key is
  // the fixed default, so a pre-HDR settings blob is not a refusal.
  for (const value of [8.0, 1, '8', '20.5', true, 250, undefined]) {
    assert.equal(hdrPaperWhiteReason(value), null, `${String(value)} is usable`)
  }
  for (const value of ['', '   ', 'abc', Number.NaN, Number.POSITIVE_INFINITY, null,
                       0.5, 0, -3]) {
    const reason = hdrPaperWhiteReason(value)
    assert.ok(reason, `${String(value)} must be refused`)
    assert.ok(reason.startsWith(`${HDR_PAPER_WHITE_SETTING}=`), reason)
  }
})

// --- which clause a form raises -------------------------------------------------

test('off means the HDR mirror says nothing about an SDR job', () => {
  // The delivery form as shipped: av1, audio/subtitle/attachments being copied, a
  // bt709 colour clause. Under the HDR gate every one of those is a refusal; with the
  // switch off none of them is, because this chain is not the one running.
  assert.deepEqual(hdrBlockers(form()), [])
  assert.deepEqual(hdrSettingsBlockers(form().settings, HDR_VIDEO), [])

  // Even with an unusable paper white typed in, an unrequested job is not policed.
  const blocked = hdrBlockers(form({ [HDR_PAPER_WHITE_SETTING]: 'abc' }))
  assert.deepEqual(clauses(blocked), ['hdr_paper_white_scrgb'],
    'the flag and paper white are checked either way; the rest only when opted in')
})

test('opting in on the shipped form names the clauses it violates', () => {
  const reasons = hdrBlockers(form(ON))

  // Not copy_attachments, not audio, not subtitles: the reused copy pass carries those,
  // so the shipped form's mapping choices are not a refusal. What is left is the encoder
  // the SDR defaults select and the colour clause that would contradict the source.
  assert.deepEqual(clauses(reasons), [
    'custom_color_primaries',
    'custom_color_range',
    'custom_color_space',
    'custom_color_transfer',
  ], 'the encoder and colour clauses the SDR defaults carry, and nothing invented')

  // export_backend is the one the switch alone does not set: HDR is a mode of GPU
  // Transfer, so opting in without moving the chain is a refusal, not a re-route.
  assert.deepEqual(clauses(hdrBlockers(form({ experimental_hdr_scrgb: true })))
    .concat(), [
    'custom_color_primaries', 'custom_color_range',
    'custom_color_space', 'custom_color_transfer', 'export_backend',
  ])
})

test('a form that satisfies the chain raises nothing', () => {
  const ready = form({ ...ON, guidance_mode: 0 }, {
    codec: HDR_CODEC, profile: 'main10', pixel_format: 'p010le', color_preset: 'preserve',
    custom_color_space: '', custom_color_primaries: '', custom_color_transfer: '',
    custom_color_range: '',
  })

  // The mapping the shipped form selects — audio and subtitles copied, attachments,
  // metadata and chapters kept — is a request this chain serves, not a refusal.
  assert.equal(ready.ts.copy_attachments, true)
  assert.deepEqual(hdrBlockers(ready), [])
  assert.deepEqual(hdrSettingsBlockers(ready.settings, HDR_VIDEO), [])

  // The same job with the encoder left on auto, the pixel format left on auto and the
  // streams switched off is still the same accepted request: those defaults ARE the
  // entry's own values, so the mirror must not invent a refusal for them.
  assert.deepEqual(hdrBlockers({
    ...ready,
    ts: { ...ready.ts, pixel_format: 'auto', profile: 'auto', aq_mode: 'spatial',
          audio_mode: 'remove', subtitle_mode: 'drop', copy_attachments: false },
  }), [])
})

test('HDR multi-layer modes are accepted by the form mirror; multi-pass and CPU flow stay refused', () => {
  for (const mode of ['dual', 'triple'] as const) {
    assert.deepEqual(clauses(hdrBlockers(form({ ...ON, reconstruction_mode: mode })))
      .filter((name) => name === 'reconstruction_mode'), [])
  }
  assert.deepEqual(clauses(hdrBlockers(form({ ...ON, multi_pass: 2 })))
    .filter((name) => name === 'multi_pass'), ['multi_pass'])
  assert.deepEqual(clauses(hdrBlockers(form({ ...ON, guidance_mode: 2 })))
    .filter((name) => name.startsWith('guidance')), ['guidance_mode'])
  assert.deepEqual(clauses(hdrBlockers(form({ ...ON, guidance_mode: 1, flow_engine: 'dis' })))
    .filter((name) => name === 'flow_engine'), ['flow_engine'],
    'guidance on a host-memory flow engine would download the tensor')

  // guidance_mode=0 needs no flow engine at all, so it stays clean on the zero-motion
  // entry whichever engine the panel happens to show.
  assert.deepEqual(hdrSettingsBlockers(
    { ...DLSS, ...ON, guidance_mode: 0, flow_engine: 'raft' } as any, HDR_VIDEO), [])
})

test('the source shape the bridge reads, and what a missing probe must not claim', () => {
  const variants: [any, string][] = [
    [{ ...HDR_VIDEO, color_transfer: 'arib-std-b67' }, 'color_transfer'],
    [{ ...HDR_VIDEO, color_transfer: 'bt709' }, 'color_transfer'],
    [{ ...HDR_VIDEO, pix_fmt: 'nv12' }, 'pix_fmt'],
    [{ ...HDR_VIDEO, bit_depth: 8 }, 'bit_depth'],
    [{ ...HDR_VIDEO, height: 2161 }, 'source_dims'],
  ]
  for (const [video, clause] of variants) {
    const fired = clauses(hdrSettingsBlockers({ ...DLSS, ...ON } as any, video))
    assert.ok(fired.includes(clause), `${clause} must fire for ${JSON.stringify(video)}`)
    assert.ok(!fired.includes('tonemap') && !fired.includes('colorspace'),
      'the mirror refuses this source; it does not offer to change it')
  }

  // No clip loaded: the page cannot assert anything about the source, so it raises no
  // source clause and no source clause only.
  assert.deepEqual(hdrSettingsBlockers({ ...DLSS, ...ON } as any, null), [])
})

test('the deliverable the chain can write: one name, one extent', () => {
  // The gate checks the extension of the path the job is handed, so an .mp4 destination
  // is refused even while the container select still reads .mkv.
  assert.ok(clauses(hdrBlockers(form(ON, {}, { outPath: 'C:\\Videos\\clip.mp4' })))
    .includes('container'))

  assert.ok(!clauses(hdrBlockers(form(ON, {}, { outputWidth: 1920, outputHeight: 1080 })))
    .includes('output_extent'), 'final GPU scaling permits a different output extent')

  // An empty path means the backend derives one beside the source; the container select
  // is then all the page has.
  assert.deepEqual(hdrBlockers(form(ON, {}, { outPath: '' }))
    .filter((reason) => reason.startsWith('输出容器')), [])
})

// --- the reused copy pass --------------------------------------------------------

test('the mapping the pass carries is not a refusal', () => {
  const ready = form({ ...ON, guidance_mode: 0 }, {
    codec: HDR_CODEC, profile: 'main10', pixel_format: 'p010le', color_preset: 'preserve',
    custom_color_space: '', custom_color_primaries: '', custom_color_transfer: '',
    custom_color_range: '',
  })

  // Attachments, metadata, chapters, copied audio and copied subtitles: each is one of
  // the clauses build_mux_command expresses, so none of them is a reason to refuse.
  assert.deepEqual(hdrBlockers(ready), [])
  assert.ok(ready.ts.copy_attachments && ready.ts.copy_metadata && ready.ts.copy_chapters)

  // A mode outside that vocabulary is refused, in its own words.
  for (const [patch, clause] of [
    [{ audio_mode: 'mp3' }, 'audio_mode'],
    [{ subtitle_mode: 'ass' }, 'subtitle_mode'],
  ] as [Record<string, string>, string][]) {
    const fired = clauses(hdrBlockers({ ...ready, ts: { ...ready.ts, ...patch } }))
    assert.ok(fired.includes(clause), `${JSON.stringify(patch)} must be refused`)
  }

  // Every value the delivery form can actually select is in the mirrored vocabulary, so
  // a hand-written payload is the only way to raise these clauses.
  for (const mode of ['copy', 'aac', 'opus', 'flac', 'pcm', 'remove']) {
    assert.ok(HDR_MUX_AUDIO_MODES.includes(mode), `${mode} is offered by the form`)
  }
  for (const mode of ['copy', 'convert', 'drop']) {
    assert.ok(HDR_MUX_SUBTITLE_MODES.includes(mode), `${mode} is offered by the form`)
  }
})

test('this stage carries no caller-written FFmpeg arguments at all', () => {
  const ready = form({ ...ON, guidance_mode: 0 }, {
    codec: HDR_CODEC, profile: 'main10', pixel_format: 'p010le', color_preset: 'preserve',
    custom_color_space: '', custom_color_primaries: '', custom_color_transfer: '',
    custom_color_range: '',
  })

  // Not a denylist: everything with something in it. The cases below include the ones that
  // used to fall through the option-name denylist, among them the argument that removes
  // the mastering-display and content-light SEIs from the deliverable.
  for (const extra of ['-c:v libx265', '-c:v=libx265', '-map 1:v:0', '-map=1:v',
                       '-color_trc bt709', '-color_trc=bt709', '-pix_fmt yuv420p',
                       '-an', '-y', '-f mp4', '-f=mp4', '-tag:v hvc1',
                       '-bsf:v filter_units=remove_types=39|40',
                       '-filter_complex [0:v]scale=1920:1080[v]',
                       'extra_output.mkv', '-metadata title=HDR', '"-c:v" libx265',
                       '-metadata title="HDR', '-metadata title=HDR \\']) {
    const reasons = hdrBlockers({ ...ready, ts: { ...ready.ts, extra_args: extra } })
    const fired = clauses(reasons)
    assert.ok(fired.includes('extra_args'), `${extra} must be refused, got ${fired}`)
    assert.match(reasons.join('\n'), /extra_args/, `the refusal quotes ${extra}`)
  }

  // A value that carries nothing is not a request, so it is not a refusal either. An
  // unfilled form control arrives as null, and null is not the token "null".
  for (const empty of ['', '   ', '\n', '\t', undefined, null]) {
    assert.equal(hdrExtraArgsReason(empty), null, `${JSON.stringify(empty)} is no request`)
    assert.deepEqual(hdrBlockers({ ...ready, ts: { ...ready.ts, extra_args: empty } }), [],
      `${JSON.stringify(empty)} must not be refused`)
  }
  // And the refusal says what it covers: the field, not a subset of its spellings.
  assert.match(String(hdrExtraArgsReason('-bsf:v filter_units=remove_types=39|40')),
    /本阶段拒绝任何非空的 extra_args/)
})

// --- the one-click preset -------------------------------------------------------

test('the preset leaves no form refusal standing', () => {
  const input = form({ ...ON, export_backend: 'auto', working_scale: 0.3 })
  const before = clauses(hdrBlockers(input))
  assert.ok(before.includes('export_backend') && before.includes('custom_color_transfer'))

  const preset = buildHdrExportPreset(input)
  const after = {
    ...input,
    settings: { ...input.settings, ...preset.settingsPatch },
    ts: { ...input.ts, ...preset.transcodePatch },
  }

  assert.deepEqual(hdrBlockers(after), [], 'every clause the page can prove is cleared')
  // The click cannot fix a source that is not HDR10: it changes settings and encoder
  // options, never the clip. That refusal has to survive the round trip.
  const sdrSource = { ...after, video: { ...HDR_VIDEO, pix_fmt: 'yuv420p', bit_depth: 8 } }
  assert.deepEqual(clauses(hdrBlockers(sdrSource)), ['bit_depth', 'pix_fmt'])
})

test('every key the click writes is named in the summary shown before it is clicked', () => {
  const preset = buildHdrExportPreset(form(ON))
  const text = preset.summary.join('\n')

  for (const key of [...Object.keys(preset.settingsPatch), ...Object.keys(preset.transcodePatch)]) {
    assert.ok(text.includes(key), `summary hides the change to ${key}`)
  }
  assert.equal(preset.settingsPatch.experimental_hdr_scrgb, true)
  assert.equal(preset.settingsPatch.export_backend, 'gpu_transfer')
  assert.equal(preset.settingsPatch.dlss_backend, 'gpu_flow_d3d12')
  assert.equal(preset.settingsPatch.reconstruction_mode, undefined,
    'the HDR preset preserves the selected reconstruction mode')
  assert.equal(preset.transcodePatch.codec, 'av1_nvenc')
  assert.equal(preset.transcodePatch.profile, 'main')
  assert.equal(preset.transcodePatch.container, HDR_CONTAINER)

  // What the click leaves alone is stated as kept, with the pass that carries it.
  assert.match(text, /映射：/)
  assert.match(text, /copy pass/)
  assert.match(text, /-c:v copy/)
  assert.match(text, /ST 2086/)
  for (const key of ['audio_mode', 'subtitle_mode', 'copy_attachments',
                     'copy_metadata', 'copy_chapters']) {
    assert.equal((preset.transcodePatch as Record<string, unknown>)[key], undefined,
      `the click must not rewrite the mapping choice ${key}`)
  }
  assert.doesNotMatch(text, /丢弃/, 'nothing is dropped, so nothing may be announced as dropped')
  // The ceilings are the chain's fixed policy, so the click neither quotes them as
  // adjustable nor touches them.
  assert.doesNotMatch(text, /MaxCLL/)
  assert.equal(HDR_SUPPORT_CEILING_SCRGB, 250.0)
  assert.equal(HDR_CLIP_CEILING_SCRGB, 125.0)
  assert.equal(HDR_DERIVED_LEGAL_PEAK_SCRGB, 207.53)
})

test('the HDR preset preserves selected multiscale and fusion settings', () => {
  const selected = {
    ...ON, reconstruction_mode: 'triple' as const,
    working_scale: 0.4, layer_scales: [0.4, 0.67, 0.85],
    layer_weights: [1, 0.8, 1.2], fusion_mode: 'data_consistency_experimental' as const,
    fusion_beta: 0.2, fusion_max_iterations: 64, fusion_relative_tolerance: 0.0001,
  }
  const preset = buildHdrExportPreset(form(selected))
  for (const key of ['reconstruction_mode', 'working_scale', 'layer_scales', 'layer_weights',
                     'fusion_mode', 'fusion_beta', 'fusion_max_iterations',
                     'fusion_relative_tolerance']) {
    assert.equal((preset.settingsPatch as Record<string, unknown>)[key], undefined,
      `the readiness preset must preserve ${key}`)
  }
  assert.match(preset.summary.join('\n'), /保留当前 reconstruction_mode=triple、layer_scales、layer_weights、fusion_mode/)
})

test('an unusable paper white is put back on the fixed default, not carried into the job', () => {
  const preset = buildHdrExportPreset(form({ ...ON, [HDR_PAPER_WHITE_SETTING]: '' }))
  assert.equal(preset.settingsPatch[HDR_PAPER_WHITE_SETTING], HDR_PAPER_WHITE_SCRGB)
  assert.match(preset.summary.join('\n'), new RegExp(HDR_PAPER_WHITE_SETTING))

  // A usable value the operator chose is left alone: the preset is a readiness patch,
  // not a reset.
  const kept = buildHdrExportPreset(form({ ...ON, [HDR_PAPER_WHITE_SETTING]: 20 }))
  assert.equal(kept.settingsPatch[HDR_PAPER_WHITE_SETTING], undefined)
  assert.match(kept.summary.join('\n'), /20 scRGB = 1600 cd\/m²/)
})

test('the preset only reaches for controls that would otherwise refuse', () => {
  const guided = buildHdrExportPreset(form({ ...ON, guidance_mode: 1, flow_engine: 'dis' }))
  assert.equal(guided.settingsPatch.flow_engine, 'nvof')
  assert.match(guided.summary.join('\n'), /flow_engine=nvof/)

  // The mapping is left where the operator put it, and only a value the reused pass does
  // not express is moved - back onto the copy, not onto silence.
  const kept = buildHdrExportPreset(form(ON))
  assert.equal(kept.transcodePatch.audio_mode, undefined)
  assert.equal(kept.transcodePatch.subtitle_mode, undefined)
  assert.equal(kept.transcodePatch.copy_attachments, undefined)
  assert.equal(kept.transcodePatch.extra_args, undefined)

  const resampled = buildHdrExportPreset(form(ON, { audio_mode: 'mp3', subtitle_mode: 'ass' }))
  assert.equal(resampled.transcodePatch.audio_mode, 'copy')
  assert.equal(resampled.transcodePatch.subtitle_mode, 'copy')
  assert.match(resampled.summary.join('\n'), /audio_mode=copy/)
  assert.match(resampled.summary.join('\n'), /subtitle_mode=copy/)
  assert.deepEqual(clauses(hdrBlockers(form(ON, { audio_mode: 'mp3' }))),
    ['audio_mode', 'custom_color_primaries', 'custom_color_range',
     'custom_color_space', 'custom_color_transfer'])

  const unsafe = buildHdrExportPreset(form(ON, { extra_args: '-c:v libx265' }))
  assert.equal(unsafe.transcodePatch.extra_args, '')
  assert.match(unsafe.summary.join('\n'), /extra_args/)

  const unguided = buildHdrExportPreset(form({ ...ON, guidance_mode: 0, flow_engine: 'raft' }))
  assert.equal(unguided.settingsPatch.flow_engine, undefined,
    'guidance off needs no flow engine, so the click does not silently pick one')

  const fixedScale = buildHdrExportPreset(form({ ...ON, working_scale: 0.5 }))
  assert.equal(fixedScale.settingsPatch.working_scale, undefined,
    '50% is served by the chain, so the click does not move it')
  assert.equal(buildHdrExportPreset(form({ ...ON, working_scale: 0.3 })).settingsPatch.working_scale,
    undefined, '30% is a whole percentage inside the served range')
  assert.equal(buildHdrExportPreset(form({ ...ON, working_scale: 0.333 })).settingsPatch.working_scale,
    1.0, 'a percentage the gate cannot name is put back on the full extent')
  assert.equal(buildHdrExportPreset(form({ ...ON, working_scale: 0.1 })).settingsPatch.working_scale,
    1.0, 'below the served range is not silently clamped to 25%')
})

// --- what the page must not promise --------------------------------------------

test('backend-only unknowns stay unknowns, and name what the pass carries conditionally', () => {
  const shipped = hdrBackendFinalChecks(form(ON))
  // The two unconditional ones are capability and source declaration; the stream line is
  // raised by audio/subtitle/metadata/chapters still being copied.
  assert.equal(shipped.length, 3)
  assert.match(shipped[2], /音轨/)
  assert.match(shipped[2], /字幕/)
  assert.match(shipped[2], /章节/)
  assert.match(shipped[2], /源素材若真的含有/, 'conditional on a probe this page cannot run')
  assert.match(shipped[2], /由后端读取源素材判定/)
  assert.doesNotMatch(shipped.join('\n'), /源素材含有音轨|一定会带过来/,
    'no claim about a stream the source probe has not read')

  // The click keeps the mapping, so it does not make the unknowns disappear: a job that
  // carries streams still has them decided by the backend.
  const preset = buildHdrExportPreset(form(ON))
  const ready = form({ ...ON, ...preset.settingsPatch }, { ...TX, ...preset.transcodePatch })
  assert.equal(hdrBackendFinalChecks(ready).length, 3)

  // With nothing asked for beyond video, only the probe-only unknowns remain.
  assert.equal(hdrBackendFinalChecks(form({ ...ON, ...preset.settingsPatch }, {
    ...TX, ...preset.transcodePatch,
    audio_mode: 'remove', subtitle_mode: 'drop',
    copy_attachments: false, copy_metadata: false, copy_chapters: false,
  })).length, 2, 'with nothing to rewrap, only the probe-only unknowns remain')

  assert.match(hdrBackendFinalChecks({ ...ready, video: null }).join('\n'), /尚未导入素材/)
})

test('the limits are stated with the switch, in both places it is offered', () => {
  const text = HDR_LIMITATIONS.join('\n')

  assert.equal(HDR_LIMITATIONS.length, 4)
  assert.match(text, /HLG/)
  assert.match(text, /后端实际能力闸门决定/)
  assert.match(text, /预览播放器不是经过审计的 HDR 显示路径/)
  assert.match(text, /FP16 链路存在数值误差/)
  assert.match(text, /不声称无损/)
  // The refused scaling is the output extent. Working scaling is served, on the same
  // resolver the CPU path and the SDR backend use, and saying otherwise scares operators
  // off a control that works.
  assert.match(text, /输出幅面缩放、色调映射一律明确拒绝/)
  assert.match(text, /被拒的是输出幅面，不是工作幅面/)
  assert.match(text, /工作缩放 25%–100% 的整数百分比由 resolve_working_dims 推导/)
  assert.match(text, /resolve_working_dims/)
  assert.doesNotMatch(text, /HLG、双三层、缩放/)
  // The bridge does author and mux the Matroska deliverable; what the chain adds when the
  // request carries more than video is the existing copy pass, with -c:v copy.
  assert.match(text, /编码入口写出单视频流 \.mkv/)
  assert.match(text, /gtx\.build_mux_command/)
  assert.match(text, /-c:v copy/)
  assert.match(text, /没有第二次编码/)
  assert.doesNotMatch(text, /不追加 copy\/mux pass|不追加第二次封装/,
    'the rewrap exists now, so a limit that denies it is a lie in the other direction')
  assert.doesNotMatch(text, /不做封装|不执行封装/)
  for (const claim of ['更快', '加速', '性能提升', '零拷贝', '无损转换']) {
    assert.ok(!text.includes(claim), `limits advertise ${claim}`)
  }
})
