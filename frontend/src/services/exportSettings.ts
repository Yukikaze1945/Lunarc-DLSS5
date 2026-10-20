import type {TranscodeSettings} from '../types'
export const TRANSCODE_DRAFT_KEY = 'lunarc.transcodeDraft'
export const TRANSCODE_DRAFT_VERSION = 1
const DRAFT_ENUMS: Partial<Record<keyof TranscodeSettings, readonly string[]>> = {
  codec: ['av1_nvenc', 'hevc_nvenc', 'h264_nvenc', 'prores_ks'],
  container: ['.mkv', '.mp4', '.mov', '.webm'],
  rate_mode: ['cq', 'constqp', 'cbr', 'vbr'],
  peak_policy: ['auto', 'custom'],
  preset: ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'],
  tune: ['auto', 'hq', 'uhq', 'll', 'ull', 'lossless'],
  multipass: ['auto', 'disabled', 'qres', 'fullres'],
  decode_mode: ['auto', 'cuda', 'cpu'],
  finish_action: ['none', 'sleep', 'shutdown'],
  gop_mode: ['auto', 'manual'],
  b_frames: ['auto', '0', '1', '2', '3', '4', '5', '6', '7'],
  b_ref_mode: ['auto', 'disabled', 'middle', 'each'],
  lookahead: ['auto', '0', '8', '16', '24', '32'],
  lookahead_level: ['auto', '0', '1', '2', '3'],
  aq_mode: ['spatial', 'temporal', 'disabled'],
  temporal_filter: ['auto', '0', '1', '2', '3', '4'],
  split_encode: ['auto', 'disabled', '2', '3', '4'],
  profile: ['auto', 'main', 'main10', 'high', 'hq'],
  tier: ['auto', 'main', 'high'],
    resolution_mode: ['original', 'source', 'double', 'half', '7680x4320', '5120x2880', '3840x2160', '2560x1440', '1920x1080', '1280x720', 'custom_size', 'custom_factor'],
  scale_algorithm: ['lanczos', 'bicubic', 'bilinear', 'nearest'],
  pixel_format: ['auto', 'yuv420p', 'nv12', 'p010le', 'p016le', 'nv16', 'p210le', 'p216le', 'yuv422p10le', 'yuv444p', 'yuv444p10msble', 'yuv444p16le'],
  color_preset: ['preserve', 'bt709_limited', 'bt709_full', 'bt2020_sdr', 'bt2020_pq', 'bt2020_hlg', 'hdr_to_sdr', 'custom'],
  custom_color_range: ['tv', 'pc'],
  audio_mode: ['copy', 'aac', 'opus', 'flac', 'pcm', 'remove'],
  audio_bitrate: ['128k', '192k', '256k', '320k', '512k'],
  subtitle_mode: ['copy', 'convert', 'drop'],
}

export function readTranscodeDraft(defaults: TranscodeSettings): TranscodeSettings {
  try {
    const raw = localStorage.getItem(TRANSCODE_DRAFT_KEY)
    if (!raw) return defaults
    const parsed = JSON.parse(raw)
    const saved = parsed?.settings
    if (!saved || typeof saved !== 'object' || Array.isArray(saved)) return defaults
    // Merge known, type-compatible fields so a schema upgrade keeps everything it
    // still understands while new settings inherit safe defaults.
    const merged = { ...defaults }
    for (const key of Object.keys(defaults) as (keyof TranscodeSettings)[]) {
      const value = saved[key]
      const fallback = defaults[key]
      const allowed = DRAFT_ENUMS[key]
      const numericBounds: Partial<Record<keyof TranscodeSettings, [number, number]>> = {
        cq: [0, 51], qp: [0, 51], gop: [1, 100000], aq_strength: [1, 15],
        custom_width: [2, 16384], custom_height: [2, 16384], custom_factor: [0.1, 8],
      }
      const bounds = numericBounds[key]
      const invalidNumber = typeof value === 'number' && (!Number.isFinite(value) ||
        (bounds !== undefined && (value < bounds[0] || value > bounds[1])) ||
        ((key === 'custom_width' || key === 'custom_height') && value % 2 !== 0))
      if (typeof value === typeof fallback && !invalidNumber &&
          !(typeof value === 'string' && allowed && !allowed.includes(value))) {
        ;(merged as any)[key] = value
      }
    }
    return merged
  } catch {
    return defaults
  }
}

export const DEFAULT_EXPORT_SETTINGS: TranscodeSettings = {
  codec: 'av1_nvenc',
  container: '.mkv',
  suffix: '_AV1_P7_UHQ_Q16',
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
  gpu: '-1',
  decode_mode: 'auto',
  finish_action: 'none',
  gop_mode: 'auto',
  gop: 240,
  b_frames: 'auto',
  b_ref_mode: 'auto',
  lookahead: 'auto',
  lookahead_level: 'auto',
  aq_mode: 'spatial',
  aq_strength: 8,
  weighted_prediction: false,
  temporal_filter: 'auto',
  split_encode: 'auto',
  profile: 'auto',
  level: 'auto',
  tier: 'auto',
  resolution_mode: 'original',
  custom_width: 1920,
  custom_height: 1080,
  custom_factor: 1.0,
  keep_aspect: true,
  scale_algorithm: 'lanczos',
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
  overwrite_output: false,
  skip_existing: true,
  extra_args: '',
}
