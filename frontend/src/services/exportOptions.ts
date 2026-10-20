import type {DLSSSettings, TranscodeSettings, VideoInfo} from '../types'
import {isSdr10Source} from './sourcePrecision.ts'
import {isHdrRequested} from './hdrExport.ts'

// These are the formats the existing encoder entries actually consume.
export function exportOptions(settings: DLSSSettings, video: VideoInfo | null, ts: TranscodeSettings) {
  const gpu = settings.export_backend === 'gpu_transfer'
  const wide = (!!settings.truehdr_enabled && settings.render_stage_enabled?.truehdr !== false) || isHdrRequested(settings) || (gpu && isSdr10Source(video))
  const prores = ts.codec === 'prores_ks'
  if (prores) return {
    gpu, wide: true,
    codec: gpu ? [...(wide ? ['hevc_nvenc', 'av1_nvenc'] : ['av1_nvenc', 'hevc_nvenc', 'h264_nvenc']), 'prores_ks'] :
      wide ? ['hevc_nvenc', 'av1_nvenc'] : ['av1_nvenc', 'hevc_nvenc', 'h264_nvenc'],
    container: ['.mov'], pixel_format: ['yuv422p10le'], profile: ['hq'],
    color_preset: ['preserve'], resolution_mode: null as string[] | null,
    audio_mode: null as string[] | null, b_ref_mode: ['auto'],
  }
  return {
    gpu, wide,
    codec: wide
      ? ['hevc_nvenc', 'av1_nvenc', ...(gpu ? ['prores_ks'] : [])]
      : ['av1_nvenc', 'hevc_nvenc', 'h264_nvenc', ...(gpu ? ['prores_ks'] : [])],
    container: wide ? ['.mkv'] : ts.codec === 'av1_nvenc' ? ['.mkv', '.mp4', '.mov', '.webm'] : ['.mkv', '.mp4', '.mov'],
    pixel_format: wide ? ['auto', 'p010le'] : gpu ? ['auto', 'yuv420p'] : ['auto', 'yuv420p', 'nv12', 'nv16', 'yuv444p'],
    profile: wide ? ts.codec === 'av1_nvenc' ? ['auto', 'main'] : ['auto', 'main10'] : ts.codec === 'h264_nvenc' ? ['auto', 'main', 'high'] : ['auto', 'main'],
    color_preset: gpu || wide ? ['preserve'] : ['preserve', 'bt709_limited', 'bt709_full', 'hdr_to_sdr', 'custom'],
    resolution_mode: null as string[] | null,
    audio_mode: ts.container === '.webm' ? ['opus', 'remove'] : null,
    b_ref_mode: ts.codec === 'av1_nvenc' ? ['auto', 'disabled', 'middle'] : ['auto', 'disabled', 'middle', 'each'],
  }
}

export function normalizeExportOptions(ts: TranscodeSettings, settings: DLSSSettings, video: VideoInfo | null): TranscodeSettings {
  const next = {...ts}
  let options = exportOptions(settings, video, next)
  if (!options.codec.includes(next.codec)) next.codec = options.codec[0]
  options = exportOptions(settings, video, next)
  if (!options.container.includes(next.container)) next.container = '.mkv'
  options = exportOptions(settings, video, next)
  if (next.codec === 'prores_ks') {
    next.container = '.mov'; next.profile = 'hq'; next.pixel_format = 'yuv422p10le'
    next.decode_mode = 'cpu'; next.weighted_prediction = false; next.b_frames = 'auto'; next.b_ref_mode = 'auto'
    next.extra_args = ''
  }
  for (const key of ['pixel_format', 'profile', 'color_preset', 'resolution_mode', 'audio_mode', 'b_ref_mode'] as const) {
    const allowed = options[key]
    if (allowed && !allowed.includes(next[key])) next[key] = allowed[0]
  }
  if (next.codec === 'h264_nvenc' && next.tune === 'uhq') next.tune = 'hq'
  if (next.codec === 'av1_nvenc') next.weighted_prediction = false
  if (next.weighted_prediction) {
    next.b_frames = '0'; next.b_ref_mode = 'disabled'
    if (next.codec === 'hevc_nvenc') next.split_encode = 'disabled'
  }
  if (next.container !== '.mkv') next.copy_attachments = false
  if (options.gpu || options.wide) {
    next.custom_color_space = ''; next.custom_color_primaries = ''
    next.custom_color_transfer = ''; next.custom_color_range = ''
  }
  if (options.wide) next.extra_args = ''
  return JSON.stringify(next) === JSON.stringify(ts) ? ts : next
}
