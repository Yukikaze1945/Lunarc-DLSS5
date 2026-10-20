import type { ExportBackend, VideoInfo } from '../types'

export function isSdr10Source(video: VideoInfo | null) {
  const prores422 = video?.codec_name === 'prores' && video.pix_fmt === 'yuv422p10le'
  return Boolean(video && !video.is_hdr && video.bit_depth === 10 &&
    (video.pix_fmt === 'yuv420p10le' || prores422) && video.color_transfer === 'bt709' &&
    video.color_primaries === 'bt709' && video.color_space === 'bt709' &&
    ['tv', 'mpeg', 'limited'].includes(video.color_range || ''))
}

// Match the actual export branches; preview precision is reported separately.
export function sourcePrecisionNotice(video: VideoInfo | null, hdrRequested: boolean, backend: ExportBackend) {
  if (!video || (!video.is_hdr && (video.bit_depth || 8) <= 8)) return null
  const source = video.is_hdr ? `${video.bit_depth || 10}-bit HDR` : `${video.bit_depth || 10}-bit SDR`
  if (hdrRequested) return { type: 'warning' as const, text: `输入为 ${source}；本作业请求实验性 HDR10 链路，是否可用由 HDR 条件校验决定。` }
  if (backend === 'gpu_transfer' && isSdr10Source(video)) return null
  if (backend === 'gpu_transfer') return {
    type: 'error' as const,
    text: `输入为 ${source}；${video.is_hdr ? '保留 HDR 需使用实验性 HDR10 链路。' : '10-bit SDR 需为 BT.709、有限范围的 4:2:0，或标记为 ProRes 的 4:2:2；当前色彩信息不符合，后端会拒绝。'}`,
  }
  return { type: 'warning' as const, text: `输入为 ${source}；当前普通增强链路按 8-bit SDR 处理，原始高位深精度不会保留。` }
}
