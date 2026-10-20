import type { DLSSSettings, ReconstructionMode } from '../types'

const percent = (scale: number) => Math.round(scale * 100)
const toScale = (value: number) => value / 100

export const layerNRKeys = ['preset', 'style', 'intensity', 'local_tone', 'local_struct',
  'skin_struct', 'use_auto_mask', 'ui_correction'] as const
export type LayerNRKey = typeof layerNRKeys[number]

export function layerNRSettings(settings: DLSSSettings, index: number) {
  const base = Object.fromEntries(layerNRKeys.map(key => [key, settings[key]])) as
    Pick<DLSSSettings, LayerNRKey>
  return index === 0 ? base : { ...base, ...settings.layer_settings?.[index] }
}

export function changeLayerNR(settings: DLSSSettings, index: number,
  key: LayerNRKey, value: number): Partial<DLSSSettings> {
  if (index === 0) return { [key]: value }
  const layers = Array.from({ length: 3 }, (_, at) =>
    at === 0 ? {} : { ...layerNRSettings(settings, at) })
  layers[index] = { ...layers[index], [key]: value }
  return { layer_settings: layers }
}

export function activeLayerCount(mode: ReconstructionMode): number {
  return mode === 'triple' ? 3 : mode === 'dual' ? 2 : 1
}

export function chooseReconstructionMode(
  settings: DLSSSettings, mode: ReconstructionMode,
): Partial<DLSSSettings> {
  const first = percent(settings.working_scale)
  if (mode === 'single') {
    return { reconstruction_mode: 'single', layer_scales: [toScale(first)],
      layer_weights: [1], fusion_mode: 'catmull_baseline' }
  }
  return {
    reconstruction_mode: mode,
    export_backend: 'gpu_transfer',
    layer_scales: Array.from({length:activeLayerCount(mode)}, (_, i) =>
      i === 0 || settings.layer_execution !== 'sequential' ? toScale(first) : settings.layer_scales?.[i] ?? toScale(first)),
    layer_weights: Array.from({length:activeLayerCount(mode)}, (_, index) => settings.layer_weights?.[index] || 1),
    // Stacking consumes the previous result, using separate Feature sessions.
    fusion_mode: 'sequential',
    layer_execution: 'sequential',
    layer_settings: Array.from({ length: 3 }, (_, index) =>
      index === 0 ? {} : { ...layerNRSettings(settings, index) }),
  }
}

export function chooseWorkingScale(
  settings: DLSSSettings, scale: number,
): Pick<DLSSSettings, 'working_scale' | 'layer_scales'> {
  const next = percent(scale)
  const previous = percent(settings.working_scale)
  const count = activeLayerCount(settings.reconstruction_mode || 'single')
  const scales = [previous, ...(settings.layer_scales || []).slice(1).map(percent)]
    .slice(0, count)
  const duplicate = scales.indexOf(next, 1)
  if (duplicate >= 1 && settings.fusion_mode !== 'sequential') scales[duplicate] = previous
  scales[0] = next
  return { working_scale: toScale(next), layer_scales: scales.map(toScale) }
}

export function chooseLayerScale(settings: DLSSSettings, index: number, scale: number): number[] {
  const count = activeLayerCount(settings.reconstruction_mode || 'single')
  const next = percent(scale)
  const scales = [percent(settings.working_scale),
    ...(settings.layer_scales || []).slice(1).map(percent)].slice(0, count)
  if (index < 1 || index >= count || next < 25 || next > 100 ||
      (settings.fusion_mode !== 'sequential' && scales.some((value, at) => at !== index && value === next))) {
    throw new RangeError('每层必须使用不同的 25–100% 整数比例')
  }
  scales[index] = next
  return scales.map(toScale)
}

export function chooseFusionMode(settings: DLSSSettings, fusion_mode: DLSSSettings['fusion_mode']): Partial<DLSSSettings> {
  const count = activeLayerCount(settings.reconstruction_mode)
  if (fusion_mode === 'sequential') return {fusion_mode, layer_execution:'sequential'}
  const candidates = [percent(settings.working_scale), ...(settings.layer_scales||[]).slice(1).map(percent),75,50,100,67,25]
  const scales = [...new Set(candidates)].slice(0,count).map(toScale)
  return {fusion_mode,layer_execution:'multiscale',layer_scales:scales}
}
