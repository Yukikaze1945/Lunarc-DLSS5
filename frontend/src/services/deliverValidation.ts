import type { DLSSSettings } from '../types'

export interface ReconstructionExportSettings {
  reconstruction_mode: DLSSSettings['reconstruction_mode']
  export_backend: DLSSSettings['export_backend']
  fusion_mode: DLSSSettings['fusion_mode']
  layer_scales?: number[]
  working_scale: number
  layer_execution?: 'sequential' | 'multiscale'
}

/** Returns the blocking export messages for independent reconstruction layers. */
export function reconstructionExportBlockers(settings: ReconstructionExportSettings): string[] {
  if (settings.reconstruction_mode === 'single') return []

  const issues: string[] = []
  if (settings.export_backend !== 'gpu_transfer') {
    issues.push('Dual / Triple 独立多尺度层只支持 GPU Transfer 导出')
  }

  const count = settings.reconstruction_mode === 'triple' ? 3 : 2
  const scales = settings.layer_scales || []
  if (scales.length !== count ||
      Math.round((scales[0] || 0) * 100) !== Math.round(settings.working_scale * 100) ||
      scales.some(scale => !Number.isFinite(scale) || scale < 0.25 || scale > 1 ||
        Math.abs(scale * 100 - Math.round(scale * 100)) > 1e-6) ||
      (settings.fusion_mode !== 'sequential' &&
        new Set(scales.map(scale => Math.round(scale * 100))).size !== count)) {
    issues.push(settings.fusion_mode === 'sequential'
      ? '每层需使用 25–100% 整数比例，且第 1 层须等于工作分辨率'
      : '每层需使用不同的 25–100% 整数比例，且第 1 层须等于工作分辨率')
  }
  return issues
}
