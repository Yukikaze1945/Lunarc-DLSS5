import { test } from 'node:test'
import assert from 'node:assert/strict'

import { reconstructionExportBlockers } from '../../frontend/src/services/deliverValidation.ts'

const base = {
  reconstruction_mode: 'dual' as const,
  export_backend: 'gpu_transfer' as const,
  fusion_mode: 'sequential' as const,
  layer_scales: [0.75, 0.75],
  working_scale: 0.75,
}

test('sequential reconstruction permits layers with the same rounded scale', () => {
  assert.deepEqual(reconstructionExportBlockers(base), [])
})

test('multiscale reconstruction still requires distinct rounded scales', () => {
  assert.ok(reconstructionExportBlockers({...base, fusion_mode: 'catmull_baseline'}).some(message => message.includes('每层需使用不同')))
})

test('all reconstruction modes retain integral range and first layer checks', () => {
  assert.ok(reconstructionExportBlockers({...base, layer_scales: [0.751, 0.75]}).length > 0)
  assert.ok(reconstructionExportBlockers({...base, layer_scales: [0.5, 0.75]}).length > 0)
})
