import { test } from 'node:test'
import assert from 'node:assert/strict'

import { completeSettings, DEFAULT_SETTINGS } from '../../frontend/src/services/workspace.ts'
import { DEFAULT_POST, DEFAULT_TEMPORAL } from '../../frontend/src/services/qualityControls.ts'

test('GPU Transfer is the default export backend', () => {
  assert.equal(DEFAULT_SETTINGS.export_backend, 'gpu_transfer')
})

test('legacy automatic export migrates only when completed settings have export controls', () => {
  assert.equal(completeSettings({export_backend: 'auto'}).export_backend, 'gpu_transfer')
  const noControls = completeSettings({
    export_backend: 'auto',
    temporal_controls: {...DEFAULT_TEMPORAL, scene_detection: false},
    post_controls: DEFAULT_POST,
  })
  assert.equal(noControls.export_backend, 'auto')
})

test('explicit saved export backend remains unchanged', () => {
  assert.equal(completeSettings({export_backend: 'legacy'}).export_backend, 'legacy')
  assert.equal(completeSettings({export_backend: 'gpu_transfer'}).export_backend, 'gpu_transfer')
})
