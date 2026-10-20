import {test} from 'node:test'
import assert from 'node:assert/strict'
import {DEFAULT_SETTINGS,completeSettings} from '../../frontend/src/services/workspace.ts'
import {changeLayerNR,layerNRSettings,chooseReconstructionMode} from '../../frontend/src/services/multiScaleSettings.ts'
import {presetChanges,presetSettings} from '../../frontend/src/services/presets.ts'

test('A/B choices are retained per layer and mode changes preserve them',()=>{
  let settings={...DEFAULT_SETTINGS,...chooseReconstructionMode(DEFAULT_SETTINGS,'triple')}
  settings={...settings,...changeLayerNR(settings,1,'preset',2)}
  settings={...settings,...changeLayerNR(settings,2,'preset',1)}
  const restored=completeSettings(JSON.parse(JSON.stringify(settings)))
  assert.deepEqual([0,1,2].map(i=>layerNRSettings(restored,i).preset),[1,2,1])
  assert.equal(presetSettings(restored).layer_settings?.[1]?.preset,2)
  assert.equal(chooseReconstructionMode(restored,'dual').layer_settings?.[1]?.preset,2)
})

test('model changes appear by name without treating the enum as a numeric magnitude',()=>{
  const settings={...DEFAULT_SETTINGS,preset:2}
  const change=presetChanges(settings).find(c=>c.key==='preset')
  assert.equal(change?.label,'NR 模型')
  assert.equal(change?.before,'A');assert.equal(change?.after,'B')
  assert.equal(change?.delta,undefined)
  assert.equal(completeSettings({...DEFAULT_SETTINGS,preset:3}).preset,3)
})
