import { test } from 'node:test'
import assert from 'node:assert/strict'
import { isSdr10Source, sourcePrecisionNotice } from '../../frontend/src/services/sourcePrecision.ts'

const sdr = { path: 'sdr.mkv', filename: 'sdr.mkv', frames: 2, fps: 25, width: 320, height: 180, duration_seconds: .08, bit_depth: 10, is_hdr: false, pix_fmt: 'yuv420p10le', color_transfer: 'bt709', color_primaries: 'bt709', color_space: 'bt709', color_range: 'tv' }
test('8-bit SDR needs no precision notice', () => {
  assert.equal(sourcePrecisionNotice({ ...sdr, bit_depth: 8 }, false, 'gpu_transfer'), null)
  assert.equal(sourcePrecisionNotice(null, false, 'auto'), null)
})
test('ordinary 10-bit SDR export names its actual precision loss', () => {
  for (const backend of ['auto', 'legacy'] as const) {
    const notice = sourcePrecisionNotice(sdr, false, backend)!
    assert.equal(notice.type, 'warning')
    assert.match(notice.text, /10-bit SDR/)
    assert.match(notice.text, /精度不会保留/)
    assert.doesNotMatch(notice.text, /HDR/)
  }
})
test('SDR GPU export accepts tagged 10-bit BT709 without requesting HDR', () => {
  assert.equal(sourcePrecisionNotice(sdr, false, 'gpu_transfer'), null)
  const notice = sourcePrecisionNotice({ ...sdr, color_transfer: '' }, false, 'gpu_transfer')!
  assert.equal(notice.type, 'error')
  assert.match(notice.text, /后端会拒绝/)
  assert.match(notice.text, /BT.709/)
})
test('tagged ProRes 422 10-bit SDR source is admitted independent of output codec', () => {
  const prores={...sdr,codec_name:'prores',pix_fmt:'yuv422p10le'}
  assert.equal(isSdr10Source(prores),true)
  assert.equal(sourcePrecisionNotice(prores,false,'gpu_transfer'),null)
  assert.equal(isSdr10Source({...prores,color_range:'pc'}),false)
  assert.equal(isSdr10Source({...prores,codec_name:'other'}),false)
})
test('HDR opt-in and HDR source are separate export decisions', () => {
  const hdr = { ...sdr, is_hdr: true, color_transfer: 'smpte2084' }
  assert.match(sourcePrecisionNotice(hdr, false, 'gpu_transfer')!.text, /保留 HDR 需使用实验性 HDR10/)
  const requested = sourcePrecisionNotice(hdr, true, 'gpu_transfer')!
  assert.equal(requested.type, 'warning')
  assert.match(requested.text, /请求实验性 HDR10/)
  assert.doesNotMatch(requested.text, /降级|暂不支持真 HDR/)
})
