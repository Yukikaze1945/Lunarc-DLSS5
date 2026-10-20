import { test } from 'node:test'
import assert from 'node:assert/strict'
import path from 'node:path'

import { resolveAppProtocolPath } from '../../frontend/electron/appProtocolPath.ts'

test('app protocol resolves renderer routes and static assets inside the renderer directory', () => {
  const root = path.win32.resolve('C:\\Lunarc\\dist')
  assert.equal(resolveAppProtocolPath('app://studio/', root, path.win32), path.win32.join(root, 'index.html'))
  assert.equal(resolveAppProtocolPath('app://studio/assets/app.js', root, path.win32), path.win32.join(root, 'assets', 'app.js'))
  assert.equal(resolveAppProtocolPath('app://studio/workspace', root, path.win32), path.win32.join(root, 'workspace'))
})

test('app protocol rejects untrusted hosts, malformed encoding, and encoded separators', () => {
  const root = path.win32.resolve('C:\\Lunarc\\dist')
  for (const url of [
    'app://elsewhere/index.html',
    'app://studio.evil/index.html',
    'app://user@studio/index.html',
    'app://studio/%ZZ',
    'app://studio/%2f..%2fsecret',
    'app://studio/%5c..%5csecret',
    'app://studio/..%20/secret',
    'app://studio/..%2e%20/secret',
    'app://studio/assets%20/app.js',
    'app://studio/%01',
  ]) assert.equal(resolveAppProtocolPath(url, root, path.win32), null, url)
})

test('app protocol rejects traversal, drive paths, UNC paths, and sibling-prefix escapes on Windows paths', () => {
  const root = path.win32.resolve('C:\\Lunarc\\dist')
  for (const url of [
    'app://studio/../secret',
    'app://studio/%2e%2e/secret',
    'app://studio/C:/secret',
    'app://studio/%5c%5cserver/share',
    'app://studio/..%5cLunarc%5cdist-evil/secret',
  ]) assert.equal(resolveAppProtocolPath(url, root, path.win32), null, url)

  const sibling = path.win32.resolve('C:\\Lunarc\\dist-evil\\index.html')
  assert.equal(path.win32.relative(root, sibling).startsWith('..'), true)
})
