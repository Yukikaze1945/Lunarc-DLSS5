const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto')
const {createRequire}=require('node:module')
const root=path.resolve(__dirname,'../..')
const frontendRequire=createRequire(path.join(root,'frontend/package.json'))

test('Electron resource filters include the native presenter used by the workbench',()=>{
  const config=frontendRequire('js-yaml').load(fs.readFileSync(path.join(root,'frontend/electron-builder.yml'),'utf8'))
  const {minimatch}=frontendRequire('minimatch')
  const filters=config.extraResources.find(item=>item.from==='..').filter
  const name='native/gpu_native_presenter.dll'
  assert(filters.some(pattern=>!pattern.startsWith('!')&&minimatch(name,pattern)))
  assert(!filters.some(pattern=>pattern.startsWith('!')&&minimatch(name,pattern.slice(1))))
})

test('beforePack refuses missing native preview DLL even with valid component resources',async()=>{
  const temp=fs.mkdtempSync(path.join(os.tmpdir(),'lunarc-packaging-'))
  try{
    const resources=path.join(temp,'build/small-resources')
    const names=['component-tools/uv.exe','components/base.lock','components/raft.lock',
      'components/raft-model.json','components/raft-wheel.json','native-deps/avcodec-61.dll']
    const content=Buffer.from('fixture'),hash=crypto.createHash('sha256').update(content).digest('hex')
    const files={}
    for(const name of names){
      const file=path.join(resources,name);fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,content)
      files[name]={bytes:content.length,sha256:hash}
    }
    fs.writeFileSync(path.join(resources,'manifest.json'),JSON.stringify({mode:'components',files}))
    const module={exports:{}}
    new Function('require','module','__dirname',fs.readFileSync(path.join(root,'frontend/scripts/validate-runtime.cjs'),'utf8'))(
      require,module,path.join(temp,'frontend/scripts'))
    await assert.rejects(module.exports(),/gpu_native_presenter\.dll/)
    fs.mkdirSync(path.join(temp,'native'));fs.writeFileSync(path.join(temp,'native/gpu_native_presenter.dll'),content)
    await module.exports()
  }finally{fs.rmSync(temp,{recursive:true,force:true})}
})
