// Validate the small package; never fall back to an offline runtime.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto')
module.exports=async function(){
  const resources=path.resolve(__dirname,'../../build/small-resources')
  const manifest=JSON.parse(fs.readFileSync(path.join(resources,'manifest.json'),'utf8'))
  if(manifest.mode!=='components')throw Error('Build packaging/build_small_resources.py first')
  for(const [name,entry] of Object.entries(manifest.files)){
    const file=path.resolve(resources,name)
    if(!file.startsWith(resources+path.sep)||!fs.existsSync(file)||fs.statSync(file).size!==entry.bytes)throw Error(`Missing resource: ${name}`)
    const hash=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(file))hash.update(chunk)
    if(hash.digest('hex')!==entry.sha256)throw Error(`Resource checksum mismatch: ${name}`)
  }
  for(const name of ['component-tools/uv.exe','components/base.lock','components/raft.lock','components/raft-model.json','components/raft-wheel.json','native-deps/avcodec-61.dll'])
    if(!manifest.files[name])throw Error(`Incomplete manifest: ${name}`)
  const presenter=path.resolve(__dirname,'../../native/gpu_native_presenter.dll')
  if(!fs.existsSync(presenter))throw Error('Missing required native/gpu_native_presenter.dll; run native/build_gpu_native_presenter.bat before packaging')
  console.log('Small installer resources verified; AI runtime is downloaded separately')
}
