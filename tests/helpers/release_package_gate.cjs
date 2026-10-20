// Packaging smoke only: payload parity, component manifest and fresh first start.
// Usage: node tests/helpers/release_package_gate.cjs <unpacked-dir> <fresh-evidence-dir>
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const {_electron}=require(path.join(root,'frontend/node_modules/playwright'));
const [payload,out]=process.argv.slice(2).map(p=>path.resolve(p));
const resources=path.join(payload,'resources');
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const result={passed:false,scope:'package files and first-start UI; no GPU regression or new component downloads',checks:[]};
(async()=>{
  assert(!fs.existsSync(out),'Use a fresh evidence directory');fs.mkdirSync(out,{recursive:true});
  const version=JSON.parse(fs.readFileSync(path.join(root,'frontend/package.json'),'utf8')).version;
  assert.equal(JSON.parse(fs.readFileSync(path.join(resources,'app/package.json'),'utf8')).version,version);
  result.version=version;
  const excluded=new Set(['nvdec_bridge_p010.dll','nvenc_bridge_sdr10.dll','gpu_native_presenter.dll']);
  const native=fs.readdirSync(path.join(root,'native')).filter(n=>n.endsWith('.dll')&&!excluded.has(n));
  for(const name of native)assert.equal(sha(path.join(resources,'native',name)),sha(path.join(root,'native',name)),name);
  for(const name of fs.readdirSync(root).filter(n=>n.endsWith('.py')))
    assert.equal(sha(path.join(resources,name)),sha(path.join(root,name)),name);
  for(const folder of ['dist','dist-electron'])for(const name of fs.readdirSync(path.join(root,'frontend',folder),{recursive:true})){
    const source=path.join(root,'frontend',folder,name);
    if(fs.statSync(source).isFile()&&!name.endsWith('.map'))assert.equal(sha(path.join(resources,'app',folder,name)),sha(source),name);
  }
  const manifest=JSON.parse(fs.readFileSync(path.join(resources,'manifest.json'),'utf8'));
  assert.equal(manifest.mode,'components');
  for(const [name,entry] of Object.entries(manifest.files)){
    const file=path.resolve(resources,name);assert(file.startsWith(resources+path.sep));
    assert.equal(fs.statSync(file).size,entry.bytes,name);assert.equal(sha(file),entry.sha256,name);
  }
  for(const name of ['runtime','torch','nvngx_dlssnr.dll','notice-archives'])assert(!fs.existsSync(path.join(resources,name)),name);
  const forbidden=/^(?:nvngx.*\.dll|nvcuda\.dll|nvml\.dll|nvencodeapi.*\.dll|nvofapi64\.dll|nvVideo.*\.dll)$/i;
  for(const dir of ['native','native-deps'])for(const name of fs.readdirSync(path.join(resources,dir)))assert(!forbidden.test(name),name);
  assert(fs.readFileSync(path.join(resources,'licenses/Lunarc-LICENSE.txt'),'utf8').includes('GNU GENERAL PUBLIC LICENSE'));
  assert(fs.existsSync(path.join(resources,'licenses/project-notices/react-range-MIT.txt')));
  result.nativeFiles=native;result.manifestFiles=Object.keys(manifest.files).length;
  result.checks.push('current Python, frontend and native DLL hashes match','component hashes verified','GPL and dependency notices','AI runtimes excluded');
  const env={...process.env,E2E_TEST:'1',E2E_BACKEND_PORT:'18993',PYTHON_PATH:'Z:/missing/python.exe',PYTHONHOME:'Z:/missing',PYTHONPATH:'Z:/missing'};
  delete env.ELECTRON_RUN_AS_NODE;
  const app=await _electron.launch({executablePath:path.join(payload,'Lunarc.exe'),args:['--user-data-dir='+path.join(out,'profile')],env,timeout:30000});
  try{
    const page=await app.firstWindow(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.getByRole('heading',{name:'开始使用',exact:true}).waitFor({timeout:30000});
    const state=await page.evaluate(async()=>({startup:await window.electronAPI.getBackendStartup(),components:await window.electronAPI.getComponents(),update:await window.electronAPI.getAppUpdate()}));
    assert.equal(state.startup.state,'setup');assert.equal(state.components.requiresSetup,true);
    assert.equal(state.update.currentVersion,version);assert.equal(state.update.packaged,true);
    assert.deepEqual(errors,[]);result.firstStart=state;
    await page.screenshot({path:path.join(out,'first-start.png')});
    result.checks.push('packaged Electron opens first-start tutorial','missing components correctly reported','updater reports exact package version');
    result.passed=true;
  }finally{await app.close()}
})().catch(e=>{result.error=String(e);console.error(e);process.exitCode=1}).finally(()=>{if(out&&fs.existsSync(out))fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({passed:result.passed,version:result.version,checks:result.checks,error:result.error}));});
