const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs'),path=require('node:path'),{createRequire}=require('node:module')
const ts=require('../../frontend/node_modules/typescript')
function compile(relative,load){
  const filename=path.resolve(__dirname,'../../frontend/src',relative),exports={}
  const code=ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,esModuleInterop:true}}).outputText
  new Function('require','exports',code)(load||createRequire(filename),exports)
  return exports
}
const translation=compile('i18n/translate.ts')
const {environmentText}=compile('services/environmentText.ts',()=>translation)

test('English environment overview translates the labels and readiness text shown in issue 1',()=>{
  assert.equal(translation.translateText('GPU 解码与编码','en'),'GPU decoding & encoding')
  assert.equal(environmentText('依赖检查通过，实际设备与模型支持在运行时验证','en'),
    'Dependency checks passed; device and model support are verified at runtime')
  assert.equal(environmentText('缺少：DLAA 运行库','en'),'Missing: DLAA runtime')
  assert.equal(environmentText('缺少：RTX VSR 运行库、NVIDIA VFG 运行库','en'),
    'Missing: RTX VSR runtime and NVIDIA VFG runtime')
})

test('all non-Chinese locales cover environment actions and composed missing-dependency labels',()=>{
  for(const locale of ['en','ja','ru']){
    for(const label of ['CUDA 驱动初始化','GPU 解码与编码','NVIDIA 驱动接口','RAFT CUDA 运算',
      'RTX VSR 运行库','NVIDIA VFG 运行库','预览与色彩转换','在软件设置中安装视频处理组件',
      '在软件设置中安装 RAFT 光流','重新安装匹配版本的原生组件']){
      assert.notEqual(translation.translateText(label,locale),label,locale+': '+label)
    }
    assert(!environmentText('缺少：DLAA 运行库、NVIDIA VFG 运行库',locale).includes('运行库'))
  }
})

test('technical names, paths and diagnostic payloads stay unchanged',()=>{
  const file='C:\\素材\\运行库\\model.dll'
  const diagnostic='GPU-output failed (3): unexpected CUDA 状态: 0x887a0005'
  assert.equal(environmentText(file,'en'),file)
  assert.equal(environmentText(diagnostic,'en'),diagnostic)
  assert.equal(environmentText('已找到：'+file+'；模型支持在创建会话时验证','en'),
    'Found: '+file+'; model support is verified when creating a session')
  assert.equal(environmentText('缺少：DLAA 运行库','zh-CN'),'缺少：DLAA 运行库')
})
