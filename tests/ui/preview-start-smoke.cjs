// Browser-only contract check. No backend process, GPU, decoder or model is opened.
const {chromium}=require('../../frontend/node_modules/playwright-core')
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict')
const root=path.resolve(__dirname,'../../frontend/dist')
const mime={'.js':'application/javascript','.css':'text/css','.html':'text/html'}
const server=http.createServer((req,res)=>{
  const requested=path.resolve(root,'.'+new URL(req.url,'http://localhost').pathname)
  const file=requested.startsWith(root+path.sep)&&fs.existsSync(requested)&&fs.statSync(requested).isFile()?requested:path.join(root,'index.html')
  res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(fs.readFileSync(file))
})
const until=async predicate=>{const deadline=Date.now()+15000;while(!predicate()){if(Date.now()>deadline)throw Error('Timed out waiting for preview state');await new Promise(r=>setTimeout(r,20))}}
async function run(){
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
  const browser=await chromium.launch({headless:true})
  const commands=[],native=[],calls=[],segments=[],errors=[],sources=[],imageRenders=[],imagePairs=[]
  const imageExports=[];let activeSaves=0,maxActiveSaves=0
  let imageCache=[],imageError='',frameDelay=0,queueSocket
  let suspended=true,video={path:'D:\\mock-a.mkv',filename:'mock-a.mkv',width:1920,height:1080,
    fps:24,frames:240,duration_seconds:10,pix_fmt:'yuv420p',bit_depth:8,is_hdr:false,media_type:'video'}
  const settings={style:0,working_scale:1,export_backend:'auto',render_pipeline:['nr'],reconstruction_mode:'single'}
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1000}})
    page.on('pageerror',e=>errors.push(e.message))
    await page.addInitScript(()=>{
      localStorage.setItem('lunarc.language.v1','zh-CN')
      window.__previewHandles=0
      window.electronAPI={isElectron:true,getComponents:async()=>null,onComponents:()=>()=>{},
        onBackendRecovery:callback=>{window.__backendRecovery=callback;return()=>{}},
        getPreviewWindow:async()=>{window.__previewHandles++;return {parent:'123'}},
        openFileDialog:async()=> 'D:\\mock-b.mkv'}
    })
    await page.routeWebSocket('ws://127.0.0.1:18923/ws/player',ws=>{
      ws.send(JSON.stringify({type:'preview_memory',suspended,releasing:false}))
      ws.onMessage(message=>{
        const data=JSON.parse(String(message));commands.push(data)
        if(['seek','play','set_view','refresh'].includes(data.action))setTimeout(()=>ws.send(JSON.stringify({
          type:'frame',encoding:'native_fp16',request_id:data.request_id,sequence:commands.length,
          frame:data.frame??0,preview:{native_display:true,layer_count:1,render_pipeline:['nr']}})),frameDelay)
      })
    })
    await page.routeWebSocket('ws://127.0.0.1:18923/ws/queue',ws=>{queueSocket=ws;ws.send(JSON.stringify({type:'init',jobs:[],active_id:null,queue_running:false}))})
    await page.route('http://127.0.0.1:18923/**',async route=>{
      const req=route.request(),url=new URL(req.url());let result={}
      if(url.pathname==='/api/source/frame'||url.pathname==='/api/image/thumbnail'||url.pathname==='/api/image/results/preview'){
        if(url.pathname==='/api/image/results/preview')imagePairs.push(Object.fromEntries(url.searchParams))
        sources.push({path:url.searchParams.get('path'),frame:Number(url.searchParams.get('frame')||0)})
        await route.fulfill({status:200,contentType:'image/png',headers:{'Access-Control-Allow-Origin':'*'},
          body:Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=','base64')});return
      }
      if(req.method()==='POST')calls.push(url.pathname)
      if(url.pathname==='/api/settings'){
        if(req.method()==='POST'){Object.assign(settings,req.postDataJSON());result={settings,preview_revision:1}}
        else result=settings
      }else if(url.pathname==='/api/video/current')result={video}
      else if(url.pathname==='/api/video/load'){video={...video,path:'D:\\mock-b.mkv',filename:'mock-b.mkv'};suspended=true;result={video}}
      else if(url.pathname==='/api/system')result={gpu:'Mock',backend_ready:true,adapter_memory_used_gb:15,adapter_memory_total_gb:16}
      else if(url.pathname==='/api/queue')result={jobs:[],active_id:null,queue_running:false}
      else if(url.pathname==='/api/preview/native'){native.push(req.method()==='POST'?req.postDataJSON():{});result={native:false}}
      else if(url.pathname.startsWith('/api/preview/memory')){
        if(url.pathname.endsWith('/resume'))suspended=false
        if(url.pathname.endsWith('/release'))suspended=true
        result={suspended,releasing:false}
      }else if(url.pathname==='/api/preview/segments'){
        const body=req.postDataJSON(),record={key:'cache-'+segments.length,job_id:'job-'+segments.length,
          output_path:'D:\\cache.mkv',frame_range:body.frame_range,original_offset_seconds:body.frame_range.start_frame/24,
          duration_seconds:(body.frame_range.end_frame-body.frame_range.start_frame)/24,status:'pending',ready:false,progress:0,body}
        segments.push(record);result=record
      }else if(url.pathname==='/api/preview/segments/list')result={segments:segments.filter(r=>r.body.source===req.postDataJSON().source).map(r=>({...r,ready:true,status:'completed',progress:100}))}
      else if(url.pathname==='/api/flow/providers')result={providers:[]}
      else if(url.pathname.startsWith('/api/runtime/dlssnr'))result={versions:[],sources:[],catalog:[],active:null}
      else if(url.pathname==='/api/image/models')result={models:[]}
      else if(url.pathname==='/api/image/formats')result={formats:{png:true,jpeg:true,webp:true,avif:false,heic:false,jxl:true,uhdr:false},
        hdr_formats:{jxl:{available:true,alpha:true,reason:''},uhdr:{available:true,alpha:false,reason:''}}}
      else if(url.pathname==='/api/image/render'){
        if(imageError){await route.fulfill({status:500,contentType:'application/json',headers:{'Access-Control-Allow-Origin':'*'},body:JSON.stringify({detail:imageError})});return}
        imageRenders.push(req.postDataJSON())
        const entry={id:'image-'+imageRenders.length,path:video.path,filename:video.filename,frame_index:req.postDataJSON().frame_index,
          width:257,height:145,hdr:false,created_at:Date.now(),host_bytes:300000,
          diagnostics:{image_memory_mode:'memory',layer_count:1,output_hdr:false,render_pipeline:['nr']}}
        imageCache=[entry,...imageCache].slice(0,2)
        result={result:entry,current:true,previous:imageCache[1]||null}
      }else if(url.pathname==='/api/image/results'){
        const eligible=imageCache.filter(item=>item.path===url.searchParams.get('path')&&(!url.searchParams.has('frame_index')||item.frame_index===Number(url.searchParams.get('frame_index'))))
        result={available:eligible.length>0,current:true,source_unchanged:true,result:eligible[0],previous:eligible[1],results:imageCache}
      }else if(url.pathname==='/api/image/export'){
        const body=req.postDataJSON();imageExports.push(body);activeSaves++;maxActiveSaves=Math.max(maxActiveSaves,activeSaves)
        await new Promise(resolve=>setTimeout(resolve,60));activeSaves--
        result={image:{path:body.output_path,rendered_result_id:imageCache[0]?.id}}
      }
      await route.fulfill({status:200,contentType:'application/json',headers:{'Access-Control-Allow-Origin':'*'},body:JSON.stringify(result)})
    })
    const renders=()=>commands.filter(c=>['seek','play','refresh','set_view'].includes(c.action))
    await page.goto(`http://127.0.0.1:${server.address().port}`)
    await page.getByRole('button',{name:'开启渲染',exact:true}).waitFor()
    await until(()=>sources.some(source=>source.path===video.path))
    await page.locator('.image-viewport canvas').waitFor()
    await page.getByText('原素材 · 未开启效果渲染',{exact:true}).waitFor()
    await page.getByRole('button',{name:'前进一帧',exact:true}).click()
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('lunarc:settings-applied',{detail:{revision:1}})))
    await page.getByRole('button',{name:'原图',exact:true}).click()
    await page.waitForTimeout(100)
    assert.equal(renders().length,0,'import, seek, settings and view changes must not render')
    assert.equal(await page.evaluate(()=>window.__previewHandles),0)
    assert.equal(native.length,0,'no native window before explicit start')
    await until(()=>sources.some(source=>source.frame===1))
    frameDelay=400
    await page.getByRole('button',{name:'开启渲染',exact:true}).click()
    await page.locator('.rail-mark[data-state=loading]').waitFor()
    assert.equal(await page.locator('.activity-glyph').evaluate(node=>getComputedStyle(node).animationName),'aperture-breathe')
    assert.equal(await page.locator('.preview-progress').count(),0,'loading has no explanatory strip')
    assert.equal(await page.locator('.gpu-memory-compact').getAttribute('data-level'),'critical')
    assert.ok(await page.locator('.image-viewport canvas').count(),'source remains visible while native frame is pending')
    await page.screenshot({path:path.resolve(__dirname,'../../build/preview-feedback-loading.png')})
    frameDelay=0
    await until(()=>renders().some(c=>c.action==='seek'))
    assert.ok(native.some(n=>n?.visible),'explicit start enables native rendering')
    await page.getByRole('button',{name:'停止渲染',exact:true}).click()
    await page.getByRole('button',{name:'开启渲染',exact:true}).waitFor({state:'visible'})
    await until(()=>!!queueSocket)
    queueSocket.send(JSON.stringify({type:'init',jobs:[{id:'export-check',name:'Export check',status:'running',progress:10,
      fps:10,eta:1,current_frame:1,total_frames:10,settings,input_path:video.path,output_path:'D:\\export.mkv',codec:'av1_nvenc',container:'mkv'}],active_id:'export-check',queue_running:true}))
    await page.locator('.rail-mark[data-state=exporting]').waitFor()
    await page.waitForFunction(()=>Number(document.querySelector('.rail-mark svg')?.getAnimations()[0]?.currentTime)>100)
    queueSocket.send(JSON.stringify({type:'init',jobs:[],active_id:null,queue_running:false}))
    await page.locator('.rail-mark[data-state=idle] svg[data-motion=settling]').waitFor()
    assert.notEqual(await page.locator('.rail-mark svg').evaluate(node=>getComputedStyle(node).transform),'matrix(1, 0, 0, 1, 0, 0)','state completion must not snap to the resting angle')
    await page.locator('.rail-mark svg[data-motion=rest]').waitFor()
    assert.equal(await page.locator('.rail-mark svg').evaluate(node=>getComputedStyle(node).transform),'matrix(1, 0, 0, 1, 0, 0)')
    const stoppedCount=renders().length
    await page.evaluate(()=>window.dispatchEvent(new Event('lunarc:settings-settled')))
    await page.getByRole('button',{name:'前进一帧',exact:true}).click()
    await page.waitForTimeout(100)
    assert.equal(renders().length,stoppedCount,'stopped preview must not restart from effects')
    await page.locator('.qi-advanced > summary').click()
    assert.equal(await page.getByRole('tab',{name:'时域',exact:true}).count(),1)
    await page.getByRole('tab',{name:'HDR',exact:true}).click()
    await page.getByRole('heading',{name:'HDR / scRGB',exact:true}).waitFor()
    await page.getByRole('tab',{name:'基础',exact:true}).click()
    assert.equal(await page.locator('.qi-advanced').getByText('运行库',{exact:true}).count(),0)
    assert.equal(await page.getByRole('combobox',{name:'画面风格',exact:true}).count(),0)
    await page.locator('.qi-advanced > summary').click()
    await page.getByRole('combobox',{name:'渲染方式',exact:true}).selectOption('single')
    await page.getByRole('button',{name:'渲染一帧',exact:true}).click()
    await until(()=>imageRenders.length===1)
    assert.equal(imageRenders[0].frame_index,2,'video single-shot must use the current frame')
    await page.getByRole('button',{name:'导出当前帧',exact:true}).click()
    await page.getByRole('heading',{name:'单帧导出',exact:true}).waitFor()
    await page.getByRole('button',{name:'保存图片',exact:true}).click()
    await page.getByText('已保存',{exact:true}).waitFor()
    assert.ok(calls.includes('/api/image/export'))
    assert.equal(imageRenders.length,1,'cached export must not re-render')
    assert.equal(renders().length,stoppedCount,'single-shot mode must not start live WS inference')
    await page.getByRole('button',{name:'重建工作台',exact:true}).click()
    imageRenders.length=0;imagePairs.length=0;imageCache=[]
    await page.getByRole('combobox',{name:'渲染方式',exact:true}).selectOption('prerender')
    await page.getByRole('combobox',{name:'预渲染长度',exact:true}).selectOption('f5')
    fs.mkdirSync(path.resolve(__dirname,'../../build'),{recursive:true})
    await page.screenshot({path:path.resolve(__dirname,'../../build/manual-preview-controls.png')})
    await page.getByRole('button',{name:'开启渲染',exact:true}).click()
    await until(()=>segments.length===1)
    assert.equal(segments[0].frame_range.end_frame-segments[0].frame_range.start_frame,5)
    assert.equal(segments[0].frame_range.start_frame,2)
    assert.equal(segments[0].body.settings.export_backend,'gpu_transfer')
    assert.ok(calls.lastIndexOf('/api/preview/memory/release')<calls.lastIndexOf('/api/preview/segments'))
    await page.getByRole('button',{name:'回看结果',exact:true}).waitFor()
    assert.equal(await page.locator('.segment-cached').count(),1)
    await page.getByRole('combobox',{name:'预渲染长度',exact:true}).selectOption('s2')
    await page.getByRole('button',{name:'开启渲染',exact:true}).click()
    await until(()=>segments.length===2)
    assert.equal(segments[1].frame_range.end_frame-segments[1].frame_range.start_frame,48)
    assert.equal(renders().length,stoppedCount,'cached rendering must not open live inference')
    await page.getByRole('button',{name:'导入',exact:true}).click()
    await page.locator('.dlss-material strong').filter({hasText:'mock-b.mkv'}).waitFor()
    await page.waitForTimeout(100)
    assert.equal(renders().length,stoppedCount,'new media must require new permission')
    await page.reload()
    await page.getByRole('button',{name:'开启渲染',exact:true}).waitFor()
    assert.equal(await page.evaluate(()=>window.__previewHandles),0,'reload must not restore render permission')
    // Images have a separate one-shot scheduler, never the video WS inference path.
    video={...video,path:'D:\\image.png',filename:'image.png',media_type:'image',width:257,height:145,frames:1,has_alpha:true}
    await page.reload()
    const renderOne=page.getByRole('button',{name:'渲染一帧',exact:true})
    await renderOne.waitFor()
    assert.equal(imageRenders.length,0,'image import must not render')
    await page.locator('.qi-advanced > summary').click()
    assert.equal(await page.getByRole('tab',{name:'时域',exact:true}).count(),0)
    assert.equal(await page.getByRole('tab',{name:'HDR',exact:true}).count(),0)
    assert.equal(await page.getByRole('heading',{name:'引导与光流',exact:true}).count(),0)
    assert.equal(await page.getByRole('heading',{name:'导出后端',exact:true}).count(),0)
    await page.locator('.qi-advanced > summary').click()

    assert.equal(await page.getByRole('combobox',{name:'图片渲染模式',exact:true}).inputValue(),'memory')
    assert.equal(await page.getByRole('checkbox',{name:'实时渲染',exact:true}).isChecked(),false)
    const videoCount=renders().length
    await renderOne.click()
    await until(()=>imageRenders.length===1&&imagePairs.length>=1)
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('lunarc:settings-applied',{detail:{revision:2}})))
    await page.waitForTimeout(100)
    assert.equal(imageRenders.length,1,'manual image must not render on setting changes')
    await renderOne.click()
    await until(()=>imageRenders.length===2&&imagePairs.some(pair=>pair.result_id==='image-2'))
    await page.getByRole('combobox',{name:'图片对比参考',exact:true}).selectOption('previous')
    await until(()=>imagePairs.some(pair=>pair.reference_id==='image-1'))
    assert.equal(imageRenders.length,2,'previous-result comparison must not run inference')
    await page.getByRole('checkbox',{name:'实时渲染',exact:true}).check()
    await until(()=>imageRenders.length===3&&imagePairs.some(pair=>pair.result_id==='image-3'))
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('lunarc:settings-applied',{detail:{revision:3}})))
    await until(()=>imageRenders.length===4&&imagePairs.some(pair=>pair.result_id==='image-4'))
    await page.getByRole('checkbox',{name:'实时渲染',exact:true}).uncheck()
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('lunarc:settings-applied',{detail:{revision:4}})))
    await page.waitForTimeout(100)
    assert.equal(imageRenders.length,4,'turning realtime off restores the one-shot gate')
    assert.equal(renders().length,videoCount,'images must not send WS inference commands')
    await page.screenshot({path:path.resolve(__dirname,'../../build/image-preview-controls.png')})
    await page.getByRole('button',{name:'导出图片',exact:true}).click()
    await page.getByRole('heading',{name:'图片导出',exact:true}).waitFor()
    const formatGroup=page.getByRole('group',{name:'图片格式',exact:true})
    await formatGroup.getByRole('button',{name:'JXL',exact:true}).waitFor()
    assert.ok(await formatGroup.getByRole('button',{name:'JXL',exact:true}).isEnabled())
    assert.ok(await formatGroup.getByRole('button',{name:'UHDR JPEG',exact:true}).isDisabled())
    const exportStart=imageExports.length
    await formatGroup.getByRole('button',{name:'JPEG',exact:true}).click()
    await formatGroup.getByRole('button',{name:'WEBP',exact:true}).click({modifiers:['Control']})
    await formatGroup.getByRole('button',{name:'PNG',exact:true}).click({modifiers:['Control']})
    const renderCount=imageRenders.length
    await page.getByRole('button',{name:'保存图片',exact:true}).click()
    await page.getByText('已保存',{exact:true}).waitFor()
    assert.deepEqual(imageExports.slice(exportStart).map(item=>item.format),['jpeg','webp','png'])
    assert.equal(maxActiveSaves,1,'formats must be saved sequentially')
    assert.equal(imageRenders.length,renderCount,'multiple formats share the rendered result')
    await page.getByRole('combobox',{name:'图片导出来源',exact:true}).selectOption('rerender')
    await formatGroup.getByRole('button',{name:'JPEG',exact:true}).click()
    await formatGroup.getByRole('button',{name:'WEBP',exact:true}).click({modifiers:['Control']})
    const rerenderStart=imageExports.length
    await page.getByRole('button',{name:'保存图片',exact:true}).click()
    await page.getByText('已保存',{exact:true}).waitFor()
    assert.deepEqual(imageExports.slice(rerenderStart).map(item=>item.render_source),['rerender','preview'])
    assert.equal(imageExports[rerenderStart+1].result_id,imageCache[0].id)
    await page.getByRole('combobox',{name:'图片导出来源',exact:true}).selectOption('preview')
    imageCache[0].hdr=true
    await page.evaluate(()=>window.dispatchEvent(new Event('lunarc:image-rendered')))
    const hdrButton=page.getByRole('button',{name:'HDR',exact:true})
    await hdrButton.waitFor();await page.waitForFunction(()=>![...document.querySelectorAll('button')].find(b=>b.textContent==='HDR')?.disabled)
    await hdrButton.click()
    await formatGroup.getByRole('button',{name:'UHDR JPEG',exact:true}).click()
    await page.locator('.image-filename span').filter({hasText:'.jpg'}).waitFor()
    assert.equal(await page.getByRole('checkbox',{name:'保留透明度',exact:true}).count(),0)
    await formatGroup.getByRole('button',{name:'JXL',exact:true}).click()
    await page.locator('.image-filename span').filter({hasText:'.jxl'}).waitFor()
    assert.ok(await page.getByRole('checkbox',{name:'保留透明度',exact:true}).isEnabled())
    await page.getByText('BT.2020 / PQ · 16-bit RGB',{exact:true}).waitFor()
    await page.screenshot({path:path.resolve(__dirname,'../../build/image-extra-codec-controls.png')})
    await page.getByRole('button',{name:'重建工作台',exact:true}).click()
    imageError='CUDA_ERROR_OUT_OF_MEMORY: cuMemAlloc input failed (2)'
    await renderOne.click()
    await page.locator('.processing-error').getByText('显存不足，本次处理已停止。请降低输出分辨率、减少处理项，或释放其他程序占用后重试。',{exact:true}).waitFor()
    await page.getByText('技术详情',{exact:true}).click()
    assert.equal(await page.locator('.processing-error pre').textContent(),imageError)
    await page.locator('.rail-mark[data-state=error]').waitFor()
    await page.evaluate(()=>window.__backendRecovery({phase:'recovering',message:'正在恢复后端'}))
    await page.locator('.rail-mark[data-state=recovering]').waitFor()
    await page.evaluate(()=>window.__backendRecovery({phase:'blocked',message:'恢复失败'}))
    await page.locator('.rail-mark[data-state=error]').waitFor()
    assert.deepEqual(errors,[])
    console.log('PASS: media-specific Advanced, video single-frame render/export, Ctrl format order and serial cache reuse, preview gates and HDR image controls')
  }finally{await browser.close()}
}
run().catch(error=>{console.error(error);process.exitCode=1}).finally(()=>server.close())
