// localized-render-sinks
import {ProcessingError} from '../components/ProcessingError'
import {ui} from '../i18n/locale'
import {useEffect,useRef,useState} from 'react'
import {ArrowUpRight,Folder,Check,Image as ImageIcon} from 'lucide-react'
import type {VideoInfo,DLSSSettings} from '../types'
import {exportImage,imageThumbnail,imageExportCapabilities,openNativeDirDialog,imageResults,renderedImagePreview,sourceFrame} from '../services/api'
import type {HdrImageFormat,ImageResultState} from '../services/api'
import {ImageViewport} from '../components/ImageViewport'
import {renderStages,hasScaleStages,stageName,vsrOutputSize} from '../services/renderPipeline'
import './image-deliver.css'

export function ImageDeliverPage({video,settings,onStatusChange,currentFrame=0,isActive=true}:{video:VideoInfo;settings:DLSSSettings;currentFrame?:number;isActive?:boolean;onStatusChange?:(message:string)=>void}) {
  const videoFrame=video.media_type!=='image'
  const frameIndex=videoFrame?currentFrame:undefined
  const [format,setFormat]=useState('png')
  const [exportFormats,setExportFormats]=useState<string[]>(['png'])
  const [saveProgress,setSaveProgress]=useState(0)
  const [colorMode,setColorMode]=useState<'sdr'|'hdr10'>('sdr')
  const [renderSource,setRenderSource]=useState<'preview'|'rerender'>('preview')
  const [cached,setCached]=useState<ImageResultState>({available:false,results:[]})
  const [resultId,setResultId]=useState('')
  const selectedByUser=useRef(false)
  const [allowStale,setAllowStale]=useState(false)
  const [directory,setDirectory]=useState('')
  const [name,setName]=useState('')
  const [quality,setQuality]=useState(95)
  const [alpha,setAlpha]=useState(true)
  const [overwrite,setOverwrite]=useState(false)
  const [busy,setBusy]=useState(false)
  const [error,setError]=useState('')
  const [result,setResult]=useState<any>(null)
  const [thumbnail,setThumbnail]=useState('')
  const [formats,setFormats]=useState<Record<string,boolean>>({png:true,jpeg:true,webp:true,avif:false,heic:false,heif:false,uhdr:false,jxl:false})
  const [hdrFormats,setHdrFormats]=useState<Record<string,HdrImageFormat>>({})
  const [previewZoom,setPreviewZoom]=useState(1)
  const [previewPan,setPreviewPan]=useState({x:.5,y:.5})
  const eligible=cached.results.filter(item=>item.path.replace(/\\/g,'/').toLowerCase()===video.path.replace(/\\/g,'/').toLowerCase()&&item.frame_index===frameIndex)
  const selected=eligible.find(item=>item.id===resultId)
  const cacheCurrent=Boolean(selected&&selected.id===cached.result?.id&&cached.current)
  const thumbnailPaired=Boolean(renderSource==='preview'&&selected&&(!result||result.color_mode==='hdr10'))
  useEffect(()=>{
    if(!isActive)return
    let active=true,timer:ReturnType<typeof setTimeout>
    const refresh=()=>{clearTimeout(timer);timer=setTimeout(()=>{void imageResults(video.path,frameIndex).then(value=>{
      if(!active)return
      setCached(value);setResultId(previous=>selectedByUser.current&&value.results.some(item=>item.id===previous)?previous:value.result?.id||'')
    }).catch(()=>{})},100)}
    refresh();window.addEventListener('lunarc:image-rendered',refresh);window.addEventListener('lunarc:settings-settled',refresh)
    return()=>{active=false;clearTimeout(timer);window.removeEventListener('lunarc:image-rendered',refresh);window.removeEventListener('lunarc:settings-settled',refresh)}
  },[video.path,settings,frameIndex,isActive])
  useEffect(()=>{setAllowStale(false)},[video.path,resultId,renderSource,frameIndex])
  useEffect(()=>{if(renderSource==='preview'&&selected&&!selected.hdr)setColorMode('sdr')},[renderSource,selected?.id])
  useEffect(()=>{let active=true;imageExportCapabilities().then(value=>{if(active){setFormats(value.formats);setHdrFormats(value.hdr_formats)}}).catch(e=>{if(active)setError(e.message)});return()=>{active=false}},[])
  useEffect(()=>{
    const slash=Math.max(video.path.lastIndexOf('\\'),video.path.lastIndexOf('/'))
    setDirectory(video.path.slice(0,slash));setName(video.filename.replace(/\.[^.]+$/,'')+(videoFrame?`_frame_${currentFrame+1}`:'_lunarc'))
    setResult(null);setError('')
    setCached({available:false,results:[]});setResultId('')
    selectedByUser.current=false
    setPreviewZoom(1);setPreviewPan({x:.5,y:.5})
  },[video.path,video.filename,frameIndex])
  useEffect(()=>{
    if(!isActive)return
    let cancelled=false,url=''
    setThumbnail('');setPreviewZoom(1);setPreviewPan({x:.5,y:.5})
    const request=thumbnailPaired&&selected?renderedImagePreview(selected.id):videoFrame&&(!result||result.color_mode==='hdr10')?sourceFrame(video.path,currentFrame):imageThumbnail(result?.color_mode==='hdr10'?video.path:result?.path||video.path,0)
    request.then(blob=>{
      if(cancelled)return
      url=URL.createObjectURL(blob);setThumbnail(url)
    }).catch(()=>{})
    return()=>{cancelled=true;if(url)URL.revokeObjectURL(url)}
  },[video.path,result?.path,thumbnailPaired,selected?.id,frameIndex,isActive])
  const chain=renderStages(settings).filter(stage=>stage!=='vfg')
  const hdr=colorMode==='hdr10'
  const hasTrueHDR=chain.includes('truehdr')
  const hdrAllowed=renderSource==='preview'?Boolean(selected?.hdr):(hasTrueHDR||video.is_hdr)
  const formatAvailable=(value:string)=>hdr?Boolean(hdrFormats[value]?.available):Boolean(formats[value])
  const hdrAvailable=Object.values(hdrFormats).some(value=>value.available)
  const alphaAvailable=exportFormats.some(value=>!['jpeg','uhdr'].includes(value)&&(!hdr||Boolean(hdrFormats[value]?.alpha)))
  const chooseFormat=(value:string,additive:boolean)=>{
    const next=additive?(exportFormats.includes(value)?exportFormats.filter(item=>item!==value):[...exportFormats,value]):[value]
    if(!next.length)return
    setExportFormats(next);setFormat(next[next.length-1]);setError('')
  }
  const changeColor=(value:'sdr'|'hdr10')=>{
    setColorMode(value);setError('')
    const available=exportFormats.filter(key=>value==='hdr10'?hdrFormats[key]?.available:formats[key])
    const next=available.length?available:[value==='hdr10'?['avif','heic','heif','uhdr','jxl'].find(key=>hdrFormats[key]?.available)||'avif':'png']
    setExportFormats(next);setFormat(next[next.length-1])
  }
  const dimensions=renderSource==='preview'&&selected?selected:hasScaleStages(settings)?vsrOutputSize(settings,video):video
  const extension=['jpeg','uhdr'].includes(format)?'jpg':format
  const hdrEncoding=format==='uhdr'?'JPEG SDR + HDR 增益图':format==='jxl'?'BT.2020 / PQ · 16-bit RGB':'BT.2020 / PQ · 10-bit'
  const outputFor=(value:string)=>`${directory.replace(/[\\/]+$/,'')}\\${name}.${['jpeg','uhdr'].includes(value)?'jpg':value}`
  const valid=Boolean(directory&&name.trim()&&!/[<>:"/\\|?*\x00-\x1f]/.test(name))
  const run=async()=>{
    setBusy(true);setSaveProgress(0);setError('');setResult(null)
    const savedPaths:string[]=[]
    let source=renderSource,cacheId=renderSource==='preview'?resultId:undefined
    try {
      for(const [index,nextFormat] of exportFormats.entries()){
        setSaveProgress(index+1);onStatusChange?.(`正在保存 ${index+1}/${exportFormats.length} · ${nextFormat.toUpperCase()}`)
        const saved=await exportImage({path:video.path,output_path:outputFor(nextFormat),format:nextFormat,quality,
          preserve_alpha:alpha&&(!hdr||Boolean(hdrFormats[nextFormat]?.alpha)),overwrite,color_mode:colorMode,
          render_source:source,result_id:cacheId,allow_stale:renderSource==='rerender'||allowStale,frame_index:frameIndex},settings)
        savedPaths.push(saved.path);setResult({...saved,paths:[...savedPaths]})
        if(source==='rerender'){
          cacheId=saved.rendered_result_id
          if(!cacheId&&index+1<exportFormats.length)throw new Error('渲染结果缓存不可用，已停止后续格式导出')
          source='preview'
        }
      }
      onStatusChange?.(`图片已保存：${savedPaths.join('、')}`)
    }catch(e:any){setError(e.message);onStatusChange?.(`图片导出失败：${e.message}`)}
    finally{setBusy(false)}
  }
  return <div className="image-deliver workspace-page">
    <div className="image-deliver-title"><h1>{ui(videoFrame?"单帧导出":"图片导出")}</h1><span>{video.filename}{videoFrame&&` · ${ui("帧")} ${currentFrame+1}`}</span></div>
    <div className="image-deliver-grid">
      <section className="image-deliver-preview" aria-label={ui("图片预览")}>
        <div className="image-deliver-canvas">{ui(thumbnail?<ImageViewport src={thumbnail} paired={thumbnailPaired} view="dlss" zoom={previewZoom} onZoom={setPreviewZoom} pan={previewPan} onPan={setPreviewPan}/>:<ImageIcon size={40}/>)}</div>
        <div className="image-deliver-caption"><span>{ui(thumbnailPaired?(selected?.hdr?'已渲染结果（SDR 显示映射）':'已渲染结果'):result?.color_mode==='hdr10'?'原图（HDR 文件已保存）':result?'已导出':videoFrame?'原视频帧':'原图片')}</span><button className="image-fit" title={ui("恢复适配")} onClick={()=>{setPreviewZoom(1);setPreviewPan({x:.5,y:.5})}}>{ui(Math.round(previewZoom*100))}%</button><strong>{ui(`${dimensions.width} × ${dimensions.height}`)}</strong></div>
        <div className="image-deliver-chain">{ui(chain.length?chain.map((stage,index)=><span key={stage}>{ui(index>0&&<i>→</i>)}{ui(stageName(stage))}</span>):<span>{ui("原图输出")}</span>)}</div>
      </section>
      <section className="image-deliver-form" aria-label={ui("图片保存设置")}>
        <fieldset disabled={busy}>
          <label className="image-field">{ui('导出来源')}<select className="image-export-select" aria-label={ui('图片导出来源')} value={renderSource} onChange={event=>setRenderSource(event.target.value as 'preview'|'rerender')}>
            <option value="preview">{ui('已渲染结果（只转换格式）')}</option><option value="rerender">{ui('重新渲染后导出')}</option>
          </select></label>
          {renderSource==='preview'&&eligible.length>1&&<label className="image-field">{ui('选择渲染结果')}<select className="image-export-select" aria-label={ui('选择渲染结果')} value={resultId} onChange={event=>{selectedByUser.current=true;setResultId(event.target.value)}}>
            {eligible.map((item,index)=><option key={item.id} value={item.id}>{ui(index===0?'最新结果':'上一份结果')} · {item.width}×{item.height} · {item.hdr?'HDR':'SDR'}</option>)}
          </select></label>}
          {renderSource==='preview'&&selected&&!cacheCurrent&&<label className="image-check"><input type="checkbox" checked={allowStale} onChange={event=>setAllowStale(event.target.checked)}/>{ui('导出上次结果（忽略之后的参数改动）')}</label>}
          <div className="image-field">{ui("输出色彩")}<div className="image-format" role="group" aria-label={ui("图片输出色彩")}>
            <button type="button" aria-pressed={!hdr} onClick={()=>changeColor('sdr')}>SDR · sRGB</button>
            <button type="button" aria-pressed={hdr} disabled={!hdrAllowed||!hdrAvailable} title={ui(!hdrAllowed?(renderSource==='preview'?'这份渲染结果是 SDR':'请先在渲染链中开启 TrueHDR'):!hdrAvailable?'请安装支持 HDR 图片的编码组件':'保存 HDR 结果')} onClick={()=>changeColor('hdr10')}>HDR</button>
          </div></div>
          <div className="image-field">{ui("格式")}<div className="image-format" role="group" aria-label={ui("图片格式")}>{ui(['png','jpeg','webp','avif','heic','heif','uhdr','jxl'].map(value=><button type="button" disabled={!formatAvailable(value)} title={ui(formatAvailable(value)?value==='uhdr'?'兼容 SDR 查看器的 HDR JPEG':value.toUpperCase():value==='uhdr'&&!hdr?'UHDR JPEG 需要 HDR 结果':hdr?(hdrFormats[value]?.reason||'HDR 图片请选择 AVIF、HEIC、HEIF、UHDR 或 JXL'):'当前运行环境缺少此格式的编码组件')} aria-pressed={exportFormats.includes(value)} key={value} onClick={event=>chooseFormat(value,event.ctrlKey||event.metaKey)}>{ui(value==='uhdr'?'UHDR JPEG':value.toUpperCase())}</button>))}</div></div>
          {ui(exportFormats.some(value=>value!=='png')&&<label className="image-field">{ui("质量 ")}<output>{ui(quality===100&&exportFormats.every(value=>['png','jxl'].includes(value)||(!hdr&&['webp','heic','heif'].includes(value)))?'100 · 无损':quality)}</output><input type="range" aria-label={ui("图片质量")} min={1} max={100} step={1} value={quality} onChange={e=>setQuality(Number(e.target.value))}/></label>)}
          <label className="image-field">{ui("保存位置")}<div className="image-directory"><input aria-label={ui("图片保存目录")} value={directory} onChange={e=>setDirectory(e.target.value)}/><button className="icon-button" type="button" aria-label={ui("选择图片保存目录")} onClick={async()=>{try{const path=await openNativeDirDialog();if(path)setDirectory(path)}catch(e:any){setError(e.message)}}}><Folder size={18}/></button></div></label>
          <label className="image-field">{ui("文件名")}<div className="image-filename"><input aria-label={ui("图片文件名")} value={name} onChange={e=>setName(e.target.value)}/><span>{exportFormats.length>1?exportFormats.map(value=>value.toUpperCase()).join(' → '):`.${extension}`}</span></div></label>
          {ui(exportFormats.some(value=>!['jpeg','uhdr'].includes(value))&&<label className="image-check"><input type="checkbox" checked={alpha&&alphaAvailable} disabled={!alphaAvailable} onChange={e=>setAlpha(e.target.checked)}/>{ui("保留透明度")}</label>)}
          <label className="image-check"><input type="checkbox" checked={overwrite} onChange={e=>setOverwrite(e.target.checked)}/>{ui("覆盖同名文件")}</label>
        </fieldset>
        <div className="image-deliver-output"><span>{ui("输出")}</span><strong>{ui(dimensions.width)} × {ui(dimensions.height)}</strong><span>{ui(hdr?hdrEncoding:'sRGB · 8-bit')}</span></div>
        {renderSource==='preview'&&!selected&&<p className="image-deliver-note">{ui('尚未渲染图片，请先在工作台渲染一帧。')}</p>}
        {renderSource==='preview'&&selected&&<p className="image-deliver-note">{ui('使用完整尺寸缓存，只转换格式，不重新运行模型。')}</p>}
        {hdr&&<p className="image-deliver-note">{ui(format==='uhdr'?'UHDR 保存 SDR 基图和 HDR 增益图，不保留透明度；兼容的查看器可恢复 HDR。':format==='jxl'?'保存 16-bit BT.2020/PQ RGB；支持透明度。质量 100 为无损编码，色域转换与量化仍适用。':'保存真实 10-bit HDR，不经过 SDR 映射。此页缩略图不显示 HDR 文件，请使用支持 HDR 的查看器。')}</p>}
        {ui(hdr&&!hdrAllowed&&<p className="image-deliver-error" role="alert">{ui(renderSource==='preview'?'这份渲染结果是 SDR':'请先在渲染链中开启 TrueHDR')}</p>)}
        {ui(hasTrueHDR&&!hdrAvailable&&<p className="image-deliver-note">{ui("请安装支持 HDR 图片的编码组件")}</p>)}
        {ui(hdr&&!alphaAvailable&&video.has_alpha&&<p className="image-deliver-note">{ui("此编码器不支持 HDR 透明度，当前保存不包含透明度。")}</p>)}
        {ui(renderStages(settings).includes('vfg')&&!videoFrame&&<p className="image-deliver-note">{ui("静态图片不执行插帧。")}</p>)}
        {ui(format==='jpeg'&&video.has_alpha&&<p className="image-deliver-note">{ui("透明区域填充白色。")}</p>)}
        <button className="button primary image-save" disabled={!valid||busy||exportFormats.some(value=>!formatAvailable(value))||(hdr&&!hdrAllowed)||(renderSource==='preview'&&(!selected||!cached.source_unchanged||(!cacheCurrent&&!allowStale)))} onClick={run}>{ui(busy?`正在保存 ${saveProgress}/${exportFormats.length}…`:'保存图片')}<ArrowUpRight size={18}/></button>
        {error&&<ProcessingError message={error}/>}
        {ui(result&&!busy&&<div className="image-deliver-saved" role="status"><Check size={18}/><div><strong>{ui("已保存")}</strong>{(result.paths||[result.path]).map((path:string)=><p key={path}>{path}</p>)}</div></div>)}
      </section>
    </div>
  </div>
}
