// localized-render-sinks
import {ui} from '../i18n/locale'
import {ProcessingError} from './ProcessingError'
import {useEffect,useMemo,useRef,useState} from 'react'
import {Range} from 'react-range'
import type {DLSSSettings,VideoInfo} from '../types'
import {localToken} from '../services/api'
import {useExportSettings,useFrameSelection,setFrameSelection} from '../services/segmentSelection'
import {DEFAULT_EXPORT_SETTINGS,readTranscodeDraft} from '../services/exportSettings'
import {normalizeExportOptions} from '../services/exportOptions'
import type {PreviewRequest} from '../services/previewRender'
import './segmentTimeline.css'

type Segment={key:string;job_id:string;output_path:string;frame_range:{start_frame:number;end_frame:number};
  original_offset_seconds:number;duration_seconds:number;status:string;ready:boolean;progress:number;error?:string}
async function request(path:string,body:unknown,signal?:AbortSignal){
  const response=await fetch(`http://127.0.0.1:18923${path}`,{method:'POST',
    headers:{'Content-Type':'application/json',...(localToken()?{'X-DLSS-Token':localToken()}:{})},body:JSON.stringify(body),signal})
  const result=await response.json().catch(()=>({detail:`请求失败 (${response.status})`}));if(!response.ok)throw new Error(result.detail||'预渲染请求失败');return result
}
function time(frame:number,fps:number){const seconds=frame/fps;return `${Math.floor(seconds/60)}:${(seconds%60).toFixed(2).padStart(5,'0')}`}

export function SegmentTimeline({video,settings,currentFrame,onSeek,onSettingsChange,onStatus,onPause,onBeforeRender,previewRequest,onRenderBusy}:{
  video:VideoInfo|null;settings:DLSSSettings;currentFrame:number;onSeek:(frame:number)=>void;
  onSettingsChange:(patch:Partial<DLSSSettings>)=>void;onStatus?:(message:string)=>void;onPause:()=>void|Promise<void>;
  onBeforeRender?:()=>Promise<void>;
  previewRequest?:PreviewRequest|null;onRenderBusy?:(busy:boolean)=>void}){
  const selection=useFrameSelection(video?.path),exportSettings=useExportSettings()
  const fallback=useMemo(()=>readTranscodeDraft(DEFAULT_EXPORT_SETTINGS),[])
  const tx=normalizeExportOptions(exportSettings||fallback,settings,video)
  const signature=JSON.stringify({source:video?.path,settings,transcode_settings:tx})
  const body=useMemo(()=>JSON.parse(signature),[signature])
  const [cache,setCache]=useState<{signature:string;records:Segment[]}>({signature:'',records:[]})
  const [submitting,setSubmitting]=useState(false),[error,setError]=useState('')
  const [startDraft,setStartDraft]=useState<string|null>(null),[endDraft,setEndDraft]=useState<string|null>(null)
  const [quickRange,setQuickRange]=useState<PreviewRequest|null>(null)
  const lastQuickRequest=useRef(0)
  const requestRevision=useRef(0),signatureRef=useRef(signature);signatureRef.current=signature
  const selectionHead=useRef<0|1>(0)
  const total=Math.max(1,video?.frames||1),fps=video?.fps||30
  const range=selection&&selection.end_frame<=total?selection:{enabled:false,start_frame:0,end_frame:total}
  const records=cache.signature===signature?cache.records:[]
  const ready=records.filter(record=>record.ready)
  const quick=quickRange?.source===video?.path?quickRange:null
  const shownRange=quick??range
  const current=records.find(record=>record.frame_range.start_frame===shownRange.start_frame&&record.frame_range.end_frame===shownRange.end_frame)
  const pending=current&&['pending','running'].includes(current.status)
  const videoSource=!!video&&video.media_type!=='image'&&!/\.(png|jpe?g|bmp|webp|tiff?|avif|heic|heif|jxl|jxr|wdp|hdp)$/i.test(video.path)
  const watchCache=range.enabled||!!quick
  useEffect(()=>{
    if(!videoSource||!watchCache)return
    let stopped=false,inFlight=false
    const controller=new AbortController()
    const update=async()=>{
      if(inFlight||stopped)return;inFlight=true
      try{const data=await request('/api/preview/segments/list',body,controller.signal)
        if(!stopped)setCache({signature,records:data.segments||[]})
      }catch(e:any){if(!stopped)setError(e.message)}finally{inFlight=false}
    }
    void update();const timer=setInterval(update,1500)
    return()=>{stopped=true;controller.abort();clearInterval(timer)}
  },[signature,watchCache,videoSource,body])
  useEffect(()=>{onRenderBusy?.(submitting||!!pending)},[submitting,!!pending,onRenderBusy])
  useEffect(()=>{setError('')},[signature])
  const save=(start:number,end:number,enabled=range.enabled)=>{
    if(!video)return
    setQuickRange(null)
    setFrameSelection(video.path,{enabled,start_frame:Math.max(0,Math.min(total-1,Math.round(start))),
      end_frame:Math.max(Math.round(start)+1,Math.min(total,Math.round(end)))})
  }
  const enable=(enabled:boolean)=>{
    setQuickRange(null)
    if(enabled){save(currentFrame,Math.min(total,currentFrame+Math.max(2,Math.round(fps*2))),true)
      if(settings.export_backend!=='gpu_transfer')onSettingsChange({export_backend:'gpu_transfer'})}
    else save(range.start_frame,range.end_frame,false)
  }
  const render=async(selection={start_frame:range.start_frame,end_frame:range.end_frame})=>{
    const revision=++requestRevision.current,originalSignature=signature
    setSubmitting(true);setError('')
    try{
      if(onBeforeRender)await onBeforeRender();else await onPause()
      if(signatureRef.current!==originalSignature)throw new Error('参数已变化，请重新开始渲染')
      const record=await request('/api/preview/segments',{...body,frame_range:selection})
      if(signatureRef.current===originalSignature){setCache(old=>({signature:originalSignature,
        records:[...(old.signature===originalSignature?old.records:[]).filter(row=>row.key!==record.key),record]}));onStatus?.('选区已加入预渲染队列')}
    }catch(e:any){if(signatureRef.current===originalSignature)setError(e.message)}finally{if(revision===requestRevision.current)setSubmitting(false)}
  }
  useEffect(()=>{
    if(!previewRequest||previewRequest.id===lastQuickRequest.current||previewRequest.source!==video?.path)return
    lastQuickRequest.current=previewRequest.id;setQuickRange(previewRequest)
    void render({start_frame:previewRequest.start_frame,end_frame:previewRequest.end_frame})
  },[previewRequest])
  const play=async(record:Segment)=>{try{await onPause()}catch(error:any){setError(error.message);return}if(signatureRef.current!==signature)return;window.dispatchEvent(new CustomEvent('lunarc:compare',{detail:{
    original:video!.path,exported:record.output_path,original_offset_seconds:record.original_offset_seconds,
    duration_seconds:record.duration_seconds}}))}
  const updateValues=(values:number[])=>{
    setStartDraft(null);setEndDraft(null)
    const side=values[0]!==range.start_frame?0:values[1]!==range.end_frame?1:selectionHead.current
    selectionHead.current=side
    let [start,end]=values
    if(start===end){if(side===0)start=Math.max(0,end-1);else end=Math.min(total,start+1)}
    save(start,end)
    // The stored end is exclusive; preview the final frame inside the selection.
    onSeek(side===0?start:end-1)
  }
  const commit=(side:'start'|'end')=>{
    const draft=side==='start'?startDraft:endDraft
    if(draft===null)return
    const frame=Number(draft)
    if(side==='start')setStartDraft(null);else setEndDraft(null)
    if(!draft||!Number.isInteger(frame)||frame<1||frame>total){setError('帧号须为素材范围内的整数');return}
    if(side==='start')save(frame-1,Math.max(range.end_frame,frame))
    else save(Math.min(range.start_frame,frame-1),frame)
    onSeek(frame-1)
  }
  return <div className="segment-timeline">
    <div className="transport-scrub"><span>{ui(String(video?currentFrame+1:0).padStart(3,'0'))}</span>
      <Range min={0} max={Math.max(1,total-1)} step={1} values={[Math.max(0,Math.min(total-1,currentFrame))]}
        disabled={!videoSource} onChange={values=>onSeek(values[0])}
        renderTrack={({props,children})=><div {...props} className="segment-playhead-track">{ui(ready.map(record=><button key={record.key} type="button" className="segment-cached" title={ui(`已预渲染 ${record.frame_range.start_frame+1}–${record.frame_range.end_frame} 帧；点击回看`)}
          style={{left:`${record.frame_range.start_frame/total*100}%`,width:`${(record.frame_range.end_frame-record.frame_range.start_frame)/total*100}%`}}
          onMouseDown={e=>e.stopPropagation()} onTouchStart={e=>e.stopPropagation()} onClick={e=>{e.stopPropagation();play(record)}}/>))}{ui(children)}</div>}
        renderThumb={({props})=>{const {key,...rest}=props;return <div key={key} {...rest} aria-label={ui("时间轴帧位置")} className="segment-playhead"/>}}/>
      <span>{ui(String(video?.frames||0).padStart(3,'0'))}</span>
    </div>
    {ui(videoSource&&<>
      <div className="segment-actions"><label><input type="checkbox" checked={range.enabled} onChange={e=>enable(e.target.checked)}/>{ui("选区导出")}</label>
        {ui((range.enabled||quick)&&<><span>{ui(time(shownRange.start_frame,fps))}–{ui(time(shownRange.end_frame,fps))} · {ui(shownRange.end_frame-shownRange.start_frame)}{ui(" 帧")}</span>
          {ui(range.enabled&&!quick&&<button type="button" disabled={submitting||!!pending||settings.export_backend!=='gpu_transfer'} onClick={()=>void render()}>{ui(submitting?'正在提交…':pending?`预渲染 ${current!.progress.toFixed(0)}%`:'预渲染选区')}</button>)}
          {ui(quick&&(submitting||pending)&&<span role="status">{ui(submitting?'正在提交…':`预渲染 ${current!.progress.toFixed(0)}%`)}</span>)}
          {ui(current?.ready&&<button type="button" onClick={()=>play(current)}>{ui("回看结果")}</button>)}
          {ui(pending&&<button type="button" onClick={()=>window.dispatchEvent(new CustomEvent('lunarc:open-deliver'))}>{ui("查看任务")}</button>)}
        </>)}
      </div>
      {ui(range.enabled&&<div className="segment-selection">
        <label>{ui("起始帧")}<input type="number" min={1} max={total} value={startDraft??range.start_frame+1} onChange={e=>setStartDraft(e.target.value)} onBlur={()=>commit('start')} onKeyDown={e=>{if(e.key==='Enter'){e.currentTarget.blur()}}}/></label>
        <Range min={0} max={total} step={1} values={[range.start_frame,range.end_frame]} onChange={updateValues}
          onFinalChange={values=>onSeek(selectionHead.current===0?values[0]:values[1]-1)}
          renderTrack={({props,children})=><div {...props} className="segment-range-track"><div className="segment-range-fill" style={{left:`${range.start_frame/total*100}%`,width:`${(range.end_frame-range.start_frame)/total*100}%`}}/>{ui(children)}</div>}
          renderThumb={({props,index})=>{const {key,...rest}=props;return <div key={key} {...rest} onFocus={()=>{selectionHead.current=index?1:0;onSeek(index?range.end_frame-1:range.start_frame)}} aria-label={ui(index?'选区结束帧':'选区起始帧')} className="segment-range-handle"/>}}/>
        <label>{ui("结束帧")}<input type="number" min={1} max={total} value={endDraft??range.end_frame} onChange={e=>setEndDraft(e.target.value)} onBlur={()=>commit('end')} onKeyDown={e=>{if(e.key==='Enter'){e.currentTarget.blur()}}}/></label>
      </div>)}
      {(error||current?.status==='failed')&&<ProcessingError message={error||current?.error||'预渲染失败'}/>}
    </>)}
  </div>
}
