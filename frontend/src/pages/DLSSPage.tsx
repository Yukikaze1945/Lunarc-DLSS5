// localized-render-sinks
import {ui} from '../i18n/locale'
import {publishPreviewMetrics} from '../services/previewMetrics'
import {SegmentTimeline} from '../components/SegmentTimeline'
import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Maximize,
  Move,
  AlertTriangle,
} from 'lucide-react'
import { ParameterHelp } from '../components/ParameterHelp'
import { FLOW_QUALITY_LABELS } from '../services/parameterHelp'
import { QualityInspector } from '../components/QualityInspector'
import { LayerNRControls } from '../components/LayerNRControls'
import {useNativePreview} from '../services/useNativePreview'
import {ImageViewport} from '../components/ImageViewport'
import {ProcessingError} from '../components/ProcessingError'
import { RenderPipeline } from '../components/RenderPipeline'
import {PresetPicker} from '../components/PresetPicker'
import {CollapsibleSection} from '../components/CollapsibleSection'
import { renderStages, hasScaleStages, vsrOutputSize } from '../services/renderPipeline'
import { VideoInfo, DLSSSettings } from '../types'
import { saveSettings, wsUrl, fetchFlowProviders, fetchJobDiagnostics, previewMemory, imageThumbnail, sourceFrame,
  runtimeRequest, type DlssnrStatus,
  renderImage,imageResults,renderedImagePreview,compareImageResults,type ImageRenderResult,type PreviewMemoryState } from '../services/api'
import { subscribeQueue } from '../services/queue'
import { resolveWorkingDims } from '../services/workingScale'
import {PREVIEW_LENGTHS,previewFrameRange,type PreviewLength,type PreviewMode,type PreviewRequest} from '../services/previewRender'
import {
  HDR_LIMITATIONS, HDR_PAPER_WHITE_SCRGB, HDR_SETTING, HDR_PAPER_WHITE_SETTING,
  hdrPaperWhiteReason, hdrSettingsBlockers, isHdrRequested, paperWhiteLabel,
} from '../services/hdrExport'
import { activeLayerCount, chooseLayerScale, chooseReconstructionMode,
  chooseWorkingScale, chooseFusionMode, changeLayerNR, type LayerNRKey } from '../services/multiScaleSettings'

interface DLSSPageProps {
  video: VideoInfo | null
  settings: DLSSSettings
  onSettingsChange: (newSettings: DLSSSettings) => void
  currentFrame: number
  setCurrentFrame: (frame: number) => void
  onStatusChange?: (msg: string) => void
  isActive?: boolean
  onActivityChange?: (activity:{busy:boolean;error:boolean})=>void
}

export const DLSSPage: React.FC<DLSSPageProps> = ({
  video,
  settings,
  onSettingsChange,
  currentFrame,
  setCurrentFrame,
  onStatusChange,
  isActive = true,
  onActivityChange,
}) => {
  // View states
  const [view, setView] = useState<'dlss' | 'orig' | 'split' | 'dual' | 'flow'>('split')
  const [splitX, setSplitX] = useState<number>(0.5)
  const [zoom, setZoom] = useState<number>(1.0)
  const [pixelPreview, setPixelPreview] = useState(false)
  const [previewImageSize, setPreviewImageSize] = useState<{width:number;height:number}|null>(null)
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0.5, y: 0.5 })
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [loop, setLoop] = useState<boolean>(true)
  const [activeFrameUrl, setActiveFrameUrl] = useState<string>('')
  const [sourcePicture,setSourcePicture]=useState<{source:VideoInfo;url:string}|null>(null)
  const [sourceError,setSourceError]=useState('')
  const sourcePictureUrl=useRef('')
  const [playerError, setPlayerError] = useState<string | null>(null)
  const [playerBusy,setPlayerBusy]=useState('')
  const [previewStats, setPreviewStats] = useState<any>(null)
  useEffect(()=>{publishPreviewMetrics(previewStats?{...previewStats,playing:isPlaying}:null)},[previewStats,isPlaying])
  const [previewPending, setPreviewPending] = useState(false)
  // Permission belongs to this loaded object, never to a persisted auto-play flag.
  const [renderVideo,setRenderVideo]=useState<VideoInfo|null>(null)
  const [renderMode,setRenderMode]=useState<PreviewMode>('realtime')
  const [previewLength,setPreviewLength]=useState<PreviewLength>('f5')
  const [previewRequest,setPreviewRequest]=useState<PreviewRequest|null>(null)
  const [launchBusy,setLaunchBusy]=useState(false),[prerenderBusy,setPrerenderBusy]=useState(false)
  const [imageRealtime,setImageRealtime]=useState(false),imageRealtimeRef=useRef(false);imageRealtimeRef.current=imageRealtime
  const [imageBusy,setImageBusy]=useState(false),imageWorkBusy=useRef(false)
  const [imagePresentBusy,setImagePresentBusy]=useState(false)
  const [imageTicket,setImageTicket]=useState(0),imageCompleted=useRef(0)
  const [imageWorkEpoch,setImageWorkEpoch]=useState(0)
  const [imageResult,setImageResult]=useState<ImageRenderResult|null>(null)
  const [previousImageResult,setPreviousImageResult]=useState<ImageRenderResult|null>(null)
  const [imageResultStale,setImageResultStale]=useState(false)
  const imageResultRef=useRef(imageResult);imageResultRef.current=imageResult
  const [imageReference,setImageReference]=useState<'original'|'previous'>('original')
  const renderRevision=useRef(0),previewRequestId=useRef(0),sourceRef=useRef(video);sourceRef.current=video
  const [memoryPaused,setMemoryPaused]=useState(false)
  const [memoryBusy,setMemoryBusy]=useState(false)
  const memoryPausedRef=useRef(false),memoryRevision=useRef(0)
  const imageSource=video?.media_type==='image'
  const singleFrameMode=!imageSource&&renderMode==='single'
  const stillSource=imageSource||singleFrameMode
  const stillSourceRef=useRef(stillSource);stillSourceRef.current=stillSource
  const renderEnabled=!!video&&renderVideo===video&&!memoryPaused&&(renderMode!=='prerender'||imageSource)
  const renderEnabledRef=useRef(false);renderEnabledRef.current=renderEnabled
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false)
  const [isPanning, setIsPanning] = useState<boolean>(false)
  const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 })

  const canvasRef = useRef<HTMLDivElement>(null)
  const splitRef=useRef(splitX)
  const viewControls=useRef({view,zoom,pan});viewControls.current={view,zoom,pan}
  // TrueHDR stills need the same FP16 presenter as video, not an 8-bit JPEG pair.
  const nativeRequired=Boolean(video&&window.electronAPI?.getPreviewWindow&&
    (!imageSource||imageResult?.hdr||(imageReference==='previous'&&previousImageResult?.hdr)||renderStages(settings).includes('truehdr')))
  const nativePanel=useNativePreview(canvasRef,Boolean(nativeRequired&&isActive&&renderEnabled),view,zoom,pan)
  const showNative=nativeRequired&&renderEnabled&&view!=='flow'
  const sourceUrl=sourcePicture?.source===video?sourcePicture.url:''
  const activityBusy=Boolean(launchBusy||imageBusy||imagePresentBusy||prerenderBusy||memoryBusy||(isActive&&previewPending)||
    (video&&isActive&&!renderEnabled&&!sourceUrl&&!sourceError))
  const activityError=Boolean(playerError||nativePanel.error||(!renderEnabled&&sourceError))
  useEffect(()=>{onActivityChange?.({busy:activityBusy,error:activityError})},[activityBusy,activityError,onActivityChange])
  useEffect(()=>()=>{onActivityChange?.({busy:false,error:false})},[onActivityChange])
  const nativeHdrActive=nativePanel.hdrActive??previewStats?.native_display?.hdr_active
  const hdrDisplayExpected=renderEnabled?(imageSource&&imageResult?
    Boolean(imageResult.hdr||(imageReference==='previous'&&previousImageResult?.hdr)):
    Boolean(video?.is_hdr||renderStages(settings).includes('truehdr'))):Boolean(video?.is_hdr)
  const showSdrMapping=hdrDisplayExpected&&(!showNative||nativeHdrActive===false)
  const nativeReadyRef=useRef(false);nativeReadyRef.current=nativePanel.ready&&view!=='flow'
  const nativeRequiredRef=useRef(false);nativeRequiredRef.current=nativeRequired&&view!=='flow'
  const renderScope=view==='flow'?'flow':nativeRequired?'native':'image'
  useEffect(()=>{
    if(!video||!isActive||(renderEnabled&&sourceUrl))return
    let cancelled=false
    const controller=new AbortController()
    setSourceError('')
    // Decode/display the source independently. Never resume GPU preview here.
    const timer=setTimeout(()=>{
      const request=imageSource?imageThumbnail(video.path,0,controller.signal):sourceFrame(video.path,currentFrame,controller.signal)
      void request.then(blob=>{
        if(cancelled)return
        const url=URL.createObjectURL(blob),previous=sourcePictureUrl.current
        sourcePictureUrl.current=url;setSourcePicture({source:video,url})
        if(previous)URL.revokeObjectURL(previous)
      }).catch(error=>{if(!cancelled&&error.name!=='AbortError')setSourceError(error.message)})
    },80)
    return()=>{cancelled=true;clearTimeout(timer);controller.abort()}
  },[video,currentFrame,isActive,renderEnabled,imageSource])
  useEffect(()=>()=>{if(sourcePictureUrl.current)URL.revokeObjectURL(sourcePictureUrl.current)},[])
  const fullImagePreview=video?.media_type!=='image'&&(!nativeRequired||view==='flow')&&(zoom>1||pixelPreview)
  const pendingNativeCommand=useRef<Record<string,unknown>|null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  // P1-10: desired playback state, separate from what the socket confirmed.
  // Lets onopen (re)send play after drops instead of freezing on "playing".
  const wantPlayingRef = useRef<boolean>(false)
  const lastPlayRef = useRef<Record<string, unknown> | null>(null)
  const applyMemoryState=useCallback((state:PreviewMemoryState)=>{
    memoryRevision.current++
    memoryPausedRef.current=state.suspended
    setMemoryPaused(state.suspended);setMemoryBusy(state.releasing)
    if(state.suspended){
      wantPlayingRef.current=false;lastPlayRef.current=null
      setIsPlaying(false);setPreviewPending(false);setPreviewStats(null)
      setPlayerBusy(state.releasing?'正在结束当前帧并释放预览显存…':state.error||'点击“开启渲染”后生成预览')
    }else setPlayerBusy('')
  },[])
  useEffect(()=>{
    let active=true;const revision=memoryRevision.current
    void previewMemory().then(state=>{if(active&&memoryRevision.current===revision)applyMemoryState(state)}).catch(()=>{})
    return()=>{active=false}
  },[applyMemoryState])

  // Optical flow hardware probe state
  const [flowProviders, setFlowProviders] = useState<any[]>([])
  const [nvofProbe, setNvofProbe] = useState<any>(null)
  const [nrModelPresets,setNrModelPresets]=useState<number[]>([])
  useEffect(()=>{
    let active=true, revision=0
    const refresh=async()=>{
      const current=++revision
      try{
        const state=await runtimeRequest<DlssnrStatus>()
        if(active&&current===revision)setNrModelPresets(state.error?[]:state.active?.model_presets??[])
      }catch{if(active&&current===revision)setNrModelPresets([])}
    }
    void refresh()
    window.addEventListener('lunarc:runtime-changed',refresh)
    return()=>{active=false;window.removeEventListener('lunarc:runtime-changed',refresh)}
  },[])
  const [recentFusionJob, setRecentFusionJob] = useState<{name:string;status:string;report:any}|null>(null)
  const inspectedJobId = useRef('')
  const [selectedLayer, setSelectedLayer] = useState(0)
  const [expandedLayer, setExpandedLayer] = useState<number | null>(null)
  const activeLayer = Math.min(selectedLayer, activeLayerCount(settings.reconstruction_mode || 'single') - 1)
  const changeNR = (key: LayerNRKey, value: number) =>
    handleParamsBatchChange(changeLayerNR(settings, activeLayer, key, value))

  useEffect(() => {
    fetchFlowProviders()
      .then((data) => {
        setFlowProviders(data.providers || [])
        setNvofProbe(data.nvof_probe || null)
      })
      .catch((err) => console.warn('Failed to fetch flow providers:', err))
  }, [])

  useEffect(() => {
    if (!isActive) return
    let stopped = false
    let sequence = 0
    let pendingKey = ''
    const unsubscribe = subscribeQueue(async queue => {
      const latestTerminal = [...queue.jobs].reverse().find((job) =>
        ['completed', 'failed', 'cancelled', 'skipped'].includes(job.status))
      if (!latestTerminal) {
        sequence++; pendingKey = ''; inspectedJobId.current = ''
        setRecentFusionJob(null)
        return
      }
      const key = `${latestTerminal.id}:${latestTerminal.status}`
      if (key === inspectedJobId.current || key === pendingKey) return
      pendingKey = key
      const request = ++sequence
      try {
        const body = await fetchJobDiagnostics(latestTerminal.id, true)
        if (stopped || request !== sequence) return
        inspectedJobId.current = key
        setRecentFusionJob({name:latestTerminal.name,status:latestTerminal.status,
          report:body?.job?.residual_backend_diagnostics ?? null})
      } catch { /* Diagnostics are optional; the requested-settings diagram remains available. */ }
      finally { if (request === sequence) pendingKey = '' }
    })
    return () => { stopped = true; unsubscribe() }
  }, [isActive])

  const requestRevisionRef = useRef(0)
  const activeRef=useRef(isActive);activeRef.current=isActive
  const settingsRevisionRef = useRef(0)
  const settingsPendingRef = useRef(false)
  const settingsTimingRef = useRef<{inputAt:number; settings_apply_ms?:number; settings_http_ms?:number}>({inputAt:0})
  const lastSeekRef = useRef<Record<string, unknown> | null>(null)
  const displayedBlobRef = useRef('')
  const sendPlayer = useCallback((command: Record<string, unknown>) => {
    // Stills use explicit HTTP render requests. Socket/view refreshes may not infer.
    if(stillSource&&command.action!=='pause')return
    if(!renderEnabledRef.current&&command.action!=='pause')return
    if(memoryPausedRef.current&&command.action!=='pause')return
    if(!activeRef.current&&command.action!=='pause')return
    setPreviewPending(command.action !== 'pause')
    const bounds = canvasRef.current?.getBoundingClientRect()
    const controls=viewControls.current
    const native=nativeRequiredRef.current
    const edge = pixelPreview || Number(command.zoom ?? controls.zoom) > 1 ? 0 : Math.max(512,
      Math.min(1920, Math.ceil(Math.max(bounds?.width || 1280, bounds?.height || 720) * devicePixelRatio)))
    const imageSource=video?.media_type==='image'
    const payload = {...command, request_id: command.action === 'refresh' ? requestRevisionRef.current : ++requestRevisionRef.current,
      view:command.view??controls.view,
      transport: native ? 'native' : 'binary', ack: true, display_edge: imageSource?0:edge,
      ...(imageSource&&!native?{view:'dual',zoom:1,pan_x:.5,pan_y:.5}: {})}
    if (command.action === 'seek') lastSeekRef.current = payload
    if (command.action === 'play') lastPlayRef.current = payload
    if (command.action === 'pause') pendingNativeCommand.current=null
    if (command.action === 'set_view' && lastPlayRef.current) lastPlayRef.current = {...lastPlayRef.current, ...payload}
    if (native&&!nativeReadyRef.current&&command.action!=='pause') {
      if(command.action!=='refresh')pendingNativeCommand.current=command
      return
    }
    if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(JSON.stringify(payload))
  }, [fullImagePreview,nativePanel.ready,nativeRequired,view==='flow',video?.media_type,stillSource])

  useEffect(() => {
    let ws: WebSocket | null = null, stopped = false, header: any = null
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined
    let latestSequence = 0, lastPresented = 0, raf = 0
    const transportTiming = new Map<number, number>()
    const connect = () => {
      if (stopped) return
      ws = new WebSocket(wsUrl('player')); ws.binaryType = 'blob'; wsRef.current = ws
      const socket = ws
      const present = async (data: any, jpeg?: Blob) => {
        if (!renderEnabledRef.current || memoryPausedRef.current || data.request_id !== requestRevisionRef.current) return
        const sequence = ++latestSequence, received = performance.now()
        const url = jpeg ? URL.createObjectURL(jpeg) : data.image
        const image = new Image(); image.src = url
        try { await image.decode() } catch {
          if (jpeg) URL.revokeObjectURL(url)
          if (data.request_id === requestRevisionRef.current) setPlayerError('预览画面解码失败')
          if (data.request_id === requestRevisionRef.current) setPreviewPending(false)
          return
        }
        const decoded = performance.now()
        const stale = () => stopped || stillSourceRef.current || !renderEnabledRef.current || memoryPausedRef.current || sequence !== latestSequence || data.request_id !== requestRevisionRef.current
        if (stale()) {if (jpeg) URL.revokeObjectURL(url); return}
        raf = requestAnimationFrame(() => {
          if (stale()) {if (jpeg) URL.revokeObjectURL(url); return}
          const previous = displayedBlobRef.current
          displayedBlobRef.current = jpeg ? url : ''
          setPlayerError(null);setPlayerBusy(''); setCurrentFrame(data.frame); setActiveFrameUrl(url)
          // Presentation opportunity after React commits, not physical scanout.
          raf = requestAnimationFrame(() => {
            if (previous) URL.revokeObjectURL(previous)
            if (stale()) return
            const painted = performance.now()
            const latestSettings = !settingsPendingRef.current && data.preview?.settings_current !== false
              && (data.preview?.settings_revision ?? 0) >= settingsRevisionRef.current
            setPreviewPending(!latestSettings)
            const metrics = {decode_ms: decoded-received, paint_wait_ms: painted-decoded,
              ...(latestSettings && settingsTimingRef.current.inputAt ? {
                ...settingsTimingRef.current, parameter_to_present_ms:painted-settingsTimingRef.current.inputAt} : {}),
              receive_to_present_ms: painted-received,
              presentation_interval_ms: lastPresented ? painted-lastPresented : 0}
            lastPresented = painted
            setPreviewStats({...data.preview, client_presentation: metrics,
              display_sequence: data.sequence, ws_send_ms: transportTiming.get(data.sequence) ?? null,
              presentation_fps: metrics.presentation_interval_ms ? 1000/metrics.presentation_interval_ms : null})
            transportTiming.delete(data.sequence)
            if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({action: 'presented', sequence: data.sequence, ...metrics}))
          })
        })
      }
      ws.onopen = () => {
        if(!renderEnabledRef.current)return
        if(stillSourceRef.current)return
        if(memoryPausedRef.current)return
        if(!activeRef.current)return
        if(nativeRequiredRef.current&&!nativeReadyRef.current)return
        if (wantPlayingRef.current && lastPlayRef.current) {
          socket.send(JSON.stringify({action: 'play', ...lastPlayRef.current,
            transport: nativeReadyRef.current ? 'native' : 'binary', ack: true, request_id: requestRevisionRef.current})); setIsPlaying(true)
        } else if (lastSeekRef.current) socket.send(JSON.stringify({...lastSeekRef.current, request_id:requestRevisionRef.current}))
      }
      ws.onmessage = event => {
        if (event.data instanceof Blob) {
          const metadata = header; header = null
          if (metadata) void present(metadata, event.data)
          return
        }
        try {
          const data = JSON.parse(event.data)
          if(data.type==='preview_memory'){
            if(data.suspended){++latestSequence;header=null}
            applyMemoryState(data)
          } else if(stillSourceRef.current){
            return
          } else if (data.type === 'frame') {
            if(!renderEnabledRef.current)return
            if(memoryPausedRef.current)return
            if(data.encoding==='native_fp16'){
              if(data.request_id!==requestRevisionRef.current)return
              ++latestSequence;const now=performance.now()
              setCurrentFrame(data.frame);setPlayerError(null);setPlayerBusy('')
              const latest=!settingsPendingRef.current&&data.preview?.settings_current!==false
              setPreviewPending(!latest)
              setPreviewStats({...data.preview,display_sequence:data.sequence,
                presentation_fps:lastPresented?1000/(now-lastPresented):null})
              lastPresented=now
              if(socket.readyState===WebSocket.OPEN)socket.send(JSON.stringify({action:'presented',sequence:data.sequence}))
            }else if (data.encoding === 'jpeg_binary') header = data
            else void present(data)
          } else if (data.request_id === requestRevisionRef.current && data.type === 'frame_timing') {
            transportTiming.set(data.sequence, data.ws_send_ms)
            if (transportTiming.size > 8) transportTiming.delete(transportTiming.keys().next().value!)
            setPreviewStats((previous: any) => previous?.display_sequence === data.sequence
              ? {...previous, ws_send_ms: data.ws_send_ms} : previous)
          } else if(data.request_id===requestRevisionRef.current&&data.type==='busy'){
            wantPlayingRef.current=false;setIsPlaying(false);setPreviewPending(false)
            setPlayerError(null);setPlayerBusy(data.message);setPreviewStats(null)
            onStatusChange?.(data.message)
          } else if (data.request_id === requestRevisionRef.current && data.type === 'error') {
            setPreviewPending(false)
            wantPlayingRef.current = false; setPlayerError(data.message); setIsPlaying(false); onStatusChange?.(`预览失败：${data.message}`)
          } else if (data.request_id === requestRevisionRef.current && data.type === 'ended') {
            wantPlayingRef.current = false; setIsPlaying(false); onStatusChange?.('播放结束')
          }
        } catch {setPlayerError('预览连接返回了无效数据')}
      }
      ws.onclose = () => {
        if (wsRef.current === socket) wsRef.current = null
        if (!stopped) {setIsPlaying(false); reconnectTimer = setTimeout(connect, 1500)}
      }
      ws.onerror = () => socket.close()
    }
    connect()
    return () => {
      stopped = true; ++latestSequence; clearTimeout(reconnectTimer); cancelAnimationFrame(raf); ws?.close()
      if (displayedBlobRef.current) URL.revokeObjectURL(displayedBlobRef.current)
      displayedBlobRef.current = ''
    }
  }, [setCurrentFrame, onStatusChange])

  const currentFrameRef = useRef(currentFrame)
  // NOTE(oxlint baseline react(refs)): intentional latest-value ref so async
  // socket callbacks always see the current frame without re-subscribing.
  currentFrameRef.current = currentFrame

  const requestFrame = useCallback(
    (
      targetFrame?: number,
      targetView?: string,
      targetSplit?: number,
      targetZoom?: number,
      targetPan?: { x: number; y: number }
    ) => {
      if (!video) return
      const f = targetFrame !== undefined ? targetFrame : currentFrameRef.current
      const controls=viewControls.current
      const v = targetView !== undefined ? targetView : controls.view
      const sx = targetSplit !== undefined ? targetSplit : splitRef.current
      const zm = targetZoom !== undefined ? targetZoom : controls.zoom
      const p = targetPan !== undefined ? targetPan : controls.pan

      sendPlayer({action: 'seek', frame: f, view: v, split_x: sx, zoom: zm, pan_x: p.x, pan_y: p.y})

    },
    [video, sendPlayer]
  )

  const stopRendering=async()=>{
    ++renderRevision.current
    renderEnabledRef.current=false;setRenderVideo(null)
    lastSeekRef.current=null;pendingNativeCommand.current=null
    clearTimeout(seekTimerRef.current);pendingSeekFrameRef.current=null
    memoryRevision.current++
    let serverBusy=false
    setMemoryBusy(true);setPlayerError(null)
    memoryPausedRef.current=true;setMemoryPaused(true)
    wantPlayingRef.current=false;lastPlayRef.current=null
    setIsPlaying(false);setPreviewPending(false)
    sendPlayer({action:'pause'})
    setPlayerBusy('正在结束当前帧并释放预览显存…')
    try{
      const state=await previewMemory('release')
      serverBusy=state.releasing
      applyMemoryState(state)
      onStatusChange?.('渲染已停止，预览显存已释放')
    }catch(error:any){
      setPlayerError(error.message);onStatusChange?.(error.message)
      try{const state=await previewMemory();serverBusy=state.releasing;applyMemoryState(state)}catch{}
      throw error
    }finally{setMemoryBusy(serverBusy)}
  }

  const startRendering=async()=>{
    if(!video||launchBusy||memoryBusy||prerenderBusy||imageBusy)return
    const source=video,revision=++renderRevision.current
    setLaunchBusy(true);setPlayerError(null)
    try{
      if(source.media_type==='image'||renderMode==='single'){
        if(!renderEnabledRef.current){
          const state=await previewMemory('resume')
          if(revision!==renderRevision.current||sourceRef.current!==source)return
          applyMemoryState(state);setRenderVideo(source)
        }
        setImageTicket(value=>value+1)
      }else if(renderMode==='prerender'){
        // Explicitly choose the same native export backend used by cached ranges.
        if(settings.export_backend!=='gpu_transfer'){
          const accepted=await saveSettings({export_backend:'gpu_transfer'})
          if(revision!==renderRevision.current||sourceRef.current!==source)return
          onSettingsChange(accepted)
        }
        if(revision!==renderRevision.current||sourceRef.current!==source)return
        setPreviewRequest({id:++previewRequestId.current,source:source.path,
          ...previewFrameRange(source,currentFrameRef.current,previewLength)})
      }else{
        const state=await previewMemory('resume')
        if(revision!==renderRevision.current||sourceRef.current!==source)return
        applyMemoryState(state);setRenderVideo(source);setPreviewPending(true)
        setPlayerBusy('正在准备预览…')
      }
    }catch(error:any){setPlayerError(error.message);onStatusChange?.(error.message)}
    finally{setLaunchBusy(false)}
  }

  const choosePreviewMode=(mode:PreviewMode)=>{
    if(mode==='single'&&view==='flow')setView('split')
    setRenderMode(mode)
    if(renderVideo){void stopRendering().catch(()=>{})}
  }

  useEffect(()=>{
    if(!stillSource||!video||!isActive||!renderEnabled||imageTicket<=imageCompleted.current||imageWorkBusy.current)return
    if(settingsPendingRef.current&&!imageRealtime)return
    if(nativeRequired&&!nativePanel.ready)return
    const source=video,ticket=imageTicket
    imageWorkBusy.current=true;setImageBusy(true);setPreviewPending(true);setPlayerError(null)
    void renderImage(source.path,settings.image_memory_mode||'memory',imageSource?undefined:currentFrameRef.current).then(data=>{
      if(sourceRef.current!==source||!renderEnabledRef.current||data.discarded)return
      setImageResult(data.result);setPreviousImageResult(data.previous||null)
      setImageResultStale(!data.current);setPreviewStats(data.result.diagnostics)
      onStatusChange?.(data.current?(imageSource?'图片渲染完成':'单帧渲染完成'):'已渲染，参数有新改动')
      window.dispatchEvent(new CustomEvent('lunarc:image-rendered',{detail:{path:source.path,result_id:data.result.id}}))
    }).catch(error=>{
      if(sourceRef.current===source&&renderEnabledRef.current){setPlayerError(error.message);onStatusChange?.(error.message)}
    }).finally(()=>{
      if(sourceRef.current===source)imageCompleted.current=Math.max(imageCompleted.current,ticket)
      imageWorkBusy.current=false;setImageBusy(false);setPreviewPending(false)
      setImageWorkEpoch(value=>value+1)
    })
  },[video,imageSource,stillSource,isActive,renderEnabled,imageTicket,imageWorkEpoch,imageRealtime,nativeRequired,nativePanel.ready,settings.image_memory_mode])

  useEffect(()=>{
    if(!stillSource||!imageResult||!renderEnabled||!isActive||imageBusy||imageWorkBusy.current)return
    if(nativeRequired&&!nativePanel.ready)return
    let cancelled=false
    const source=video,controller=new AbortController()
    setImagePresentBusy(true)
    const reference=imageReference==='previous'?previousImageResult?.id:undefined
    const present=async()=>{
      if(nativeRequired){
        const report=await compareImageResults(imageResult.id,reference)
        if(!cancelled)setPreviewStats((value:any)=>({...value,native_display:report}))
      }else{
        const blob=await renderedImagePreview(imageResult.id,reference,controller.signal)
        if(cancelled||sourceRef.current!==source)return
        const url=URL.createObjectURL(blob),previous=displayedBlobRef.current
        displayedBlobRef.current=url;setActiveFrameUrl(url)
        requestAnimationFrame(()=>{if(previous)URL.revokeObjectURL(previous)})
      }
    }
    void present().catch(error=>{if(!cancelled&&error.name!=='AbortError')setPlayerError(error.message)})
      .finally(()=>{if(!cancelled)setImagePresentBusy(false)})
    return()=>{cancelled=true;controller.abort();setImagePresentBusy(false)}
  },[stillSource,imageResult,imageReference,previousImageResult,renderEnabled,isActive,imageBusy,nativeRequired,nativePanel.ready,video])

  useEffect(()=>{
    const restored=()=>{
      ++renderRevision.current;renderEnabledRef.current=false;setRenderVideo(null)
      lastSeekRef.current=null;lastPlayRef.current=null;wantPlayingRef.current=false;setIsPlaying(false)
      setPlayerError(null);setPreviewPending(false)
      setImageRealtime(false);setImageResult(null);setPreviousImageResult(null)
      setImageTicket(0);imageCompleted.current=0
    }
    window.addEventListener('lunarc:backend-restored',restored)
    return()=>window.removeEventListener('lunarc:backend-restored',restored)
  },[])

  useEffect(() => {
    const pending = () => {
      settingsPendingRef.current = true
      settingsTimingRef.current = {inputAt:performance.now()}
      setPlayerError(null)
      if(stillSource)setImageResultStale(Boolean(imageResultRef.current))
      if(renderEnabledRef.current&&!memoryPausedRef.current&&(!stillSource||imageRealtimeRef.current))setPreviewPending(true)
    }
    const applied = (event: Event) => {
      const detail = (event as CustomEvent).detail
      settingsRevisionRef.current = detail.revision ?? settingsRevisionRef.current
      settingsTimingRef.current = detail
      if(stillSource){
        if(imageSource&&imageRealtimeRef.current&&renderEnabledRef.current)setImageTicket(value=>value+1)
      }else if (video) sendPlayer({action:'refresh'})
    }
    const settled = () => {
      settingsPendingRef.current = false
      if(stillSource&&video){
        setImageWorkEpoch(value=>value+1)
        const source=video,result=imageResultRef.current
        if(result)void imageResults(source.path,singleFrameMode?result.frame_index:undefined).then(state=>{
          if(sourceRef.current===source&&imageResultRef.current?.id===result.id)setImageResultStale(!state.current)
        }).catch(()=>{})
      }else if (video) sendPlayer({action:'refresh'})
    }
    window.addEventListener('lunarc:settings-pending', pending)
    window.addEventListener('lunarc:settings-applied', applied)
    window.addEventListener('lunarc:settings-settled', settled)
    return () => {
      window.removeEventListener('lunarc:settings-pending', pending)
      window.removeEventListener('lunarc:settings-applied', applied)
      window.removeEventListener('lunarc:settings-settled', settled)
    }
  }, [video, sendPlayer,stillSource,singleFrameMode,imageSource])

  useEffect(() => {
    ++renderRevision.current;renderEnabledRef.current=false;setRenderVideo(null)
    lastSeekRef.current=null;setPreviewRequest(null);setPrerenderBusy(false)
    sendPlayer({action:'pause'})
    if(renderVideo)void stopRendering().catch(()=>{})
    clearTimeout(seekTimerRef.current)
    pendingSeekFrameRef.current = null
    wantPlayingRef.current = false
    lastPlayRef.current = null
    setIsPlaying(false)
    setActiveFrameUrl('')
    setPlayerError(null);setPlayerBusy('')
    pendingNativeCommand.current=null
    setPreviewStats(null)
    setImageRealtime(false);setImageResult(null);setPreviousImageResult(null);setImageResultStale(false)
    setImageReference('original');setImageTicket(0);imageCompleted.current=0
    setPreviewPending(false)
    setZoom(1);setPan({x:.5,y:.5});setPixelPreview(false)
    if(video?.media_type==='image')setView(current=>current==='flow'?'split':current)
  }, [video])

  // The native window already holds both GPU pictures. View, wipe and pan
  // changes redraw there, without sending a new frame through the backend.
  useEffect(() => {
    if (!video||!renderEnabled||stillSource) return
    if(nativeRequired&&viewControls.current.view!=='flow'&&!nativePanel.ready)return
    if(pendingNativeCommand.current) {
      const command=pendingNativeCommand.current;pendingNativeCommand.current=null
      sendPlayer(command);return
    }
    if (isPlaying) {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        sendPlayer({
            action: 'set_view',
            view:viewControls.current.view,
            split_x:splitRef.current,
            zoom:viewControls.current.zoom,
            pan_x:viewControls.current.pan.x,
            pan_y:viewControls.current.pan.y,
          })
      }
    } else {
      requestFrame(currentFrameRef.current)
    }
  }, [video, isPlaying, requestFrame, renderScope, sendPlayer,renderEnabled,stillSource])

  useEffect(()=>{
    if(!video||video.media_type==='image'||(nativeRequired&&view!=='flow'))return
    if(isPlaying)sendPlayer({action:'set_view',view,split_x:splitX,zoom,pan_x:pan.x,pan_y:pan.y})
    else requestFrame(currentFrameRef.current,view,splitX,zoom,pan)
  },[view,splitX,zoom,pan,nativeRequired])

  // Play / Pause handling (P1-08 sends loop; P1-10 never claims playing
  // without an open socket — disconnected play becomes buffered intent)
  const togglePlay = () => {
    if (!video || stillSource||!renderEnabledRef.current) return
    const nextPlaying = !isPlaying
    wantPlayingRef.current = nextPlaying

    if (nextPlaying) {
      // Record intent first so a later (re)connect can replay it (P1-10).
      lastPlayRef.current = {
        frame: currentFrameRef.current,
        fps: video.fps || 30,
        view,
        split_x: splitX,
        zoom,
        pan_x: pan.x,
        pan_y: pan.y,
        loop,
      }
    } else {
      lastPlayRef.current = null
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      if (nextPlaying) {
        sendPlayer({ action: 'play', ...lastPlayRef.current })
        setIsPlaying(true)
        onStatusChange?.('正在播放')
      } else {
        sendPlayer({ action: 'pause' })
        setIsPlaying(false)
        onStatusChange?.('播放已暂停')
      }
    } else {
      setIsPlaying(false)
      onStatusChange?.(nextPlaying ? '等待连接，后台就绪后自动开始…' : '播放已暂停')
    }
  }

  const lastSeekTimeRef = useRef<number>(0)
  const pendingSeekFrameRef = useRef<number | null>(null)
  const seekTimerRef = useRef<any>(null)

  // NOTE(oxlint baseline react(set-state-in-effect)): intentional sync —
  // leaving the tab must immediately stop playback and its socket stream.
  useEffect(() => {
    if (isActive === false && isPlaying) {
      setIsPlaying(false)
      wantPlayingRef.current = false
      lastPlayRef.current = null
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          sendPlayer({ action: 'pause' })
        } catch {}
      }
      onStatusChange?.('播放已暂停')
    }
  }, [isActive, isPlaying, onStatusChange])

  useEffect(() => {
    return () => {
      clearTimeout(seekTimerRef.current)
      if (isPlaying) {
        onStatusChange?.('播放已暂停')
      }
    }
  }, [isPlaying, onStatusChange])

  const handleSeek = (frame: number) => {
    const total = video?.frames || 1
    const clamped = Math.max(0, Math.min(total - 1, frame))
    setCurrentFrame(clamped)
    if(singleFrameMode){
      renderEnabledRef.current=false;setRenderVideo(null);setPreviewPending(false)
      return
    }
    if (isPlaying) {
      setIsPlaying(false)
      wantPlayingRef.current = false
      lastPlayRef.current = null
    }

    const now = Date.now()
    if (now - lastSeekTimeRef.current > 50) {
      lastSeekTimeRef.current = now
      clearTimeout(seekTimerRef.current)
      pendingSeekFrameRef.current = null
      requestFrame(clamped)
    } else {
      pendingSeekFrameRef.current = clamped
      clearTimeout(seekTimerRef.current)
      seekTimerRef.current = setTimeout(() => {
        if (pendingSeekFrameRef.current !== null) {
          requestFrame(pendingSeekFrameRef.current)
          pendingSeekFrameRef.current = null
        }
      }, 50)
    }
  }

  // Parameter update helper
  const handleParamChange = async (key: keyof DLSSSettings, val: any) => {
    const patch: Partial<DLSSSettings> = { [key]: val }
    const updated = { ...settings, ...patch }
    onSettingsChange(updated)
    try {
      await saveSettings(patch)
      onStatusChange?.('参数已保存')

    } catch (e) {
      onSettingsChange(settings)
      onStatusChange?.(`设置未保存：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleParamsBatchChange = async (newParams: Partial<DLSSSettings>) => {
    const updated = { ...settings, ...newParams }
    onSettingsChange(updated)
    try {
      await saveSettings(newParams)
      onStatusChange?.('参数已更新')

    } catch (e) {
      onSettingsChange(settings)
      onStatusChange?.(`设置未保存：${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const handleWorkingScaleChange = (scale: number) => {
    handleParamsBatchChange(chooseWorkingScale(settings, scale))
  }

  // The experimental HDR pair is written through its own helper because a refusal has
  // to be visible: the backend is the authority on both values, and a panel that had
  // already moved its own toggle would be showing a setting that was never stored.
  const saveHdrPatch = async (patch: Partial<DLSSSettings>) => {
    onSettingsChange({ ...settings, ...patch })
    try {
      await saveSettings(patch)
    } catch (e: any) {
      onSettingsChange(settings)
      onStatusChange?.(`实验性 HDR 设置被后端拒绝：${e?.message ?? String(e)}`)
    }
  }

  // Paper white is typed as text so a half-written value ("8.", "0.5", "abc") can be
  // shown and refused as it is being written, instead of being snapped to a number the
  // operator never typed. Only a value that passes the backend's own predicate is sent.
  const [paperWhiteDraft, setPaperWhiteDraft] = useState<string | null>(null)
  const hdrOn = isHdrRequested(settings)
  const savedPaperWhite = settings.hdr_paper_white_scrgb ?? HDR_PAPER_WHITE_SCRGB
  const paperWhiteInput = paperWhiteDraft ?? String(savedPaperWhite)
  const paperWhiteRefusal = hdrPaperWhiteReason(
    paperWhiteDraft === null ? savedPaperWhite : paperWhiteDraft)
  const handlePaperWhiteChange = (raw: string) => {
    setPaperWhiteDraft(raw)
    if (!hdrPaperWhiteReason(raw)) {
      saveHdrPatch({ [HDR_PAPER_WHITE_SETTING]: Number(raw) })
    }
  }
  const hdrRefusals = hdrOn ? hdrSettingsBlockers(settings, video) : []
  const handleLayerScaleChange = (index: number, scale: number) => {
    try {
      handleParamChange('layer_scales', chooseLayerScale(settings, index, scale))
    } catch (error) {
      onStatusChange?.(String(error))
    }
  }

  const handleLayerWeightChange = (index: number, weight: number) => {
    const count = activeLayerCount(settings.reconstruction_mode || 'single')
    const weights = Array.from({ length: count }, (_, at) =>
      at === index ? weight : (settings.layer_weights?.[at] || 1))
    handleParamChange('layer_weights', weights)
  }

  const handleEngineChange = (eng: string) => {
    if (isHdrRequested(settings) && Number(settings.guidance_mode ?? 0) === 1 && !['nvof','fidelityfx','raft'].includes(eng)) {
      // Refused out loud rather than saved and met at the first frame: a CPU flow
      // engine would have to download the plane to compute motion, which is exactly
      // what this chain does not do.
      onStatusChange?.('HDR 光流引导需要可用的 NVIDIA / AMD / RAFT CUDA 设备路径')
      return
    }
    let defaultQ = '1F'
    if (eng === 'dis') defaultQ = 'Medium'
    else if (eng === 'raft') defaultQ = 'Balanced'
    else if (eng === 'fidelityfx') defaultQ = 'Quality'
    else if (eng === 'nvof') defaultQ = '1F'
    handleParamsBatchChange({ flow_engine: eng, flow_quality: defaultQ, ...(eng==='fidelityfx'?{export_backend:'gpu_transfer' as const}:{}) })
  }

  const nvofProvider = flowProviders.find((p) => p.engine === 'nvof')
  const nvofGpuName = nvofProvider?.device || nvofProbe?.device || 'NVIDIA GPU'
  const displayGpu = nvofGpuName.replace(/^NVIDIA\s+(GeForce\s+)?/i, '')
  const isNvofAvailable = nvofProvider?.available ?? nvofProbe?.available ?? false
  const nvofGrids = nvofProvider?.grids && nvofProvider.grids.length > 0
    ? nvofProvider.grids
    : [
        { id: 1, label: '1 × 1 网格' },
        { id: 2, label: '2 × 2 网格' },
        { id: 4, label: '4 × 4 网格' },
      ]
  const nvofPresets = nvofProvider?.presets && nvofProvider.presets.length > 0
    ? nvofProvider.presets
    : [
        { id: 'fast', label: 'FAST - 极速 (推荐)' },
        { id: 'medium', label: 'MEDIUM - 平衡' },
        { id: 'slow', label: 'SLOW - 质量优先' },
      ]

  const nvofGrid = settings.flow_quality.startsWith('2') ? '2' : settings.flow_quality.startsWith('4') ? '4' : '1'
  const nvofPreset = settings.flow_quality.endsWith('S') ? 'SLOW' : settings.flow_quality.endsWith('M') ? 'MEDIUM' : 'FAST'

  const handleNvofGridChange = (newGrid: string) => {
    const pChar = nvofPreset === 'SLOW' ? 'S' : nvofPreset === 'MEDIUM' ? 'M' : 'F'
    handleParamChange('flow_quality', `${newGrid}${pChar}`)
  }

  const handleNvofPresetChange = (newPreset: string) => {
    const pChar = newPreset === 'SLOW' ? 'S' : newPreset === 'MEDIUM' ? 'M' : 'F'
    handleParamChange('flow_quality', `${nvofGrid}${pChar}`)
  }

  // Split Wipe mouse drag
  const handleMouseDown = (e: React.PointerEvent<HTMLElement>) => {
    if (nativeRequiredRef.current || video?.media_type==='image') return
    const rect=canvasRef.current?.getBoundingClientRect()
    if (view === 'split' && e.button === 0 && rect && Math.abs(e.clientX-rect.left-splitRef.current*rect.width)<18) {
      e.currentTarget.setPointerCapture(e.pointerId)
      setIsDraggingSplit(true)
      handleSplitMove(e)
    } else if (e.button === 2 || (e.button === 0 && zoom > 1.0)) {
      e.currentTarget.setPointerCapture(e.pointerId)
      setIsPanning(true)
      setPanStart({ x: e.clientX, y: e.clientY })
    }
  }

  const handleMouseMove = (e: React.PointerEvent<HTMLElement>) => {
    if (isDraggingSplit && canvasRef.current) {
      handleSplitMove(e)
    } else if (isPanning && zoom > 1.0) {
      const dx = (e.clientX - panStart.x) / (canvasRef.current?.clientWidth || 1)
      const dy = (e.clientY - panStart.y) / (canvasRef.current?.clientHeight || 1)
      setPan((prev) => ({
        x: Math.max(0, Math.min(1, prev.x - dx)),
        y: Math.max(0, Math.min(1, prev.y - dy)),
      }))
      setPanStart({ x: e.clientX, y: e.clientY })
    }
  }

  const handleSplitMove = (e: React.PointerEvent<HTMLElement>) => {
    if (!canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const frac = (e.clientX - rect.left) / rect.width
    splitRef.current=Math.max(0.01, Math.min(0.99, frac))
    setSplitX(splitRef.current)
  }

  const handleMouseUp = (e: React.PointerEvent<HTMLElement>) => {
    if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId)
    if(isDraggingSplit)setSplitX(splitRef.current)
    setIsDraggingSplit(false)
    setIsPanning(false)
  }

  return (
    <div className="dlss-workbench select-none">
      <main className="dlss-view-column">
        <div className="dlss-view-toolbar">
          <div className="dlss-view-modes" role="group" aria-label={ui("预览视图")}>{ui(([
            {id:'split',label:'对比'},{id:'dual',label:'双画面'},{id:'dlss',label:'增强'},{id:'orig',label:'原图'},{id:'flow',label:'光流'},
          ] as const).filter(item=>!stillSource||item.id!=='flow').map(item=><button key={item.id} type="button" aria-pressed={view===item.id} onClick={()=>setView(item.id)}>{ui(item.label)}</button>))}</div>
          {stillSource&&imageResult&&<label className="image-reference-picker">{ui('对比参考')}<select aria-label={ui('图片对比参考')} value={imageReference} disabled={imageBusy} onChange={event=>setImageReference(event.target.value as 'original'|'previous')}>
            <option value="original">{ui('原图')}</option><option value="previous" disabled={!previousImageResult}>{ui('上一份结果')}</option>
          </select></label>}
          <div className="dlss-zoom-tools">
            <button type="button" className="button quiet preview-release-button" disabled={!video||memoryBusy||launchBusy} onClick={()=>void stopRendering().catch(()=>{})} title={ui('暂停预览并释放模型、纹理和缓存；保留参数和模型文件')}>{ui(memoryBusy?'处理中…':'释放预览显存')}</button>
            {ui(nativeRequired&&renderEnabled&&<span className="native-preview-status">{ui(hdrDisplayExpected&&nativeHdrActive===true?'HDR · FP16 scRGB':hdrDisplayExpected&&nativeHdrActive===false?'SDR 映射 · FP16':'GPU 预览')}</span>)}
            <select aria-label={ui("预览缩放")} value={pixelPreview?'native':zoom} onChange={e=>{if(e.target.value==='native')return;setPixelPreview(false);setZoom(Number(e.target.value));setPan({x:.5,y:.5})}}>
              {ui(pixelPreview&&<option value="native">{ui("原像素")}</option>)}{ui(![.5,1,2,3].includes(zoom)&&<option value={zoom}>{ui(Math.round(zoom*100))}%</option>)}<option value={1}>{ui("适配")}</option><option value={.5}>{ui("缩小")}</option><option value={2}>{ui("放大 2×")}</option><option value={3}>{ui("放大 3×")}</option>
            </select>
            <button type="button" className="icon-button" aria-label={ui("按输出原始像素查看")} title={ui("按输出原始像素查看 VSR 细节")} onClick={()=>{
              const bounds=canvasRef.current?.getBoundingClientRect()
              if (!video || !bounds) return
              const size=view!=='orig'&&hasScaleStages(settings)?vsrOutputSize(settings,video):video
              setPixelPreview(true)
              setZoom(Math.max(1,size.width*(view==='dual'?2:1)/(bounds.width*devicePixelRatio),size.height/(bounds.height*devicePixelRatio)))
              setPan({x:.5,y:.5})
            }}>1:1</button>
            <button className="icon-button" title={ui("全屏预览")} aria-label={ui("全屏预览")} onClick={()=>{const action=document.fullscreenElement?document.exitFullscreen():canvasRef.current?.requestFullscreen();action?.catch(e=>onStatusChange?.(`无法全屏：${e.message}`))}}><Maximize size={17}/></button>
          </div>
        </div>
        <div className="preview-render-controls">
          {imageSource&&<><label>{ui('图片渲染模式')}<select aria-label={ui('图片渲染模式')} value={settings.image_memory_mode||'memory'} disabled={imageBusy||launchBusy||memoryBusy}
            onChange={event=>handleParamChange('image_memory_mode',event.target.value as 'memory'|'fast')}>
            <option value="memory">{ui('省显存（逐项处理）')}</option><option value="fast">{ui('快速（全链路常驻）')}</option>
          </select></label><label className="image-realtime-toggle"><input type="checkbox" checked={imageRealtime} disabled={launchBusy||memoryBusy} onChange={event=>{
            const enabled=event.target.checked;setImageRealtime(enabled)
            if(enabled){if(imageBusy)setImageTicket(value=>value+1);else void startRendering()}
            else imageCompleted.current=imageTicket
          }}/>{ui('实时渲染')}</label></>}
          {video?.media_type!=='image'&&<><label>{ui('渲染方式')}<select aria-label={ui('渲染方式')} value={renderMode} disabled={memoryBusy||launchBusy||prerenderBusy} onChange={event=>choosePreviewMode(event.target.value as PreviewMode)}>
            <option value="single">{ui('单帧渲染')}</option><option value="realtime">{ui('实时渲染')}</option><option value="prerender">{ui('预渲染')}</option>
          </select></label>{renderMode==='prerender'&&<label>{ui('预渲染长度')}<select aria-label={ui('预渲染长度')} value={previewLength} disabled={launchBusy||prerenderBusy} onChange={event=>setPreviewLength(event.target.value as PreviewLength)}>
            {PREVIEW_LENGTHS.map(option=><option key={option.id} value={option.id}>{ui(option.label)}</option>)}
          </select></label>}</>}
          <button className="button primary" type="button" disabled={!video||memoryBusy||launchBusy||prerenderBusy||imageBusy} onClick={()=>void (stillSource?startRendering():renderEnabled?stopRendering():startRendering()).catch(()=>{})}>
            {renderEnabled&&!stillSource?<Pause size={15}/>:<Play size={15}/>}{ui(memoryBusy||launchBusy?'处理中…':imageBusy?'正在渲染…':stillSource?'渲染一帧':prerenderBusy?'预渲染中…':renderEnabled?'停止渲染':'开启渲染')}
          </button>
          {singleFrameMode&&imageResult&&<button className="button" type="button" disabled={imageBusy||imageResult.frame_index!==currentFrame} onClick={()=>window.dispatchEvent(new Event('lunarc:open-image-deliver'))}>{ui('导出当前帧')}</button>}
        </div>
        {stillSource&&imageResult&&imageReference==='previous'&&previousImageResult&&<div className="image-comparison-caption">{ui('上一份结果')} · {previousImageResult.filename} · {previousImageResult.width}×{previousImageResult.height} · {previousImageResult.hdr?'HDR':'SDR'} ↔ {ui('本次结果')} · {imageResult.width}×{imageResult.height} · {imageResult.hdr?'HDR':'SDR'}</div>}
        {(playerError||nativePanel.error||(!renderEnabled&&sourceError))&&<ProcessingError message={playerError||nativePanel.error||sourceError}
          onDismiss={()=>{setPlayerError(null);setSourceError('')}}/>}
        <section ref={canvasRef} className="dlss-monitor" aria-label={ui("视频预览画布")} title={ui(showNative?'上方菜单缩放；右键拖动画面':video?.media_type==='image'?'Alt＋滚轮缩放；左键拖动画面；滚轮上下移动':'实际处理层数和模式显示在画面中')} onPointerDown={handleMouseDown} onPointerMove={handleMouseMove} onPointerUp={handleMouseUp} onPointerCancel={handleMouseUp} onContextMenu={event=>event.preventDefault()}>


          {ui(!renderEnabled?sourceUrl?<ImageViewport src={sourceUrl} zoom={zoom} onZoom={value=>{setPixelPreview(false);setZoom(value)}} pan={pan} onPan={setPan}/>:<div className="dlss-empty-preview"><strong>{ui(video?'正在读取原素材…':'导入素材，开始重建。')}</strong></div>:showNative?(sourceUrl?<ImageViewport src={sourceUrl} zoom={zoom} onZoom={value=>{setPixelPreview(false);setZoom(value)}} pan={pan} onPan={setPan}/>:<div className="dlss-empty-preview" aria-label={ui("原生 GPU 显示区域")}><strong>{ui(playerBusy||'正在准备原生预览…')}</strong></div>):stillSource&&activeFrameUrl?<ImageViewport src={activeFrameUrl} paired view={view} zoom={zoom} onZoom={value=>{setPixelPreview(false);setZoom(value)}} pan={pan} onPan={setPan} split={splitX} onSplit={value=>{splitRef.current=value;setSplitX(value)}}/>:activeFrameUrl?<img src={activeFrameUrl} alt={ui("当前预览帧")} className="dlss-preview-image" draggable={false}
            onLoad={event=>{setPreviewImageSize({width:event.currentTarget.naturalWidth,height:event.currentTarget.naturalHeight})}}
            style={pixelPreview&&previewImageSize?{width:previewImageSize.width/devicePixelRatio,height:previewImageSize.height/devicePixelRatio}:undefined}/>:<div className="dlss-empty-preview"><div className="empty-glyph"><span/><span/><span/></div><strong>{ui(video?(renderEnabled?'正在准备预览…':'点击“开启渲染”后生成预览'):'导入素材，开始重建。')}</strong></div>)}
          {ui(zoom>1.01&&<div className="pan-hint"><Move size={13}/>{ui(showNative?'右键拖动画布以查看细节':video?.media_type==='image'||!renderEnabled?'左键拖动画面 · 滚轮上下移动':'右键拖动画布以查看细节')}</div>)}
          {ui(renderEnabled&&!nativeRequired&&!stillSource&&view==='split'&&activeFrameUrl&&<div className="split-marker" aria-hidden="true" style={{left:`${splitX*100}%`}}/>)}
          {ui(showSdrMapping&&<div className="preview-warning" title={ui(showNative?'当前显示器未启用 HDR，原生预览正在映射为 SDR。':'当前预览将 HDR 映射到 SDR 显示，不改变 HDR 导出结果。')}><AlertTriangle size={13}/>{ui("HDR → SDR 预览")}</div>)}
          {ui(showNative&&hdrDisplayExpected&&nativeHdrActive===true&&<div className="preview-warning">HDR · FP16 scRGB</div>)}
          {ui(!renderEnabled&&sourceUrl&&<div className="preview-execution">{ui('原素材 · 未开启效果渲染')}</div>)}
          {ui(!previewPending && previewStats?.layer_count && <div className="preview-execution" aria-label={ui("预览实际执行")} title={ui(`实际尺寸 ${previewStats.preview_resolution?.join(' × ')} · 处理 ${Number(previewStats.processing_fps).toFixed(1)} FPS · 播放 ${Number(previewStats.presentation_fps||0).toFixed(1)} FPS · 跳过 ${previewStats.dropped_frames||0} 帧${previewStats.solver?` · GPU 融合 ${previewStats.solver.converged===false?'未收敛':'已执行'}`:''}`)}>
            {ui(previewStats.render_pipeline?.includes('nr') ? `${previewStats.layer_count} 层` : previewStats.render_pipeline?.includes('vsr') ? 'VSR' : previewStats.render_pipeline?.includes('dlaa') ? 'DLAA' : settings.truehdr_enabled ? 'TrueHDR' : '原帧预览')}
            {ui(previewStats.render_pipeline?.includes('vsr') && <>{ui((() => {
              const stages=[...(previewStats.native_input_effect?.stages||[]),...(previewStats.native_output_effect?.stages||[])]
              const vsr=stages.find((stage:any)=>stage.stage==='vsr')
              const grade=['低','中','高','极高'][vsr?.counters?.quality_level-1]
              return vsr?.counters?.model_loaded ? `${grade?` · ${ui(grade)}`:''}${vsr.output_size?` · ${vsr.output_size.join('×')}`:''}` : ''
            })())}</>)}
            {ui(previewStats.layer_count > 1 && <> · {ui(previewStats.fusion_mode_actual?.startsWith('sequential') ? '顺序堆叠' : previewStats.fusion_mode_actual?.includes('data_consistency') ? '多尺度融合' : '仅第一层')}</>)}
            {ui(isPlaying && <> · {ui(Number(previewStats.presentation_fps || previewStats.processing_fps).toFixed(1))} fps</>)}
            {ui(previewStats.solver?.converged === false && <>{ui(" · 未收敛")}</>)}
          </div>)}

        </section>
        <div className="dlss-material"><strong title={ui(video?.path)}>{video?.filename||ui('未载入素材')}</strong><span>{ui(video?`${video.width} × ${video.height} · ${video.is_hdr?(video.media_type==='image'?'HDR':'HDR10'):'SDR'}`:'')}</span></div>
        {ui(video?.media_type!=='image'&&<div className="dlss-transport">
          <SegmentTimeline video={video} settings={settings} currentFrame={currentFrame} onSeek={handleSeek} onSettingsChange={handleParamsBatchChange} onStatus={onStatusChange} onBeforeRender={stopRendering} onPause={async()=>{if(!memoryPausedRef.current)await stopRendering()}} previewRequest={previewRequest} onRenderBusy={setPrerenderBusy}/>
          <div className="transport-buttons"><button type="button" aria-label={ui("跳至首帧")} onClick={()=>handleSeek(0)}><SkipBack size={16}/></button><button type="button" aria-label={ui("后退一帧")} onClick={()=>handleSeek(currentFrame-1)}>−1</button><button type="button" className="transport-play" disabled={!renderEnabled||stillSource} onClick={togglePlay} aria-label={ui(isPlaying?'暂停播放':'开始播放')}>{ui(isPlaying?<Pause size={16}/>:<Play size={16} />)}</button><button type="button" aria-label={ui("前进一帧")} onClick={()=>handleSeek(currentFrame+1)}>+1</button><button type="button" aria-label={ui("跳至末帧")} onClick={()=>handleSeek((video?.frames||1)-1)}><SkipForward size={16}/></button><button type="button" aria-pressed={loop} aria-label={ui("循环播放")} onClick={()=>setLoop(!loop)}><RotateCcw size={15}/></button></div>
          <div className="transport-meta">{ui(video?`${video.fps||30} fps · ${video.duration_seconds.toFixed(2)} s`:'—')}</div>
        </div>)}
      </main>
      <QualityInspector image={imageSource} settings={settings} onSettingsChange={onSettingsChange} onStatusChange={onStatusChange}
        primary={<>
          <PresetPicker settings={settings} onChange={handleParamsBatchChange} onStatus={onStatusChange}/>
          <RenderPipeline image={video?.media_type==='image'} settings={settings} nrModelPresets={nrModelPresets} onChange={handleParamsBatchChange}/>
          {ui(renderStages(settings).includes('nr')&&<>
          <CollapsibleSection title={ui("DLSSNR 参数")}>
          {ui(activeLayerCount(settings.reconstruction_mode || 'single') > 1 &&
            <div className="nr-layer-tabs" role="group" aria-label={ui("图层参数")}>
              {ui(Array.from({length:activeLayerCount(settings.reconstruction_mode)}, (_, index) =>
                <button type="button" key={index} aria-pressed={activeLayer === index}
                  onClick={() => setSelectedLayer(index)}>{ui(`第 ${index+1} 层`)}</button>))}
            </div>)}
          <LayerNRControls settings={settings} index={activeLayer} modelPresets={nrModelPresets} onChange={changeNR}/>
          </CollapsibleSection>
          </>)}
        </>}
        basic={<div className="dlss-inspector-content">
          {renderStages(settings).includes('nr')&&<section className="dlss-control-card"><div className="dlss-card-head"><div><h3>{ui("神经重建")}</h3></div></div>
            <label className="dlss-check-row"><input type="checkbox" checked={settings.ui_correction===1} onChange={e=>handleParamChange('ui_correction',e.target.checked?1:0)}/>{ui("NR 界面修正")}<ParameterHelp name="ui_correction" label="NR 界面修正" /></label>
          </section>}
          {!imageSource&&<><section className="dlss-control-card"><div className="dlss-card-head"><div><h3>{ui("引导与光流")}</h3></div><span className="dlss-chip">{ui(settings.flow_engine.toUpperCase())}</span></div>
            <label className="dlss-control-label">{ui("引导模式")}<select value={settings.guidance_mode} onChange={e=>handleParamChange('guidance_mode',Number(e.target.value))}><option value={0}>{ui("关闭 · 零运动入口")}</option><option value={1}>{ui("启用 · 光流引导")}</option></select></label>
            {ui(settings.guidance_mode===1&&<><label className="dlss-control-label">{ui("光流引擎")}<select value={settings.flow_engine} onChange={e=>handleEngineChange(e.target.value)}><option value="nvof">NVIDIA Optical Flow</option><option value="fidelityfx">AMD FidelityFX Optical Flow</option><option value="dis">OpenCV DIS</option><option value="raft">RAFT-Large</option></select></label>
              {ui(settings.flow_engine==='fidelityfx'?<label className="dlss-control-label"><span>{ui("引擎质量")}<ParameterHelp name="flow_quality_amd" label="光流质量" /></span><select aria-label={ui("AMD 光流质量")} value={settings.flow_quality==='Performance'?'Performance':'Quality'} onChange={e=>handleParamChange('flow_quality',e.target.value)}><option value="Performance">{ui("性能")}</option><option value="Quality">{ui("质量")}</option></select></label>:settings.flow_engine==='nvof'?<div className="dlss-control-grid"><label>{ui("网格")}<select value={nvofGrid} onChange={e=>handleNvofGridChange(e.target.value)}>{ui(nvofGrids.map((g:any)=><option key={g.id} value={String(g.id)}>{ui(g.id)} × {ui(g.id)}{ui(" 网格")}</option>))}</select></label><label>{ui("预设")}<select value={nvofPreset} onChange={e=>handleNvofPresetChange(e.target.value)}>{ui(nvofPresets.map((p:any)=><option key={p.id} value={p.id.toUpperCase()}>{ui(FLOW_QUALITY_LABELS[p.id.toUpperCase()] || p.label)}</option>))}</select></label></div>:<label className="dlss-control-label"><span>{ui("引擎质量")}<ParameterHelp name={settings.flow_engine==='raft'?'flow_quality_raft':'flow_quality_dis'} label="光流质量" /></span><select value={settings.flow_quality} onChange={e=>handleParamChange('flow_quality',e.target.value)}>{ui((settings.flow_engine==='dis'?['Fast','Medium','Slow']:['Fast','Balanced','Quality']).map(v=><option key={v} value={v}>{ui(FLOW_QUALITY_LABELS[v] || v)}</option>))}</select></label>)}
              {ui(settings.flow_engine==='raft' && <div className="dlss-control-grid">
                <label><span>{ui("分析长边")}<ParameterHelp name="raft_analysis_edge" label="分析长边" /></span><select aria-label={ui("RAFT 分析长边")} value={settings.raft_analysis_edge ?? (settings.flow_quality==='Fast'?512:settings.flow_quality==='Quality'?1080:720)} onChange={e=>handleParamChange('raft_analysis_edge',Number(e.target.value))}>{ui([512,720,1080].map(v=><option key={v} value={v}>{ui(v)}</option>))}</select></label>
                <label><span>{ui("迭代次数")}<ParameterHelp name="raft_iterations" label="迭代次数" /></span><input aria-label={ui("RAFT 迭代次数")} type="number" min={1} max={24} step={1} value={settings.raft_iterations ?? (settings.flow_quality==='Fast'?4:settings.flow_quality==='Quality'?12:6)} onChange={e=>{const v=Number(e.target.value);if(Number.isInteger(v)&&v>=1&&v<=24)handleParamChange('raft_iterations',v)}} /></label>
              </div>)}
              <div className="dlss-control-grid">{ui((['motion_scale_x','motion_scale_y'] as const).map((key,i)=><label key={key}>{ui("运动 ")}{ui(i?'Y':'X')} · {ui(settings[key].toFixed(2))}<input type="range" min={0} max={4} step={.05} value={settings[key]} onChange={e=>handleParamChange(key,Number(e.target.value))}/></label>))}</div>
              {ui(settings.flow_engine==='nvof'&&<div className="provider-summary"><span className={isNvofAvailable?'provider-state-ready':'provider-state-off'}/><span>{ui(isNvofAvailable?`${displayGpu} · 设备可用`:nvofProvider?.error||'NVOF 设备不可用')}</span></div>)}
            </>)}
          </section>
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>{ui("导出后端")}</h3></div><span className="dlss-chip dlss-chip-export">{ui("仅导出")}</span></div><label className="dlss-control-label">{ui("执行链路")}<select aria-label={ui("导出后端")} value={settings.export_backend} onChange={e=>{if(e.target.value!=='gpu_transfer'&&(settings.reconstruction_mode||'single')!=='single'){onStatusChange?.('Dual / Triple 只支持 GPU Transfer；请先切回 Single');return}if(e.target.value!=='gpu_transfer'&&hdrOn){onStatusChange?.('实验性 HDR/scRGB 只在 GPU Transfer 上运行；请先关闭该开关');return}handleParamChange('export_backend',e.target.value)}}><option value="auto">{ui("Auto · Legacy 路径")}</option><option value="legacy">Legacy FFmpeg</option><option value="gpu_transfer">{ui("GPU Transfer · 条件不符则拒绝")}</option></select></label><p className="dlss-note">{ui("原生预览复用 GPU Transfer 处理核心；实际层数和处理帧率显示在画面中。")}</p><details className="dlss-note"><summary>{ui("支持条件与预览范围")}</summary><p>{ui("单层预览与所有导出链使用同一 GPU 宿主。关闭引导使用零运动纹理入口；开启光流需所选 NVIDIA、AMD 或 RAFT CUDA 引导后端可用。Dual / Triple 在原生预览和 GPU Transfer 导出中使用独立会话。缺少必要入口或输入条件不符时明确拒绝，不会换用其他宿主。旧 host-pointer 配置已移除，载入时须迁移到当前 GPU 宿主。")}</p></details></section></>}
        </div>}
        layers={<div className="dlss-inspector-content">
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>{ui("工作分辨率")}</h3></div><span className="dlss-chip">{ui(Math.round(settings.working_scale*100))}%</span></div><label className="dlss-slider"><span>{ui("工作缩放")}<output>{ui(Math.round(settings.working_scale*100))}%</output></span><input aria-label={ui("工作分辨率缩放")} type="range" min={25} max={100} step={1} value={Math.round(settings.working_scale*100)} onChange={e=>handleWorkingScaleChange(Number(e.target.value)/100)}/></label><div className="scale-presets">{ui([1,.75,.67,.63,.5,.25].map(s=><button type="button" key={s} aria-pressed={Math.abs(settings.working_scale-s)<.005} onClick={()=>handleWorkingScaleChange(s)}>{ui(Math.round(s*100))}%</button>))}</div>{ui((()=>{const [w,h]=resolveWorkingDims(video?.width||3840,video?.height||2160,settings.working_scale);return <div className="dimension-row"><span>{ui("源 ")}{ui(video?.width||3840)} × {ui(video?.height||2160)}</span><strong>{ui("请求 ")}{ui(w)} × {ui(h)}</strong></div>})())}</section>
          <p className="dlss-note">{ui("工作缩放支持 25%–100% 的整数百分比；显示尺寸由共享解析器按 GPU 对齐规则计算。")}</p><section className="dlss-control-card"><div className="dlss-card-head"><div><h3>{ui("残差重构")}</h3></div><span className="dlss-chip">{ui(settings.residual_pyramid?'已请求':'关闭')}</span></div><label className="dlss-check-row"><input type="checkbox" checked={settings.residual_pyramid} disabled={settings.experimental_hdr_scrgb || settings.reconstruction_mode !== 'single'} title={ui("仅控制低分辨率重构时的源细节保留，不关闭 DLSSNR。HDR 与多层融合始终保留源细节。")} onChange={e=>handleParamChange('residual_pyramid',e.target.checked)}/>{ui("保留源细节残差")}</label><label className="dlss-slider"><span>{ui("残差倍率")}<output>{ui(settings.residual_strength.toFixed(2))}×</output></span><input aria-label={ui("残差倍率")} type="range" min={0} max={1.5} step={.05} value={settings.residual_strength} onChange={e=>handleParamChange('residual_strength',Number(e.target.value))}/></label></section>
          <section className="dlss-control-card"><div className="dlss-card-head"><div><h3>{ui("尺度与融合")}</h3></div><span className="dlss-chip">{ui("预览 / 导出")}</span></div><label className="dlss-control-label">{ui("重建层数")}<select value={settings.reconstruction_mode||'single'} onChange={e=>handleParamsBatchChange(chooseReconstructionMode(settings,e.target.value as DLSSSettings['reconstruction_mode']))}><option value="single">{ui("单层")}</option><option value="dual">{ui("双层")}</option><option value="triple">{ui("三层")}</option></select></label>
            <div className="layer-diagram" aria-label={ui("请求的独立层尺寸与融合权重")}>{ui(Array.from({length:activeLayerCount(settings.reconstruction_mode||'single')},(_,i)=>{const scale=i===0?settings.working_scale:settings.layer_scales?.[i]??settings.working_scale;const [w,h]=resolveWorkingDims(video?.width||3840,video?.height||2160,scale);return <div className="layer-node" key={i}><span className="layer-index">L{ui(i+1)}</span><div><strong>{ui(w)} × {ui(h)}</strong><small>{ui(Math.round(scale*100))}% · weight {ui((settings.layer_weights?.[i]||1).toFixed(2))}</small></div>{ui(i<activeLayerCount(settings.reconstruction_mode||'single')-1&&<span className="layer-link">→</span>)}</div>}))}</div>{!imageSource&&<><p className="dlss-note">{ui("以上为当前请求值；下面的最近任务实际状态来自该任务的后端诊断，不代表实时预览。")}</p>
            <div className="dlss-control-card"><div className="dlss-card-head"><div><h3>{ui("最近任务实际")}</h3></div><span className="dlss-chip">{ui(recentFusionJob?.status||'暂无报告')}</span></div>{ui(recentFusionJob?<><p className="dlss-note">{recentFusionJob.name}</p>{ui(recentFusionJob.report?.fusion_mode&&<div className="dimension-row"><span>{ui("后端报告模式")}</span><strong>{ui(recentFusionJob.report.fusion_mode)}</strong></div>)}{ui(Array.isArray(recentFusionJob.report?.layer_working_dims)&&<div className="dimension-row"><span>{ui("实际层尺寸")}</span><strong>{ui(recentFusionJob.report.layer_working_dims.map((d:number[])=>`${d[0]}×${d[1]}`).join(' / '))}</strong></div>)}{ui(Array.isArray(recentFusionJob.report?.layer_dlss_calls)&&<div className="dimension-row"><span>{ui("各层 DLSS 调用数")}</span><strong>{ui(recentFusionJob.report.layer_dlss_calls.join(' / '))}</strong></div>)}{ui(Array.isArray(recentFusionJob.report?.fusion_solver_frames)&&recentFusionJob.report.fusion_solver_frames.length>0?<div className="dimension-row"><span>{ui("CG 求解帧数 / 收敛")}</span><strong>{ui(recentFusionJob.report.fusion_solver_frames.length)} / {ui(recentFusionJob.report.fusion_solver_frames.every((r:{converged?:boolean})=>r.converged)?'全部':'部分或无')}</strong></div>:recentFusionJob.report?.fusion_solver_frames?.details_omitted?<div className="dimension-row"><span>{ui("CG 求解帧数 / 收敛")}</span><strong>{ui(recentFusionJob.report.fusion_solver_frames.item_count)} / {ui(recentFusionJob.report.fusion_solver_frames.all_converged?"全部":"部分或无")}</strong></div>:<p className="dlss-note">{ui("该任务诊断未包含 CG 求解帧记录；不推断实际求解状态。")}</p>)}</>:<p className="dlss-note">{ui("尚无可读取的终态任务诊断。导出任务完成后，此处读取最新任务报告。")}</p>)}</div>
</>}
            {ui((settings.reconstruction_mode||'single')!=='single'&&Array.from({length:activeLayerCount(settings.reconstruction_mode)},(_,i)=>{const scale=i===0?settings.working_scale:settings.layer_scales?.[i]??settings.working_scale;return <div className="layer-inputs" key={i}><strong>{ui(`第 ${i+1} 层 · ${Math.round(scale*100)}%`)}</strong><button type="button" className="button quiet" aria-expanded={expandedLayer===i} aria-controls={`nr-layer-${i}`} onClick={()=>{setSelectedLayer(i);setExpandedLayer(expandedLayer===i?null:i)}}>{ui(expandedLayer===i?"收起本层参数":"调整本层参数")}</button>{ui(i>0&&<label>{ui("独立输入比例")}<select value={Math.round(scale*100)} onChange={e=>handleLayerScaleChange(i,Number(e.target.value)/100)}>{ui(Array.from({length:76},(_,at)=>at+25).map(v=><option key={v} value={v} disabled={settings.fusion_mode!=="sequential"&&(settings.layer_scales?.some((item,at)=>at!==i&&Math.round(item*100)===v)||Math.round(settings.working_scale*100)===v)}>{ui(v)}%</option>))}</select></label>)}{ui(settings.fusion_mode!=="sequential"&&<label className="dlss-slider"><span>{ui("观测权重")}<output>{ui((settings.layer_weights?.[i]||1).toFixed(2))}</output></span><input aria-label={ui(`第 ${i+1} 层观测权重`)} type="range" min={.05} max={3} step={.05} value={settings.layer_weights?.[i]||1} onChange={e=>handleLayerWeightChange(i,Number(e.target.value))}/></label>)}{ui(expandedLayer===i&&<div id={`nr-layer-${i}`} className="layer-inline-params"><LayerNRControls settings={settings} index={i} modelPresets={nrModelPresets} onChange={(key,value)=>handleParamsBatchChange(changeLayerNR(settings,i,key,value))}/></div>)}</div>}))}
            {ui((settings.reconstruction_mode||'single')!=='single'&&<><label className="dlss-control-label">{ui("融合输出")}<select value={settings.fusion_mode||'catmull_baseline'} onChange={e=>handleParamsBatchChange(chooseFusionMode(settings,e.target.value as DLSSSettings['fusion_mode']))}><option value="sequential">{ui("顺序堆叠")}</option><option value="catmull_baseline">{ui("仅第一层 · 对照")}</option><option value="data_consistency_experimental">{ui("多层融合 · 实验性")}</option></select></label>{ui(settings.fusion_mode==='data_consistency_experimental'&&<details className="solver-details"><summary>{ui("GPU CG 求解器参数")}</summary><div className="dlss-control-grid"><label>β<select value={settings.fusion_beta??.1} onChange={e=>handleParamChange('fusion_beta',Number(e.target.value))}>{ui([.01,.05,.1,.2,.5,1].map(v=><option key={v} value={v}>{ui(v)}</option>))}</select></label><label>{ui("迭代")}<select value={settings.fusion_max_iterations??128} onChange={e=>handleParamChange('fusion_max_iterations',Number(e.target.value))}>{ui([32,64,128,256,512].map(v=><option key={v} value={v}>{ui(v)}</option>))}</select></label><label>{ui("相对阈值")}<select value={settings.fusion_relative_tolerance??.001} onChange={e=>handleParamChange('fusion_relative_tolerance',Number(e.target.value))}>{ui([.01,.001,.0001,.00001].map(v=><option key={v} value={v}>{ui(v)}</option>))}</select></label></div></details>)}<p className="dlss-note">{ui("顺序堆叠逐层处理上一层结果。多尺度融合为独立分支；“仅第一层”用于对照。")}</p></>)}
          </section>
        </div>}
        hdr={<div className="dlss-inspector-content"><section className="dlss-control-card"><div className="dlss-card-head"><div><h3>HDR / scRGB</h3></div><span className="dlss-chip dlss-chip-export">{ui("仅导出")}</span></div><label className="dlss-check-row"><input type="checkbox" aria-label={ui("实验性 HDR/scRGB")} checked={hdrOn} onChange={e=>saveHdrPatch({[HDR_SETTING]:e.target.checked})}/>{ui("请求实验性 HDR/scRGB 导出")}</label><label className="dlss-control-label">{ui("纸白 · scRGB")}<input type="text" inputMode="decimal" aria-label={ui("HDR 纸白 (scRGB)")} value={paperWhiteInput} onChange={e=>handlePaperWhiteChange(e.target.value)}/></label>{ui(paperWhiteRefusal?<p className="dlss-warning">{ui(paperWhiteRefusal)}</p>:<p className="dlss-note">{ui(paperWhiteLabel(Number(paperWhiteInput)))} · 1 scRGB = 80 cd/m²</p>)}{ui(hdrOn&&hdrRefusals.length>0&&<ul className="dlss-refusals" aria-label={ui("HDR 拒绝项")}>{ui(hdrRefusals.map(reason=><li key={reason}><AlertTriangle size={13}/>{ui(reason)}</li>))}</ul>)}<div className="hdr-preview-warning"><AlertTriangle size={16}/><p>{ui("HDR10 以 FP16 处理，再统一映射为 8-bit 显示；未校准，不作 HDR 色彩参考。HDR 开关控制导出。")}</p></div><p className="dlss-note">{ui("兼容性由后端能力闸门决定；不满足条件会拒绝，不会自动回退。HDR10 输出约束由服务端校验。")}</p><ul className="dlss-limitations">{ui(HDR_LIMITATIONS.map(item=><li key={item}>{ui(item)}</li>))}</ul></section></div>}
      />
    </div>
  )
}
