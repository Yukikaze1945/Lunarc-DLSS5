// localized-render-sinks
import {ui} from './i18n/locale'
import {useCallback, useEffect, useRef, useState, useSyncExternalStore, type DragEvent} from 'react'
import {Home, Layers3, Film, Image as ImageIcon, Activity, Bookmark, Settings2, Info, ArrowLeftRight, BookOpen} from 'lucide-react'
import {ActivityMark} from './components/ActivityMark'
import {activityState} from './services/activityState'
import {Header} from './components/Header'
import {BottomBar, type TabType} from './components/BottomBar'
import {DLSSPage} from './pages/DLSSPage'
import {DeliverPage} from './pages/DeliverPage'
import {ImageDeliverPage} from './pages/ImageDeliverPage'
import {WorkspacePages} from './pages/WorkspacePages'
import {ComparePage} from './pages/ComparePage'
import type {VideoInfo, DLSSSettings, SystemInfo} from './types'
import {fetchSystemInfo, fetchCurrentVideo, loadVideoFile, fetchSettings, saveSettings} from './services/api'
import {subscribeQueue, queueSnapshot} from './services/queue'
import {DEFAULT_SETTINGS, completeSettings, readWorkspace, storeWorkspace} from './services/workspace'
import {applyTheme, readTheme} from './services/appearance'
import './App.css'
import {RuntimeGate} from './components/RuntimeGate'
import {Guide} from './components/Guide'
import {droppedMediaPath} from './services/mediaDrop'
import {useLocale} from './i18n/locale'
const NAV: {id:TabType;label:string;icon:typeof Home}[] = [
  {id:'home',label:'首页',icon:Home},
  {id:'dlss',label:'重建工作台',icon:Layers3},
  {id:'deliver',label:'导出与队列',icon:Film},
  {id:'image-deliver',label:'图片导出',icon:ImageIcon},
  {id:'compare',label:'视频对比',icon:ArrowLeftRight},
  {id:'diagnostics',label:'运行与诊断',icon:Activity},
  {id:'presets',label:'预设管理',icon:Bookmark},
  {id:'settings',label:'软件设置',icon:Settings2},
  {id:'guide',label:'使用教程',icon:BookOpen},
  {id:'about',label:'关于 Lunarc',icon:Info},
]
function WorkspaceApp() {
  const [dragging, setDragging] = useState(false)
  const dragDepth = useRef(0)
  const importRevision = useRef(0)
  const importQueue = useRef<Promise<void>>(Promise.resolve())
  const [activeTab,setActiveTab]=useState<TabType>('dlss')
  const [comparePair,setComparePair]=useState<{original:string;exported:string;original_offset_seconds?:number;duration_seconds?:number}|null>(null)
  useEffect(()=>{const open=(event:Event)=>{const pair=(event as CustomEvent).detail;
    if(pair?.original&&pair?.exported){setComparePair(pair);setActiveTab('compare')}};
    window.addEventListener('lunarc:compare',open);return()=>window.removeEventListener('lunarc:compare',open)},[])
  useEffect(()=>{const open=()=>setActiveTab('image-deliver');window.addEventListener('lunarc:open-image-deliver',open);return()=>window.removeEventListener('lunarc:open-image-deliver',open)},[])
  useEffect(()=>{const open=()=>setActiveTab('deliver');window.addEventListener('lunarc:open-deliver',open);return()=>window.removeEventListener('lunarc:open-deliver',open)},[])
  const [theme,setTheme]=useState(readTheme)
  useEffect(()=>applyTheme(theme),[theme])
  const [video,setVideo]=useState<VideoInfo|null>(null)
  const [system,setSystem]=useState<SystemInfo|null>(null)
  const [settings,setSettings]=useState<DLSSSettings>(DEFAULT_SETTINGS)
  const updateSettings = useCallback((value: DLSSSettings) => setSettings(completeSettings(value)), [])
  const [recent,setRecent]=useState<string[]>(()=>readWorkspace()?.recent||[])
  const [currentFrame,setCurrentFrame]=useState(0)
  const [statusText,setStatusText]=useState('连接本地工作台…')
  const [ready,setReady]=useState(false)
  const [backendState,setBackendState]=useState<'starting'|'ready'|'recovering'|'error'>('starting')
  const [importing,setImporting]=useState(false),[appError,setAppError]=useState(false)
  const [previewActivity,setPreviewActivity]=useState({busy:false,error:false})
  const [queueErrorId,setQueueErrorId]=useState<string|null>(null)
  const observedJobs=useRef<Map<string,string>|null>(null)
  const recoveryDraft=useRef({video,settings,currentFrame});recoveryDraft.current={video,settings,currentFrame}
  const queue = useSyncExternalStore(subscribeQueue, queueSnapshot)
  useEffect(()=>{
    let cancelled=false, initialized=false, inFlight=false
    const init=async()=>{
      if(inFlight||cancelled)return
      inFlight=true
      try {
        const [sys,v,s]=await Promise.all([fetchSystemInfo(),fetchCurrentVideo(),fetchSettings()])
        if(cancelled)return
        setSystem(sys);if(v)setVideo(v)
        const draft=readWorkspace()
        if(draft){try{const restored=await saveSettings(draft.settings);if(!cancelled)setSettings(completeSettings(restored));setStatusText('已恢复参数')}catch(e:any){setSettings(completeSettings(s));setStatusText(`草稿未应用：${e.message}；保留后端参数`)}}
        else{
          const current=s.style===3?await saveSettings({style:DEFAULT_SETTINGS.style}):s
          if(cancelled)return
          setSettings(completeSettings(current));setStatusText('工作台就绪')
        }
        if(!cancelled){initialized=true;setReady(true);setBackendState('ready');setAppError(false)}
      }catch(e:any){
        const startup=await window.electronAPI?.getBackendStartup?.().catch(()=>null)
        if(!cancelled){setStatusText(startup?.state==='error'?startup.message:`后端连接失败：${e.message} · 正在重试`);setBackendState('error')}
      }finally{inFlight=false}
    }
    void init()
    const timer=setInterval(()=>{if(!initialized){void init();return}void fetchSystemInfo().then(s=>{if(!cancelled)setSystem(s)}).catch(()=>{})},5000)
    const unsubscribe=window.electronAPI?.onBackendExit?.(info=>{if(!cancelled){setReady(false);setBackendState('error');setStatusText(`后端已退出 (${info.code})，正在检查恢复`)}})
    const recovery=window.electronAPI?.onBackendRecovery?.(state=>{
      if(cancelled)return
      setBackendState(state.phase==='blocked'?'error':'recovering')
      setStatusText(state.phase==='restored'?'后端已启动，正在恢复工作台…':state.message)
      if(state.phase!=='restored'){setReady(false);return}
      void (async()=>{
        try{
          const draft=recoveryDraft.current
          if(draft.video){const restored=await loadVideoFile(draft.video.path);if(cancelled)return;setVideo(restored)}
          const restored=await saveSettings(recoveryDraft.current.settings)
          if(cancelled)return
          setSettings(completeSettings(restored));setCurrentFrame(draft.currentFrame)
          initialized=true;setReady(true);setBackendState('ready');setAppError(false);setStatusText(state.message)
          window.dispatchEvent(new Event('lunarc:backend-restored'))
        }catch(error:any){if(!cancelled){setBackendState('error');setStatusText(`后端已启动，工作台恢复失败：${error.message}`)}}
      })()
    })
    return()=>{cancelled=true;clearInterval(timer);unsubscribe?.();recovery?.()}
  },[])
  useEffect(()=>{
    if(!queue)return
    const previous=observedJobs.current
    const failed=previous&&queue.jobs.find(job=>job.status==='failed'&&previous.get(job.id)!=='failed')
    if(failed)setQueueErrorId(failed.id)
    else if(queue.jobs.some(job=>job.status==='running'||(job.status==='completed'&&previous?.get(job.id)==='running')))setQueueErrorId(null)
    else setQueueErrorId(current=>current&&queue.jobs.some(job=>job.id===current)?current:null)
    observedJobs.current=new Map(queue.jobs.map(job=>[job.id,job.status]))
  },[queue])
  useEffect(()=>{const q=queue;if(!q)return;try{
    const terminal=q.jobs.filter(j=>['completed','failed','cancelled','skipped'].includes(j.status)).map(j=>({id:j.id,name:j.name,status:j.status,input_path:j.input_path,output_path:j.output_path,frame_range:j.frame_range,source_fps:j.source_fps,error:j.error,recorded:new Date().toISOString()}))
    if(terminal.length){let history:any[]=[];try{const h=JSON.parse(localStorage.getItem('lunarc.history.v1')||'[]');if(Array.isArray(h))history=h}catch{}
      const merged=[...terminal.map(j=>({...j,recorded:history.find(h=>h.id===j.id)?.recorded||j.recorded})),...history.filter(h=>!terminal.some(j=>j.id===h.id))].slice(0,100);const serialized=JSON.stringify(merged);if(serialized!==localStorage.getItem('lunarc.history.v1'))localStorage.setItem('lunarc.history.v1',serialized)}
    }catch{}},[queue])
  useEffect(()=>{if(ready){try{storeWorkspace(settings,recent)}catch{setStatusText('本机草稿无法保存；当前后端参数仍保留')}}},[settings,recent,ready])
  const importFile=async(path:string)=>{
    const revision=++importRevision.current;setStatusText('正在读取素材…');setImporting(true);setAppError(false)
    // Serialize source replacement on the backend too, and skip obsolete drops.
    const request=importQueue.current.then(async()=>{
      if(revision!==importRevision.current)return
      try{const loaded=await loadVideoFile(path);if(revision!==importRevision.current)return;setVideo(loaded);setCurrentFrame(0);setRecent(r=>[path,...r.filter(x=>x!==path)].slice(0,8));setActiveTab('dlss');setStatusText(`已载入 ${loaded.filename}`)}catch(e:any){if(revision===importRevision.current){setAppError(true);setStatusText(`素材载入失败：${e.message}`)}}finally{if(revision===importRevision.current)setImporting(false)}
    })
    importQueue.current=request.catch(()=>{});await request
  }
  const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer.types).includes('Files')
  const dropMedia = (event: DragEvent) => {
    if (!hasFiles(event)) return
    event.preventDefault(); event.stopPropagation(); dragDepth.current=0; setDragging(false)
    const resolve = window.electronAPI?.getPathForFile
    if (!resolve) { setStatusText('请在桌面程序中拖入素材'); return }
    try {
      const path=droppedMediaPath(event.dataTransfer.files, resolve)
      if (path) void importFile(path)
      else setStatusText('请拖入视频或图片文件')
    } catch (error: any) { setStatusText(`素材载入失败：${error.message}`) }
  }
  const applySettings=async(next:DLSSSettings)=>{const saved=await saveSettings(completeSettings(next));setSettings(completeSettings(saved));setStatusText('参数已验证并保存')}
  const running=queue?.jobs.find(j=>j.status==='running')
  const pending=queue?.jobs.filter(j=>j.status==='pending').length||0
  const markState=activityState({recovering:backendState==='recovering',
    error:backendState==='error'||appError||previewActivity.error||Boolean(queueErrorId&&queue?.jobs.some(job=>job.id===queueErrorId&&job.status==='failed')),
    exporting:Boolean(running),loading:backendState==='starting'||importing||previewActivity.busy})
  const queueSummary=running?`${running.name} · ${Math.round(running.progress)}%`:pending?`${pending} 个任务等待导出`:'队列空闲'
  return <div className="app-shell" onDragEnter={event=>{if(hasFiles(event)){event.preventDefault();if((event.target as Element).closest?.('.dlssnr-runtime')){dragDepth.current=0;setDragging(false);return}dragDepth.current++;setDragging(true)}}}
    onDragOver={event=>{if(hasFiles(event)){event.preventDefault();event.dataTransfer.dropEffect='copy'}}}
    onDragLeave={event=>{if(hasFiles(event)){event.preventDefault();dragDepth.current=Math.max(0,dragDepth.current-1);if(!dragDepth.current)setDragging(false)}}}
    onDrop={dropMedia}>
    {ui(dragging&&<div className="media-drop-overlay" role="status"><span>{ui("松开以载入素材")}</span></div>)}
    <aside className="navigation"><ActivityMark state={markState}/><nav aria-label={ui("主导航")}>{ui(NAV.map(n=><button key={n.id} aria-label={ui(n.label)} title={ui(n.label)} className={activeTab===n.id?'active':''} onClick={()=>setActiveTab(n.id)} aria-current={activeTab===n.id?'page':undefined}><n.icon size={20}/></button>))}</nav></aside>
    <div className="app-content"><Header activeTab={activeTab} onNavigate={setActiveTab} theme={theme} onThemeChange={()=>setTheme(t=>t==='dark'?'light':'dark')} onFileLoaded={importFile} onError={message=>{setAppError(true);setStatusText(message)}} image={video?.media_type==='image'}/><main className="page-host">
      <div className={`mounted-page ${activeTab==='dlss'?'':'hidden'}`}><DLSSPage video={video} settings={settings} onSettingsChange={updateSettings} currentFrame={currentFrame} setCurrentFrame={setCurrentFrame} onStatusChange={setStatusText} onActivityChange={setPreviewActivity} isActive={ready&&activeTab==='dlss'}/></div>
      <div className={`mounted-page ${activeTab==='deliver'?'':'hidden'}`}><DeliverPage video={video?.media_type==='image'?null:video} settings={settings} onSettingsChange={updateSettings} onStatusChange={setStatusText} isActive={activeTab==='deliver'}/></div>
      <div className={`mounted-page ${activeTab==='image-deliver'?'':'hidden'}`}>{ui(video?<ImageDeliverPage isActive={activeTab==='image-deliver'} video={video} currentFrame={currentFrame} settings={settings} onStatusChange={setStatusText}/>:<div className="workspace-page"><h1>{ui("图片导出")}</h1><p>{ui("先导入图片或视频，再选择格式与保存位置。")}</p></div>)}</div>
      {ui(activeTab==='compare'&&<ComparePage initialPair={comparePair}/>)}
      {ui(activeTab==='guide'&&<Guide/>)}
      {ui(!['dlss','deliver','image-deliver','compare','guide'].includes(activeTab)&&<WorkspacePages page={activeTab} video={video} system={system} settings={settings} recent={recent} onImport={importFile} onNavigate={setActiveTab} onApply={applySettings} onStatus={setStatusText}/>)}
    </main><BottomBar system={system} statusText={statusText} queueSummary={queueSummary} onQueueClick={()=>setActiveTab('deliver')}/></div>
  </div>
}
export function App(){useLocale();return <RuntimeGate><WorkspaceApp/></RuntimeGate>}
export default App
