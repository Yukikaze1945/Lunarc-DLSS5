// localized-render-sinks
import {ui} from '../i18n/locale'
import {jobComparison} from '../services/jobComparison'
import {useEffect, useState} from 'react'
import {ArrowUpRight, FolderOpen, RotateCcw, Download, RefreshCw, Copy} from 'lucide-react'
import type {DLSSSettings, SystemInfo, VideoInfo, RenderJob} from '../types'
import {Brand} from '../components/Brand'
import {ComponentPanel} from '../components/ComponentPanel'
import {AppUpdatePanel} from '../components/AppUpdatePanel'
import {EnvironmentPanel} from '../components/EnvironmentPanel'
import {DlssnrRuntimePanel} from '../components/DlssnrRuntimePanel'
import {McpConnectionPanel} from '../components/McpConnectionPanel'
import {fetchDiagnosticLogs, fetchFlowProviders, fetchJobDiagnostics, fetchQueue, openNativeFileDialog} from '../services/api'
import {completeSettings, DEFAULT_SETTINGS} from '../services/workspace'
import {LanguagePicker} from '../components/LanguagePicker'
import {PresetLibrary} from '../components/PresetLibrary'
import {GpuMemoryPanel} from '../components/GpuMemoryPanel'
import type {TabType} from '../components/BottomBar'

type Props = {page: TabType; video: VideoInfo|null; system: SystemInfo|null; settings: DLSSSettings;
  recent: string[]; onImport: (p: string)=>void; onNavigate: (p:TabType)=>void;
  onApply: (s:DLSSSettings)=>Promise<void>; onStatus:(s:string)=>void}
function stored<T>(key:string,fallback:T):T {try{return JSON.parse(localStorage.getItem(key)||'null')??fallback}catch{return fallback}}
export function WorkspacePages(p:Props) {
  const [message,setMessage]=useState('')
  const [logs,setLogs]=useState<any[]>([])
  const [jobs,setJobs]=useState<RenderJob[]>([])
  const [contracts,setContracts]=useState<any>(null)
  const [diagnostics,setDiagnostics]=useState<any>(null)
  const [selected,setSelected]=useState('')
  const [query,setQuery]=useState('')
  const [density,setDensity]=useState(()=>localStorage.getItem('lunarc.density')||'comfortable')
  const [reduced,setReduced]=useState(()=>localStorage.getItem('lunarc.reduced-motion')==='true')
  const [busy,setBusy]=useState(false)
  const [vsrRuntimePath,setVsrRuntimePath]=useState(p.settings.vsr_runtime_path||'')
  const [dlssDllPath,setDlssDllPath]=useState(p.settings.dlss_dll_path||'')
  useEffect(()=>{setVsrRuntimePath(p.settings.vsr_runtime_path||'');setDlssDllPath(p.settings.dlss_dll_path||'')},[p.settings.vsr_runtime_path,p.settings.dlss_dll_path])
  const vramLabel = p.system?.vram_total_gb != null ? `${p.system.vram_total_gb} GB · 本进程缓存 ${p.system.vram_used_gb ?? '未知'} GB` : '未知'
  async function refresh() {
    try { const [l,q,f]=await Promise.all([fetchDiagnosticLogs(),fetchQueue(),fetchFlowProviders()]);setLogs(l.logs);setJobs(q.jobs);setContracts(f);setMessage('已刷新实时状态') }
    catch(e:any){setMessage(e.message)}
  }
  useEffect(()=>{if(p.page==='diagnostics'){void refresh(); const t=setInterval(refresh,3000);return()=>clearInterval(t)}},[p.page])
  useEffect(()=>{document.documentElement.dataset.density=density;localStorage.setItem('lunarc.density',density)},[density])
  useEffect(()=>{document.documentElement.dataset.reducedMotion=String(reduced);localStorage.setItem('lunarc.reduced-motion',String(reduced))},[reduced])
  async function apply(s:DLSSSettings){setBusy(true);try{await p.onApply(s);setMessage('参数已验证并保存')}catch(e:any){setMessage(`未应用：${e.message}`)}finally{setBusy(false)}}
  function download(){const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify({version:1,settings:p.settings},null,2)],{type:'application/json'}));a.href=url;a.download='lunarc-settings.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
  const load=async()=>{try{const path=await openNativeFileDialog();if(path)p.onImport(path)}catch(e:any){p.onStatus(e.message)}}
  if(p.page==='home')return <div className="workspace-page home-page">
    <section className="home-hero"><div><span className="eyebrow">{ui("LUNARC / RECONSTRUCTION WORKSPACE")}</span><h1>{ui("每一帧，")}<br/><em>{ui("清晰可见。")}</em></h1><p>{ui("载入素材，调整画面，导出视频。")}</p><div className="hero-actions"><button className="button primary" onClick={load}><FolderOpen size={17}/>{ui("导入视频或图像")}</button>{ui(p.video&&<button className="button" onClick={()=>p.onNavigate('dlss')}>{ui("继续当前素材")}<ArrowUpRight size={16}/></button>)}</div></div><div className="optical-art" aria-hidden="true"><i/><i/><i/><span>01<br/>LIGHT / MOTION</span></div></section>
    <div className="home-grid"><section className="panel"><span className="eyebrow">{ui("RECENT MATERIALS")}</span><h2>{ui("最近载入")}</h2>{ui(p.recent.length?p.recent.map(path=><button key={path} className="recent-row" onClick={()=>p.onImport(path)} title={path}><FolderOpen size={16}/><span>{path.split(/[\\/]/).pop()}<small>{path}</small></span><ArrowUpRight size={16}/></button>):<p className="muted">{ui("载入的素材会显示在这里。原文件保持不变。")}</p>)}</section><section className="panel pipeline-card"><span className="eyebrow">{ui("WORKFLOW")}</span><h2>{ui("三步完成交付")}</h2><ol><li><b>01</b><span>{ui("载入与检查")}<small>{ui("读取色彩、帧率和素材信息")}</small></span></li><li><b>02</b><span>{ui("重建与比较")}<small>{ui("单层 / 双层 / 三层 · 独立历史")}</small></span></li><li><b>03</b><span>{ui("验证与导出")}<small>{ui("GPU 视频 + 原音轨、字幕与 metadata")}</small></span></li></ol></section></div>
    <div className="capability-strip"><span><i className="status-dot"/>{ui(p.system?.backend_ready?'后端已连接':'后端连接中')}</span><span>{ui(p.system?.gpu||'GPU 信息待连接')}</span><span>{ui("HDR10 · 实验性")}</span><span>{ui("性能未测量")}</span></div>
  </div>
  if(p.page==='presets')return <PresetLibrary/>
  if(p.page==='settings')return <div className="workspace-page"><div className="page-title"><span className="eyebrow">{ui("PREFERENCES")}</span><h1>{ui("软件设置")}</h1><p>{ui("按你的习惯设置工作台。")}</p><button className="button" onClick={()=>p.onNavigate('guide')}>{ui("打开使用教程")}</button></div><ComponentPanel/><McpConnectionPanel/><details className="panel"><summary>{ui("运行库路径覆盖")}</summary><div className="settings-grid"><label className="field">{ui("VFX 运行库目录")}<input value={vsrRuntimePath} placeholder={ui("自动查找")} onChange={e=>setVsrRuntimePath(e.target.value)}/></label><label className="field">{ui("DLSS DLL 路径")}<input value={dlssDllPath} placeholder={ui("自动查找")} onChange={e=>setDlssDllPath(e.target.value)}/></label></div><button className="button" disabled={busy} onClick={()=>void apply({...p.settings,vsr_runtime_path:vsrRuntimePath,dlss_dll_path:dlssDllPath})}>{ui("保存路径覆盖")}</button></details><div className="settings-grid"><section className="panel"><h2>{ui("工作台显示")}</h2><LanguagePicker/><label className="field">{ui("控件密度")}<select value={density} onChange={e=>setDensity(e.target.value)}><option value="comfortable">{ui("舒适")}</option><option value="compact">{ui("紧凑")}</option></select></label><label className="check-field"><input type="checkbox" checked={reduced} onChange={e=>setReduced(e.target.checked)}/>{ui("减少界面动效")}</label><p></p></section><section className="panel"><h2>{ui("配置与兼容性")}</h2><p>{ui("参数保存在本机，可备份或恢复默认值。")}</p><div className="inline-form"><button className="button" onClick={download}><Download size={15}/>{ui("备份参数")}</button><button className="button" disabled={busy} onClick={()=>{if(confirm(ui('恢复全部增强参数为默认值？当前素材和导出设置保留。')))void apply(DEFAULT_SETTINGS)}}><RotateCcw size={15}/>{ui("恢复增强默认值")}</button></div><p>{ui("队列任务在加入时固定参数快照；之后修改不会改写正在运行的任务。")}</p></section></div><p role="status" className="feedback">{ui(message)}</p></div>
  if(p.page==='about')return <div className="workspace-page about-page"><Brand/><span className="eyebrow">ORIGINAL LIGHT / MOTION IDENTITY</span><h1>Lunarc</h1><AppUpdatePanel/><p className="about-lead">{ui("调整画面、检查预览、导出视频。")}</p><p>{ui("基于 DLSSNR 与 GPU 视频链的视频增强工作台。")}</p><div className="settings-grid"><section className="panel"><h2>{ui("支持与验证")}</h2><p>{ui("SDR / HDR10 单层、双层、三层短 GPU 导出已验证。HDR10 与数据一致性融合、场景检测及静态低频稳定仍为实验性。")}</p><p>{ui("AMD FidelityFX 已接入原生设备路径；RAFT-Large 使用可配置长边的 CUDA 分析。NVOF 不提供本项目可用的置信度输出。")}</p></section><section className="panel"><h2>{ui("范围与限制")}</h2><p>{ui("预览为 8-bit 显示路径，未经 HDR 显示校准。抗闪烁只用于相似静态区域，不进行运动重投影；极端运动与闪光可能误判切镜。")}</p><p>{ui("复制的字幕事件尾部与音频 padding 可超过视频结尾。性能：NOT_CHARACTERIZED。")}</p></section></div><p>© 2026 Yukikaze1945 · <a href="https://github.com/Yukikaze1945/Lunarc-DLSS5/blob/main/LICENSE" target="_blank" rel="noreferrer">GPL-3.0-or-later</a>{ui(" · 可按许可证再分发，不提供担保。")}</p><button className="button" onClick={()=>p.onNavigate('diagnostics')}>{ui("查看当前机器诊断")}<ArrowUpRight size={15}/></button></div>
  if(p.page==='diagnostics')return <div className="workspace-page"><div className="page-title diagnostics-heading"><div><span className="eyebrow">{ui("OBSERVABILITY")}</span><h1>{ui("运行与诊断")}</h1><p>{ui("查看当前机器与任务状态。")}</p></div><button className="button" onClick={refresh}><RefreshCw size={15}/>{ui("刷新")}</button></div><div className="settings-grid"><section className="panel"><h2>{ui("当前机器")}</h2><dl className="fact-list"><dt>GPU</dt><dd>{ui(p.system ? (p.system.gpu || '未检测到可用 GPU') : '待连接')}</dd><dt>{ui("显存")}</dt><dd>{ui(vramLabel)}</dd><dt>{ui("后端")}</dt><dd>{ui(p.system?.backend_ready?'已就绪':'未连接')}</dd><dt>{ui("性能")}</dt><dd>NOT_CHARACTERIZED</dd></dl></section><section className="panel"><h2>{ui("任务实际链路")}</h2><select aria-label={ui("诊断任务")} value={selected} onChange={async e=>{setSelected(e.target.value);try{setDiagnostics(e.target.value?await fetchJobDiagnostics(e.target.value):null)}catch(err:any){setMessage(err.message)}}}><option value="">{ui("选择后端任务")}</option>{ui(jobs.map(j=><option value={j.id} key={j.id}>{j.name} · {ui(j.status)}</option>))}</select>{ui(diagnostics?<pre className="diagnostic-json">{JSON.stringify(diagnostics,null,2)}</pre>:<p>{ui("执行任务后检查实际 fusion mode、图层 reset、host image 边界与标量 readback。尚未执行不能视作已启用。")}</p>)}</section></div><GpuMemoryPanel memory={p.system?.gpu_memory}/><EnvironmentPanel/><DlssnrRuntimePanel/><section className="panel"><h2>{ui("光流合同")}</h2><p>{ui("NVOF：current → previous，输入像素单位；每层独立历史，HDR 使用独立 8-bit motion guidance。")}</p><p>{ui("AMD FidelityFX：原生引导，每层独立历史；重置后的前六帧返回零运动。可用状态见下方实时诊断。")}</p><details><summary>{ui("当前 provider 原始诊断")}</summary><pre className="diagnostic-json">{contracts?JSON.stringify(contracts,null,2):'尚未获取 provider 数据'}</pre></details></section><section className="panel log-panel"><div className="inline-form"><h2>{ui("实时日志")}</h2><input aria-label={ui("搜索日志")} placeholder={ui("搜索级别、模块或消息")} value={query} onChange={e=>setQuery(e.target.value)}/><button className="button" onClick={async()=>{try{await navigator.clipboard.writeText(logs.map(l=>`${l.time} ${l.level} ${l.tag} ${l.message}`).join('\n'));setMessage('日志已复制')}catch(e:any){setMessage(e.message)}}}><Copy size={15}/>{ui("复制")}</button></div><div className="log-lines">{logs.filter(l=>JSON.stringify(l).toLowerCase().includes(query.toLowerCase())).map((l,i)=><div key={i} className={l.level==='ERROR'?'error':''}><time>{l.time}</time><b>{l.level}</b><span>{l.message}</span></div>)}{!logs.length&&<p>暂无日志。</p>}</div></section><details className="panel"><summary>{ui("导出历史")}</summary>{ui(stored<any[]>('lunarc.history.v1',[]).filter(j=>j.status==='completed').map(j=><button className="button quiet" key={j.id} onClick={()=>window.dispatchEvent(new CustomEvent('lunarc:compare',{detail:jobComparison(j)}))}>{j.name}{ui(" · 查看对比")}</button>))}<pre className="diagnostic-json">{JSON.stringify(stored('lunarc.history.v1',[]),null,2)}</pre></details><p role="status" className="feedback">{ui(message)}</p></div>
  return null
}
