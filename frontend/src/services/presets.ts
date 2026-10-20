import {completeSettings,DEFAULT_SETTINGS} from './workspace.ts'
import {configuredRenderStages} from './renderPipeline.ts'
import {VSR_MODES} from './vsrModes.ts'
import type {DLSSSettings} from '../types'
import {localToken} from './api.ts'

export type Preset = {name:string; settings:DLSSSettings}
const OPTIONAL_DEFAULTS:Partial<DLSSSettings>={dlaa_preset:0,truehdr_enabled:false,
  truehdr_contrast:100,truehdr_saturation:100,truehdr_middle_gray:50,truehdr_peak_nits:650,truehdr_debanding:true,
  layer_settings:[],layer_execution:'sequential'}
export function presetSettings(raw:Partial<DLSSSettings>):DLSSSettings {
  const quality=raw.flow_quality||DEFAULT_SETTINGS.flow_quality
  const settings=completeSettings({...OPTIONAL_DEFAULTS,
    raft_analysis_edge:quality==='Fast'?512:quality==='Quality'?1080:720,
    raft_iterations:quality==='Fast'?4:quality==='Quality'?12:6,...raw})
  const disabled=Object.fromEntries(Object.entries(settings.render_stage_enabled??{}).filter(([,value])=>value===false))
  if(Object.keys(disabled).length)settings.render_stage_enabled=disabled
  else delete settings.render_stage_enabled
  if(settings.frame_generation!=='off')delete settings.frame_generation_saved_mode
  return {...settings,render_pipeline:configuredRenderStages(settings)}
}
export const DEFAULT_PRESET=presetSettings(DEFAULT_SETTINGS)
export function samePreset(a:Partial<DLSSSettings>,b:Partial<DLSSSettings>):boolean {
  const sorted=(value:any):any=>Array.isArray(value)?value.map(sorted):value&&typeof value==='object'
    ?Object.fromEntries(Object.keys(value).sort().map(key=>[key,sorted(value[key])])):value
  return JSON.stringify(sorted(presetSettings(a)))===JSON.stringify(sorted(presetSettings(b)))
}
const KEY='lunarc.presets.v1'
const MIGRATED='lunarc.presets.backend-migrated.v1'
let snapshot:Preset[]=[]
let cacheLoaded=false
let initialized=false, initializing:Promise<void>|null=null
let writeQueue=Promise.resolve()
const API='http://127.0.0.1:18923/api/presets'
function headers(json=false):Record<string,string>{const token=localToken();return {...(json?{'Content-Type':'application/json'}:{}),...(token?{'X-DLSS-Token':token}:{})}}
async function request(path='',method='GET',body?:unknown):Promise<{presets:Preset[];revision:number}>{
  const response=await fetch(`${API}${path}`,{method,headers:headers(body!==undefined),...(body===undefined?{}:{body:JSON.stringify(body)})})
  const data=await response.json().catch(()=>({}))
  if(!response.ok)throw new Error(data.detail||'预设同步失败')
  return data
}
function normalize(list:unknown):Preset[]{return Array.isArray(list)?list.filter((p:any)=>typeof p?.name==='string'&&p.name.trim()&&p.settings&&typeof p.settings==='object'&&!Array.isArray(p.settings)).map((p:any)=>({name:p.name,settings:presetSettings(p.settings)})):[]}
function setCache(list:Preset[],notify=true){snapshot=normalize(list);try{localStorage.setItem(KEY,JSON.stringify(snapshot))}catch{};if(notify)window.dispatchEvent(new Event('lunarc:presets'))}
function cached():Preset[]{try{return normalize(JSON.parse(localStorage.getItem(KEY)||'[]'))}catch{return []}}
async function initialize():Promise<void>{
  if(initialized)return
  if(initializing)return initializing
  initializing=(async()=>{
    const legacy=cached()
    const server=await request()
    let authoritative=normalize(server.presets)
    if(localStorage.getItem(MIGRATED)!=='1'){
      if(!localStorage.getItem('lunarc.presets.migration-backup.v1'))localStorage.setItem('lunarc.presets.migration-backup.v1',localStorage.getItem(KEY)||'[]')
      const merged=await request('/import','POST',{presets:legacy})
      authoritative=normalize(merged.presets)
      localStorage.setItem(MIGRATED,'1')
    }
    setCache(authoritative)
    initialized=true
  })().finally(()=>{initializing=null})
  return initializing
}
function enqueue<T>(operation:()=>Promise<T>):Promise<T>{const result=writeQueue.then(operation);writeQueue=result.then(()=>undefined,()=>undefined);return result}
function reportSyncError(error:unknown){window.dispatchEvent(new CustomEvent('lunarc:preset-error',{detail:{message:error instanceof Error?error.message:'预设同步失败'}}))}
export function initializePresets():Promise<void>{return initialize().catch(error=>{reportSyncError(error);throw error})}
export function presetSnapshot():Preset[] {
  if(!cacheLoaded){cacheLoaded=true;snapshot=cached()}
  return snapshot
}
export function subscribePresets(listener:()=>void) {
  void initialize().catch(reportSyncError)
  const focus=()=>{void refreshPresets().catch(reportSyncError)}
  window.addEventListener('lunarc:presets',listener);window.addEventListener('storage',listener)
  window.addEventListener('focus',focus)
  return()=>{window.removeEventListener('lunarc:presets',listener);window.removeEventListener('storage',listener);window.removeEventListener('focus',focus)}
}
export async function refreshPresets():Promise<void>{
  await enqueue(async()=>{await initialize();const data=await request();setCache(data.presets)})
}
export function writePresets(value:Preset[]):Promise<void>{
  return enqueue(async()=>{
    await initialize()
    const before=snapshot, after=normalize(value)
    const nextNames=new Set(after.map(p=>p.name))
    for(const old of before)if(!nextNames.has(old.name))await request('/delete','POST',{name:old.name})
    for(const preset of after){const existing=before.find(p=>p.name===preset.name)
      if(!existing||JSON.stringify(existing.settings)!==JSON.stringify(preset.settings))
        await request('/save','POST',{name:preset.name,settings:preset.settings,overwrite:Boolean(existing)})
    }
    const data=await request();setCache(data.presets)
  }).catch(error=>{reportSyncError(error);throw error})
}
export async function savePreset(name:string,settings:DLSSSettings):Promise<void> {
  name=name.trim();if(!name||name.length>80)throw new Error('请输入 1–80 字的预设名称')
  await enqueue(async()=>{
    await initialize()
    const existing=snapshot.some(p=>p.name===name)
    const data=await request('/save','POST',{name,settings:presetSettings(settings),overwrite:existing})
    setCache(data.presets)
  }).catch(error=>{reportSyncError(error);throw error})
}
export async function deletePreset(name:string):Promise<void>{
  await enqueue(async()=>{
    await initialize()
    const data=await request('/delete','POST',{name})
    setCache(data.presets)
  }).catch(error=>{reportSyncError(error);throw error})
}
export async function importPresets(presets:Preset[]):Promise<void>{
  await enqueue(async()=>{
    await initialize()
    const data=await request('/import','POST',{presets:normalize(presets)})
    setCache(data.presets)
  }).catch(error=>{reportSyncError(error);throw error})
}

const labels:Record<string,string>={preset:'NR 模型',render_pipeline:'渲染顺序',style:'画面风格',intensity:'NR 强度',local_tone:'局部色调强度',
  local_struct:'局部结构强度',skin_struct:'皮肤结构强度',use_auto_mask:'自动遮罩',ui_correction:'NR 界面修正',
  guidance_mode:'光流引导',flow_engine:'光流引擎',flow_quality:'光流质量',working_scale:'工作缩放',
  reconstruction_mode:'重建层数',layer_execution:'图层执行方式',layer_scales:'图层缩放',layer_weights:'图层权重',
  fusion_mode:'融合方式',fusion_beta:'融合 β',fusion_max_iterations:'求解迭代上限',fusion_relative_tolerance:'求解误差阈值',
  residual_strength:'残差倍率',residual_pyramid:'保留源细节',motion_scale_x:'运动 X',motion_scale_y:'运动 Y',
  dlaa_preset:'DLAA 模型',vsr_quality:'VSR 档位',vsr_mode:'VSR 模式',vsr_strength:'VSR 效果强度',vsr_output_mode:'VSR 输出方式',vsr_target:'VSR 输出分辨率',
  vsr_scale:'VSR 放大倍率',vsr_width:'VSR 输出宽度',vsr_height:'VSR 输出高度',frame_generation:'插帧模式',
  frame_multiplier:'插帧倍率',framegen_target_fps:'插帧目标帧率',framegen_quality:'插帧质量',framegen_scene_detection:'镜头切换检测',
  truehdr_enabled:'TrueHDR',truehdr_contrast:'TrueHDR 对比度',truehdr_saturation:'TrueHDR 饱和度',
  truehdr_middle_gray:'TrueHDR 中灰亮度',truehdr_peak_nits:'TrueHDR 目标峰值',truehdr_debanding:'AI 去色带',
  total_gain:'总强度',lightness_gain:'明度总强度',chroma_gain:'色度强度',low_frequency_gain:'低频强度',
  high_frequency_gain:'高频强度',hue_protection:'色相保护',dark_protection:'暗部保护',highlight_protection:'亮部保护',
  soft_compression:'过度修正抑制',roi_kind:'区域保护',roi_bounds:'保护区域边界',roi_strength:'区域保护强度',
  mode:'时域稳定方式',strength:'时域稳定强度',scene_detection:'镜头切换检测',scene_threshold:'镜头切换阈值',
  raft_analysis_edge:'RAFT 分析长边',raft_iterations:'RAFT 迭代次数',export_backend:'导出后端',
  experimental_hdr_scrgb:'HDR 导出',hdr_paper_white_scrgb:'HDR 纸白',dlss_dll_path:'DLAA 运行库',vsr_runtime_path:'VFX 运行库'}
const words:Record<string,string>={nr:'DLSSNR',dlaa:'DLAA',vsr:'RTX VSR',vfg:'NVIDIA VFG（固定）',truehdr:'TrueHDR',
  vfg_dynamic:'NVIDIA VFG（动态）',off:'关闭',single:'单层',dual:'双层',triple:'三层',sequential:'顺序堆叠',
  catmull_baseline:'仅第一层',data_consistency_experimental:'多层融合',scale:'按倍率',resolution:'按分辨率',
  nvof:'NVIDIA 光流',fidelityfx:'AMD 光流',raft:'RAFT',raft_large:'RAFT-Large',gpu_transfer:'GPU Transfer',auto:'自动',legacy:'Legacy'}
export function presetValue(key:string,value:unknown):string {
  if(key==='preset')return ({0:'运行库原有默认值',1:'A',2:'B',3:'旧预设 3'} as Record<string,string>)[String(value)]??String(value)
  if(value===undefined||value===null)return '默认'
  if(Array.isArray(value))return value.map(v=>presetValue(key,v)).join(' / ')
  if(typeof value==='boolean')return value?'开启':'关闭'
  if(key==='style')return ['默认','自然','电影'][Number(value)]??String(value)
  if(key==='vsr_quality')return ['低','中','高','极高'][Number(value)-1]??String(value)
  if(key==='vsr_mode')return VSR_MODES.find(([mode])=>mode===value)?.[1]??String(value)
  if(key==='vsr_strength')return `${Math.round(Number(value)*100)}%`
  if(key==='framegen_quality')return ['低','中','高'][Number(value)]??String(value)
  if(['use_auto_mask','ui_correction','guidance_mode'].includes(key))return value?'开启':'关闭'
  if(typeof value==='number')return Number(value.toFixed(4)).toString()
  return words[String(value)]??String(value)
}
export type PresetChange={key:string;label:string;before:string;after:string;delta?:number;percent?:number}
export function presetChanges(settings:DLSSSettings):PresetChange[] {
  const baseline=DEFAULT_PRESET
  const current=presetSettings(settings)
  const changes:PresetChange[]=[]
  const visit=(value:any,base:any,path:string)=>{
    if(Array.isArray(value)&&['layer_scales','layer_weights'].includes(path)){
      value.forEach((entry,index)=>visit(entry,base?.[index]??base?.[0],`第 ${index+1} 层.${path}`))
      return
    }
    if(Array.isArray(value)&&path==='post_controls.roi_bounds'){
      value.forEach((entry,index)=>visit(entry,base?.[index],`${path}.${['左边界','上边界','右边界','下边界'][index]}`))
      return
    }
    if(value&&typeof value==='object'&&!Array.isArray(value)){
      for(const key of Object.keys(value))visit(value[key],base?.[key],path?`${path}.${key}`:key)
      return
    }
    if(JSON.stringify(value)===JSON.stringify(base))return
    const key=path.split('.').pop()!
    if(path.startsWith('render_stage_enabled.')) {
      if(value===true)return
      const stageNames:Record<string,string>={nr:'DLSSNR',dlaa:'DLAA',vsr:'RTX VSR',vfg:'NVIDIA VFG',truehdr:'TrueHDR'}
      changes.push({key:path,label:`${stageNames[key]} 开关`,before:'启用',after:'停用'})
      return
    }
    if(key==='frame_generation_saved_mode')return
    if(key==='neural_mode'||key==='dlss_backend'||key.startsWith('_'))return
    const delta=typeof value==='number'&&typeof base==='number'&&
      !['preset','style','use_auto_mask','ui_correction','guidance_mode','dlaa_preset','vsr_quality','framegen_quality','roi_kind','debug_mode'].includes(key)?value-base:undefined
    changes.push({key:path,label:labels[key]??key,before:presetValue(key,base),after:presetValue(key,value),delta,
      percent:delta!==undefined&&base!==0?delta/Math.abs(base)*100:undefined})
  }
  for(const [key,value] of Object.entries(current)){
    if(key==='layer_settings'){
      const nr={preset:1,style:0,intensity:1,local_tone:1,local_struct:1,skin_struct:1,use_auto_mask:1,ui_correction:0}
      ;(value as Record<string,number>[]|undefined)?.forEach((layer,index)=>{if(index)visit(layer,nr,`第 ${index+1} 层`)})
    }else visit(value,(baseline as any)[key],key)
  }
  return changes.map(item=>({...item,label:item.key.startsWith('第 ')?`${item.key.split('.')[0]} · ${item.label}`:item.label}))
}
