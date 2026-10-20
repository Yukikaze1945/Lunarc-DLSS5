// localized-render-sinks
import {ui,useLocale} from '../i18n/locale'
import React, { useEffect, useMemo, useState } from 'react'
import {foldSections} from './CollapsibleSection'
import { RotateCcw, Search, SlidersHorizontal } from 'lucide-react'
import type { DLSSSettings, PostControls, TemporalControls } from '../types'
import { saveSettings } from '../services/api'
import { DEFAULT_POST, DEFAULT_TEMPORAL } from '../services/qualityControls'
import {renderStages} from '../services/renderPipeline'
import './qualityInspector.css'
import { ParameterHelp } from './ParameterHelp'

type TabId = 'basic' | 'layers' | 'quality' | 'temporal' | 'hdr'
type SearchEntry = { label: string; keys: string; tab: TabId }

const TABS: { id: TabId; label: string }[] = [
  { id: 'basic', label: '基础' },
  { id: 'layers', label: '图层' },
  { id: 'quality', label: '画质' },
  { id: 'temporal', label: '时域' },
  { id: 'hdr', label: 'HDR' },
]

const SEARCH_ENTRIES: SearchEntry[] = [
  { label: 'NR 界面修正', keys: 'ui correction 界面 字幕 保护', tab: 'basic' },
  { label: '引导与光流', keys: 'guidance 引导 motion 运动 flow 光流 provider engine', tab: 'basic' },
  { label: '导出后端', keys: 'backend 后端 export 导出 gpu transfer legacy', tab: 'basic' },
  { label: '工作分辨率与残差', keys: 'scale 缩放 working 工作分辨率 residual 残差 pyramid 金字塔', tab: 'layers' },
  { label: '图层与融合', keys: 'layer 图层 scale 尺度 weight 权重 fusion 融合', tab: 'layers' },
  { label: '强度与频率', keys: 'gain 增益 强度 lightness 亮度 chroma 色度 low 低 high 高 frequency 频率 residual 残差', tab: 'quality' },
  { label: '保护与区域', keys: 'protection 保护 hue 色相 dark 暗部 highlight 高光 compression 压缩 roi 区域 feather 羽化 bounds 边界 debug 调试', tab: 'quality' },
  { label: 'HDR 与纸白', keys: 'hdr 纸白 scrgb paper white', tab: 'hdr' },
  { label: '场景与稳定', keys: 'scene 场景 detection 检测 stability 稳定 temporal 时域 history 历史 weight 权重', tab: 'temporal' },
]

interface QualityInspectorProps {
  settings: DLSSSettings
  onSettingsChange: (settings: DLSSSettings) => void
  onStatusChange?: (message: string) => void
  primary: React.ReactNode
  basic: React.ReactNode
  layers: React.ReactNode
  hdr: React.ReactNode
  image?: boolean
}

function Slider({ label, value, min, max, step, onChange, disabled = false, helpKey }:
  { label: string; value: number; min: number; max: number; step: number;
    onChange: (value: number) => void; disabled?: boolean; helpKey?: string }) {
  return (
    <label className={`qi-field${disabled ? ' qi-disabled' : ''}`}>
      <span className="qi-field-head"><span>{ui(label)}{ui(helpKey && <ParameterHelp name={helpKey} label={label} />)}</span><output>{ui(Number(value).toFixed(2))}</output></span>
      <input aria-label={ui(label)} type="range" min={min} max={max} step={step} value={value}
        disabled={disabled} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  )
}

export const QualityInspector: React.FC<QualityInspectorProps> = ({
  settings, onSettingsChange, onStatusChange, primary, basic, layers, hdr, image = false,
}) => {
  const locale = useLocale()
  const [tab, setTab] = useState<TabId>('basic')
  const [query, setQuery] = useState('')
  const post: PostControls = { ...DEFAULT_POST, ...settings.post_controls }
  const temporal: TemporalControls = { ...DEFAULT_TEMPORAL, ...settings.temporal_controls }
  const normalizedQuery = query.trim().toLowerCase()
  const hasNR=renderStages(settings).includes('nr')
  const availableTabs = TABS.filter(item=>
    !(image&&(item.id==='temporal'||item.id==='hdr'||(!hasNR&&item.id==='basic'))) &&
    !(!hasNR&&['layers','quality','temporal'].includes(item.id)))
  const entries = SEARCH_ENTRIES.filter(entry=>availableTabs.some(item=>item.id===entry.tab) &&
    !(image&&(entry.label==='引导与光流'||entry.label==='导出后端')))
  useEffect(() => {
    if (!availableTabs.some((item) => item.id === tab)) setTab(availableTabs[0]?.id||'basic')
  }, [image, tab,hasNR])
  const matches = (label: string, key = '') => !normalizedQuery ||
    `${ui(label)} ${label} ${key}`.toLowerCase().includes(normalizedQuery)
  const searchResults = useMemo(() => {
    if (!normalizedQuery) return []
    return entries.filter((entry) =>
      `${ui(entry.label)} ${entry.label} ${entry.keys}`.toLowerCase().includes(normalizedQuery))
  }, [normalizedQuery,locale,image,hasNR])

  const savePost = async (next: PostControls) => {
    const patch = { post_controls: next }
    onSettingsChange({ ...settings, ...patch })
    try {
      await saveSettings(patch)
      onStatusChange?.('画质控制已保存')
    } catch (error) {
      onSettingsChange(settings)
      onStatusChange?.(`画质设置被后端拒绝：${error instanceof Error ? error.message : String(error)}`)
    }
  }
  const saveTemporal = async (next: TemporalControls) => {
    const patch = { temporal_controls: next }
    onSettingsChange({ ...settings, ...patch })
    try {
      await saveSettings(patch)
      onStatusChange?.('时域控制已保存')
    } catch (error) {
      onSettingsChange(settings)
      onStatusChange?.(`时域设置被后端拒绝：${error instanceof Error ? error.message : String(error)}`)
    }
  }
  const changePost = (key: keyof PostControls, value: number | PostControls['roi_kind']) => {
    savePost({ ...post, [key]: value })
  }
  const changeBound = (index: number, value: number) => {
    const bounds: PostControls['roi_bounds'] = [...post.roi_bounds]
    bounds[index] = value
    savePost({ ...post, roi_bounds: bounds })
  }
  const resetControls = async () => {
    const patch = image ? { post_controls: DEFAULT_POST } : { post_controls: DEFAULT_POST, temporal_controls: DEFAULT_TEMPORAL }
    onSettingsChange({ ...settings, ...patch })
    try {
      await saveSettings(patch)
      onStatusChange?.(image ? '画质控制已重置为严格中性' : '画质与时域控制已重置为严格中性')
    } catch (error) {
      onSettingsChange(settings)
      onStatusChange?.(`重置被后端拒绝：${error instanceof Error ? error.message : String(error)}`)
    }
  }

  return (
    <aside className="quality-inspector" aria-label={ui("DLSS 参数检查器")}>
      <div className="qi-primary" role="group" aria-label={ui("基础控制")}>{ui(primary)}</div>
      {availableTabs.length>0&&<details className="qi-advanced">
        <summary>{ui("高级")}<span className="qi-advanced-sign" aria-hidden="true">＋</span></summary>
      <header className="qi-header">
        <button className="qi-reset" type="button" onClick={resetControls}>
          <RotateCcw size={14} aria-hidden="true" />{ui(image ? " 重置画质" : " 重置画质/时域")}</button>
        <label className="qi-search">
          <Search size={15} aria-hidden="true" />
          <input value={query} onChange={(event) => setQuery(event.target.value)}
            aria-label={ui("搜索设置")} placeholder={ui("搜索设置")} />
        </label>
        {ui(normalizedQuery && (
          <div className="qi-search-results" role="listbox" aria-label={ui("设置搜索结果")}>
            {ui(searchResults.length ? searchResults.map((result) => {
              const tabInfo = TABS.find((item) => item.id === result.tab)!
              return <button type="button" role="option" aria-selected={tab === result.tab}
                key={result.label} onClick={() => setTab(result.tab)}>
                <span>{ui(result.label)}</span><small>{ui(tabInfo.label)} →</small>
              </button>
            }) : <p>{ui("没有匹配的设置")}</p>)}
          </div>
        ))}
        <div className="qi-tabs" role="tablist" aria-label={ui("参数分区")}>
          {ui(availableTabs.map((item) => <button key={item.id} type="button" role="tab"
            id={`qi-tab-${item.id}`} aria-controls={`qi-panel-${item.id}`}
            aria-selected={tab === item.id} onClick={() => setTab(item.id)}>
            <span>{ui(item.label)}</span>
          </button>))}
        </div>
      </header>

      <div className="qi-panel" role="tabpanel" id={`qi-panel-${tab}`}
        aria-labelledby={`qi-tab-${tab}`} key={tab}>
        {ui(tab === 'basic' && foldSections(basic))}
        {ui(tab === 'layers' && foldSections(layers))}
        {ui(tab === 'hdr' && !image && foldSections(hdr))}
        {ui(tab === 'quality' && foldSections(
          <div className="qi-section-stack">
            <div className="qi-export-note"><SlidersHorizontal size={16} />
              <div><strong>{ui("预览与 GPU Transfer 导出")}</strong><span>{ui("控制作用于模型残差；中性值保持原结果。")}</span></div>
            </div>
            <section className="qi-section">
              <div className="qi-section-title"><h3>{ui("整体与频率")}</h3><span>{ui("严格中性 1.00")}</span></div>
              {ui(([
                ['total_gain', '总强度', 0, 2], ['lightness_gain', '明度总强度', 0, 2],
                ['chroma_gain', '色度强度', 0, 2], ['low_frequency_gain', '低频强度', 0, 2],
                ['high_frequency_gain', '高频强度', 0, 2],
              ] as const).map(([key, label, min, max]) => matches(label, key) &&
                <Slider key={key} label={label} helpKey={key} value={post[key]} min={min} max={max} step={0.01}
                  onChange={(value) => changePost(key, value)} />))}
            </section>
            <section className="qi-section">
              <div className="qi-section-title"><h3>{ui("保护与压缩")}</h3><span>{ui("范围 0–1")}</span></div>
              {ui(([
                ['hue_protection', '色相保护'], ['dark_protection', '暗部保护'],
                ['highlight_protection', '亮部保护'], ['soft_compression', '过度修正抑制'],
              ] as const).map(([key, label]) => matches(label, key) &&
                <Slider key={key} label={label} helpKey={key} value={post[key]} min={0} max={1} step={0.01}
                  onChange={(value) => changePost(key, value)} />))}
            </section>
            <section className="qi-section">
              <div className="qi-section-title"><h3>{ui("保护区域 ROI")}</h3><span>{ui("归一化坐标")}</span></div>
              {ui(matches('ROI 类型', 'roi_kind') && <label className="qi-control">
                <span>{ui("区域形状")}<ParameterHelp name="roi_kind" label="区域保护" /></span><select value={post.roi_kind}
                  onChange={(event) => changePost('roi_kind', Number(event.target.value) as PostControls['roi_kind'])}>
                  <option value={0}>{ui("关闭")}</option><option value={1}>{ui("矩形")}</option><option value={2}>{ui("椭圆")}</option>
                </select>
              </label>)}
              {ui(post.roi_kind !== 0 && <>
                <div className="qi-roi-grid">
                  {ui((['左', '上', '右', '下'] as const).map((label, index) => matches(`ROI ${label}`, 'roi_bounds') &&
                    <label key={label} className="qi-control"><span>{ui(label)}</span>
                      <input type="number" min={0} max={1} step={0.01} value={post.roi_bounds[index]}
                        aria-label={ui(`ROI ${label} 边界`)} onChange={(event) => changeBound(index, Number(event.target.value))} />
                    </label>))}
                </div>
                {ui(matches('ROI 羽化', 'roi_feather') && <Slider label="向内羽化" helpKey="roi_feather" value={post.roi_feather}
                  min={0} max={0.25} step={0.005} onChange={(value) => changePost('roi_feather', value)} />)}
              </>)}
              {ui(matches('调试视图', 'debug_mode') && <label className="qi-control">
                <span>{ui("调试视图")}<ParameterHelp name="debug_mode" label="调试视图" /></span><select value={post.debug_mode}
                  onChange={(event) => changePost('debug_mode', Number(event.target.value) as PostControls['debug_mode'])}>
                  <option value={0}>{ui("关闭")}</option><option value={1}>{ui("原始残差")}</option>
                  <option value={2}>{ui("受控残差")}</option><option value={3}>{ui("保护/保留权重")}</option>
                </select>
              </label>)}
              {ui(matches('调试说明', 'debug tooltip raw controlled protection weight') && <p className="qi-footnote">{ui("原始残差显示模型修正；受控残差显示增益与保护后的残差；保护/保留权重显示作用于残差的权重遮罩。")}</p>)}
            </section>
          </div>
        ))}
        {ui(tab === 'temporal' && foldSections(
          <div className="qi-section-stack">
            <div className="qi-export-note"><SlidersHorizontal size={16} />
              <div><strong>{ui("预览与 GPU Transfer 导出")}</strong><span>{ui("静态区域低频累积；没有光流重投影。")}</span></div>
            </div>
            <section className="qi-section">
              <div className="qi-section-title"><h3>{ui("场景连续性")}</h3><span>{ui("实验性采样检测")}</span></div>
              {ui(matches('场景切换检测', 'scene_detection') && <label className="qi-toggle">
                <span><strong>{ui("场景切换检测")}<ParameterHelp name="scene_detection" label="场景切换检测" /></strong></span>
                <input type="checkbox" checked={temporal.scene_detection}
                  onChange={(event) => saveTemporal({ ...temporal, scene_detection: event.target.checked })} />
              </label>)}
            </section>
            <section className="qi-section">
              <div className="qi-section-title"><h3>{ui("残差稳定")}</h3><span>{ui("默认关闭")}</span></div>
              {ui(matches('时域模式', 'temporal mode stability') && <label className="qi-control">
                <span>{ui("抗闪烁")}<ParameterHelp name="temporal" label="抗闪烁" /></span><select value={temporal.mode}
                  onChange={(event) => saveTemporal({ ...temporal, mode: event.target.value as TemporalControls['mode'],
                    scene_detection: event.target.value === 'off' ? true : temporal.scene_detection })}>
                  <option value="off">{ui(temporal.scene_detection ? '仅镜头检测' : '关闭')}</option><option value="static_low_frequency">{ui("静态低频累积")}</option>
                </select>
              </label>)}
              {ui(matches('历史权重', 'history_weight') && <Slider label="历史权重" helpKey="history_weight" value={temporal.history_weight}
                min={0} max={0.9} step={0.01} disabled={temporal.mode === 'off'}
                onChange={(value) => saveTemporal({ ...temporal, history_weight: value })} />)}

            </section>
          </div>
        ))}
      </div>
      </details>}
    </aside>
  )
}
