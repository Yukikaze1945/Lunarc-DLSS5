// localized-render-sinks
import {ui, useLocale} from '../i18n/locale'
import {useEffect, useState} from 'react'
import {RefreshCw} from 'lucide-react'
import type {EnvironmentReport} from '../types'
import {fetchEnvironment} from '../services/api'
import {environmentText} from '../services/environmentText'

const labels = {ok: '检查通过', pending: '依赖齐全', missing: '缺少组件', error: '检测失败'}
export function EnvironmentPanel() {
  const locale = useLocale()
  const [report, setReport] = useState<EnvironmentReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [startup, setStartup] = useState<{message: string; logPath?: string} | null>(null)
  async function check(refresh: boolean) {
    setBusy(true); setError('')
    try {setReport(await fetchEnvironment(refresh)); setStartup(null)}
    catch (e: any) {
      setError(e.message)
      try {setStartup(await window.electronAPI?.getBackendStartup?.() || null)} catch {}
    } finally {setBusy(false)}
  }
  useEffect(() => {void check(false)}, [])
  return <section className="panel environment-panel">
    <div className="environment-heading"><h2>{ui("运行环境")}</h2><button className="button" disabled={busy} onClick={() => void check(true)}><RefreshCw size={15}/>{ui(busy ? '检测中…' : '重新检测')}</button></div>
    {ui(error && <p role="alert">{ui(startup?.message || error)}{ui(startup?.logPath && <><br/>{ui("日志：")}{ui(startup.logPath)}</>)}</p>)}
    {ui(report && <>
      <p>{ui(report.standalone && report.isolated ? '独立运行环境' : '开发运行环境')} · {ui(new Date(report.checked_at * 1000).toLocaleTimeString())}{ui(" 检测")}</p>
      <div className="environment-features">{ui(report.features.map(check => <div className="environment-row" key={check.id}><strong>{ui(check.label)}</strong><span className={`environment-status ${check.status}`}>{ui(labels[check.status])}</span><span>{environmentText(check.detail,locale)}</span></div>))}</div>
      <details><summary>{ui("组件详情")}</summary>{ui(report.checks.map(check => <div className="environment-check" key={check.id}><div><strong>{ui(check.label)}</strong><span className={`environment-status ${check.status}`}>{ui(labels[check.status])}</span></div><p>{environmentText(check.detail,locale)}</p>{ui(check.status !== 'ok' && check.action && <p>{ui(check.action)}</p>)}</div>))}</details>
    </>)}
    <p role="status">{ui(busy ? '正在检查驱动、CUDA、运行库与模型…' : report?.features.length ? '依赖检查不代替实际运行验证。' : '')}</p>
  </section>
}
