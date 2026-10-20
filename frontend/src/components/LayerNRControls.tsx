// localized-render-sinks
import {ui} from '../i18n/locale'
import type { CSSProperties } from 'react'
import type { DLSSSettings } from '../types'
import { layerNRSettings, type LayerNRKey } from '../services/multiScaleSettings'
import { ParameterHelp } from './ParameterHelp'
import {NRModelSelect} from './NRModelSelect'

export function LayerNRControls({settings, index, onChange, modelPresets=[1]}: {
  settings: DLSSSettings; index: number; modelPresets?:number[]; onChange: (key: LayerNRKey, value: number) => void
}) {
  const values = layerNRSettings(settings, index)
  return <div className="nr-layer-controls" aria-label={ui(`第 ${index+1} 层参数`)}>
    {index>0&&<NRModelSelect value={values.preset} available={modelPresets} label={`第 ${index+1} 层 NR 模型`} onChange={value=>onChange('preset',value)}/>}
    <div className="style-switch" role="group" aria-label={ui(`第 ${index+1} 层画面风格`)}>
      {ui([{value:0,label:'默认'},{value:2,label:'电影'},{value:1,label:'自然'}].map(item =>
        <button type="button" key={item.value} aria-pressed={values.style===item.value}
          onClick={()=>onChange('style', item.value)}>{ui(item.label)}</button>))}
    </div>
    {ui(([['intensity','NR 强度'],['local_tone','局部色调强度'],
      ['local_struct','局部结构强度'],['skin_struct','皮肤结构强度']] as const).map(([key,label]) =>
      <label className="primary-slider" key={key}><span>{ui(label)}<ParameterHelp name={key} label={label}/>
        <output>{ui(Number(values[key].toFixed(2)))}</output></span>
        <input aria-label={ui(`第 ${index+1} 层${ui(label)}`)} type="range" min={0} max={2} step={.05}
          value={values[key]} style={{'--range-fill':`${values[key]/2*100}%`} as CSSProperties}
          onChange={e=>onChange(key, Number(e.target.value))}/>
      </label>))}
    <label className="primary-check"><input type="checkbox" aria-label={ui(`第 ${index+1} 层自动遮罩`)}
      checked={values.use_auto_mask===1} onChange={e=>onChange('use_auto_mask',e.target.checked?1:0)}/>{ui("自动遮罩")}<ParameterHelp name="use_auto_mask" label="自动遮罩"/></label>
  </div>
}
