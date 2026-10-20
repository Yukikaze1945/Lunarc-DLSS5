import {ui} from '../i18n/locale'

export function NRModelSelect({value, onChange, available=[1], label='NR 模型'}: {
  value:number; onChange:(value:number)=>void; available?:number[]; label?:string
}) {
  // Preserve legacy hints until edited. Runtime-default (0) selects B only on
  // a recognized A/B runtime; the old single-network runtime resolves to A.
  const displayed=value===0 ? (available.includes(2)?2:1) : value
  return <label>{ui(label)}<select aria-label={ui(label)} value={displayed}
    ref={element=>{if(element&&displayed!==1&&displayed!==2)element.selectedIndex=-1}}
    title={ui(value===3?'当前保留旧预设 3，选择 A 或 B 后应用新模型':'模型 B 需要已识别的 310.9.16 运行库')}
    onChange={event=>onChange(Number(event.target.value))}>
    <option value={1} disabled={!available.includes(1)}>A</option>
    <option value={2} disabled={!available.includes(2)}>B</option>
  </select></label>
}
