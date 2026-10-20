const shown=new Set<string>()
type NoticeStorage={getItem:(key:string)=>string|null;setItem:(key:string,value:string)=>void}

/** Inform once per file identity; never ask for consent or block activation. */
export function runtimeIdentityNotice(value:unknown, storage?:NoticeStorage):string|null {
  if(!value||typeof value!=='object')return null
  const record=value as {id?:unknown;trust?:unknown}
  if(record.trust!=='unrecognized'||typeof record.id!=='string'||!/^[a-f\d]{64}$/i.test(record.id))return null
  const key=`lunarc.runtime.identity-notice.v1:${record.id.toLowerCase()}`
  if(shown.has(key))return null
  try{
    const target=storage??(typeof localStorage!=='undefined'?localStorage:undefined)
    if(target?.getItem(key)==='1'){shown.add(key);return null}
    target?.setItem(key,'1')
  }catch{/* Storage restrictions do not block a user-provided runtime. */}
  shown.add(key)
  return '未识别身份 DLL，已导入使用。'
}
