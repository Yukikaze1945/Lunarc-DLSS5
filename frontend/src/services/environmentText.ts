import {translateText, type Locale} from '../i18n/translate'

/** Localize composed dependency lists without translating paths or raw errors. */
export function environmentText(value:string,locale:Locale):string {
  if(locale!=='zh-CN'&&value.startsWith('缺少：')){
    const labels=value.slice('缺少：'.length).split(/[、,]/).map(label=>translateText(label.trim(),locale))
    return translateText('缺少：{0}',locale).replace('{0}',new Intl.ListFormat(locale,{type:'conjunction'}).format(labels))
  }
  return translateText(value,locale)
}
