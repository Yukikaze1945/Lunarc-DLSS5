import type {VideoInfo} from '../types'

export const PREVIEW_LENGTHS = [
  {id:'f5',label:'5 帧',frames:5}, {id:'f10',label:'10 帧',frames:10},
  {id:'s0.5',label:'0.5 秒',seconds:.5}, {id:'s1',label:'1 秒',seconds:1},
  {id:'s2',label:'2 秒',seconds:2}, {id:'s3',label:'3 秒',seconds:3},
  {id:'s5',label:'5 秒',seconds:5},
] as const
export type PreviewLength = typeof PREVIEW_LENGTHS[number]['id']
export type PreviewMode = 'realtime'|'prerender'|'single'
export type PreviewRequest = {id:number;source:string;start_frame:number;end_frame:number}

export function previewFrameRange(video:Pick<VideoInfo,'frames'|'fps'>,frame:number,length:PreviewLength){
  const total=Math.max(1,Math.floor(video.frames||1))
  const start=Math.max(0,Math.min(total-1,Math.floor(frame)))
  const option=PREVIEW_LENGTHS.find(option=>option.id===length)??PREVIEW_LENGTHS[0]
  const count='frames' in option?option.frames:Math.ceil(option.seconds*(video.fps>0?video.fps:30))
  return {start_frame:start,end_frame:Math.min(total,start+count)}
}
