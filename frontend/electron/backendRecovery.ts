export type NativeOperation = {id:string; phase:string; remaining_ms:number; timed_out:boolean}
export type BackendHealth = {service:string; pid:number; healthy:boolean; incident_id:string|null; reason:string|null; queue_storage_error?:string|null;
  recovery_checkpoint_ready?:boolean; native_operation?:NativeOperation|null}
export type RecoveryState = {phase:'recovering'|'restored'|'blocked'; message:string}

// Recovery is owned by Electron, so the failed native worker cannot supervise itself.
export class BackendRecovery {
  private busy=false
  private lastAttempt:number|null=null
  private incidents=new Set<string>()
  private stopped=false
  private restart:()=>Promise<void>
  private notify:(state:RecoveryState)=>void
  private clock:()=>number
  constructor(restart:()=>Promise<void>, notify:(state:RecoveryState)=>void,clock:()=>number=Date.now) {
    this.restart=restart;this.notify=notify;this.clock=clock
  }
  stop(){this.stopped=true}
  async recover(incident:string, reason:string) {
    if(this.stopped||this.busy||this.incidents.has(incident))return
    this.incidents.add(incident)
    if(this.incidents.size>100)this.incidents.delete(this.incidents.values().next().value!)
    const now=this.clock()
    if(this.lastAttempt!==null&&now-this.lastAttempt<60000){
      this.notify({phase:'blocked',message:'GPU 恢复后再次失败，已停止自动重启；请调整处理链后重试'})
      return
    }
    this.busy=true;this.lastAttempt=now
    this.notify({phase:'recovering',message:`${reason}，正在恢复后端…`})
    try{
      await this.restart()
      if(!this.stopped)this.notify({phase:'restored',message:'后端已恢复；中断的导出可手动重试'})
    }catch(error){
      if(!this.stopped)this.notify({phase:'blocked',message:`后端恢复失败：${(error as Error).message}`})
    }finally{this.busy=false}
  }
}

export function mayRestartBackend(health:BackendHealth, ownedPid:number|null):boolean {
  return health.service==='lunarc' && ownedPid!==null && health.pid===ownedPid &&
    !health.healthy && !!health.incident_id && !health.queue_storage_error && health.recovery_checkpoint_ready!==false
}

// A missing HTTP response alone is insufficient evidence. Keep only the latest
// owned operation whose queue checkpoint was confirmed before the native call.
export class BackendLiveness {
  private operation:{pid:number; id:string; phase:string; expiresAt:number}|null=null
  private clock:()=>number
  constructor(clock:()=>number=()=>performance.now()) {this.clock=clock}
  observe(health:BackendHealth, ownedPid:number|null) {
    this.operation=null
    const op=health.native_operation
    if(health.service!=='lunarc'||ownedPid===null||health.pid!==ownedPid||!health.healthy||
       health.queue_storage_error||health.recovery_checkpoint_ready!==true||!op||
       typeof op.id!=='string'||typeof op.phase!=='string'||
       !Number.isFinite(op.remaining_ms)||op.remaining_ms<0||op.remaining_ms>600000)return
    this.operation={pid:ownedPid,id:op.id,phase:op.phase,expiresAt:this.clock()+op.remaining_ms+15000}
  }
  unresponsive(ownedPid:number|null) {
    const op=this.operation
    if(!op||op.pid!==ownedPid||this.clock()<op.expiresAt)return null
    return {incident:`hang:${op.pid}:${op.id}`,reason:`GPU 操作无响应：${op.phase}`}
  }
}
