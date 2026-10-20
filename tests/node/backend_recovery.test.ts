import {test} from 'node:test'
import assert from 'node:assert/strict'
import {BackendRecovery,BackendLiveness,mayRestartBackend} from '../../frontend/electron/backendRecovery.ts'

test('one incident restarts once, preserving the single child boundary',async()=>{
  let starts=0,unblock:()=>void=()=>{}
  const events:any[]=[]
  const recovery=new BackendRecovery(async()=>{starts++;await new Promise<void>(resolve=>{unblock=resolve})},e=>events.push(e))
  const running=recovery.recover('first','device removed')
  await recovery.recover('first','device removed');await recovery.recover('other','device removed')
  assert.equal(starts,1);unblock();await running
  assert.deepEqual(events.map(e=>e.phase),['recovering','restored'])
})
test('repeated fault is blocked within the recovery window',async()=>{
  let now=0,starts=0;const events:any[]=[]
  const recovery=new BackendRecovery(async()=>{starts++},e=>events.push(e),()=>now)
  await recovery.recover('a','reset');now=5000;await recovery.recover('b','reset')
  assert.equal(starts,1);assert.equal(events.at(-1).phase,'blocked')
  now=65000;await recovery.recover('c','reset');assert.equal(starts,2)
})
test('failed start and app shutdown never announce successful recovery',async()=>{
  const events:any[]=[]
  const recovery=new BackendRecovery(async()=>{throw Error('port occupied')},e=>events.push(e))
  await recovery.recover('a','reset');assert.equal(events.at(-1).phase,'blocked')
  recovery.stop();await recovery.recover('b','reset');assert.equal(events.length,2)
})
test('unowned process, healthy backend and failed checkpoint cannot be restarted',()=>{
  const health={service:'lunarc',pid:123,healthy:false,incident_id:'fault',reason:'reset'}
  assert(mayRestartBackend(health,123))
  assert(!mayRestartBackend(health,456));assert(!mayRestartBackend(health,null))
  assert(!mayRestartBackend({...health,healthy:true},123))
  assert(!mayRestartBackend({...health,queue_storage_error:'disk full'},123))
  assert(!mayRestartBackend({...health,recovery_checkpoint_ready:false},123))
})

test('unresponsive native owner recovers only after its observed deadline and grace',()=>{
  let now=0
  const liveness=new BackendLiveness(()=>now)
  const health={service:'lunarc',pid:123,healthy:true,incident_id:null,reason:null,
    recovery_checkpoint_ready:true,native_operation:{id:'one',phase:'GPU frame',remaining_ms:30000,timed_out:false}}
  assert.equal(liveness.unresponsive(123),null)
  liveness.observe(health,123)
  now=44999;assert.equal(liveness.unresponsive(123),null)
  now=45000;assert.equal(liveness.unresponsive(123)?.incident,'hang:123:one')
  assert.equal(liveness.unresponsive(456),null)
  liveness.observe({...health,native_operation:null},123)
  assert.equal(liveness.unresponsive(123),null)
})

test('healthy idle, external backend and failed queue checkpoint never imply a native hang',()=>{
  let now=0
  const liveness=new BackendLiveness(()=>now)
  const health={service:'lunarc',pid:123,healthy:true,incident_id:null,reason:null,
    recovery_checkpoint_ready:true,native_operation:{id:'one',phase:'GPU',remaining_ms:0,timed_out:false}}
  for(const invalid of [{...health,pid:456},{...health,recovery_checkpoint_ready:false},
    {...health,queue_storage_error:'disk full'},{...health,native_operation:null}]){
    liveness.observe(invalid,123);now+=100000
    assert.equal(liveness.unresponsive(123),null)
  }
})
