import {test} from 'node:test'
import assert from 'node:assert/strict'
import {runtimeIdentityNotice} from '../../frontend/src/services/runtimeNotice.ts'

test('unrecognized file informs once per hash without requiring consent',()=>{
  const values=new Map<string,string>()
  const storage={getItem:(key:string)=>values.get(key)??null,setItem:(key:string,value:string)=>{values.set(key,value)}}
  const record={id:'a'.repeat(64),trust:'unrecognized'}
  assert.equal(runtimeIdentityNotice(record,storage),'未识别身份 DLL，已导入使用。')
  assert.equal(runtimeIdentityNotice(record,storage),null)
  assert.equal(values.get(`lunarc.runtime.identity-notice.v1:${record.id}`),'1')
  assert.equal(runtimeIdentityNotice({...record,id:'b'.repeat(64)},storage),'未识别身份 DLL，已导入使用。')
})

test('stored notice survives reload and unavailable storage does not block use',()=>{
  assert.equal(runtimeIdentityNotice({id:'c'.repeat(64),trust:'unrecognized'},
    {getItem:()=> '1',setItem:()=>assert.fail('already shown')}),null)
  const broken={getItem:()=>{throw Error('disabled')},setItem:()=>{throw Error('disabled')}}
  const record={id:'d'.repeat(64),trust:'unrecognized'}
  assert.equal(runtimeIdentityNotice(record,broken),'未识别身份 DLL，已导入使用。')
  assert.equal(runtimeIdentityNotice(record,broken),null)
})

test('identified files and malformed responses need no notice',()=>{
  for(const record of [null,{},'bad',{id:'bad',trust:'unrecognized'},
    {id:'e'.repeat(64),trust:'nvidia_signature'},{id:'f'.repeat(64),trust:'pinned_community'}]){
    assert.equal(runtimeIdentityNotice(record),null)
  }
})
