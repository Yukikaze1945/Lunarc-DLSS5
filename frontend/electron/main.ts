import { app, BrowserWindow, dialog, ipcMain, protocol, net, shell } from 'electron'
import crypto from 'node:crypto'
import path from 'node:path'
import http from 'node:http'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { spawn, execFileSync, ChildProcess } from 'node:child_process'
import fs from 'node:fs'
import {isLanguagePreference, resolveLocale, translateText, type LanguagePreference} from '../src/i18n/translate'
import {runtimeCandidates, runtimeDirectory, runtimeFingerprint, runtimeEnvironment, probeRuntime, type BackendStartup} from './backendRuntime'
import {AppUpdates} from './appUpdates'
import {ComponentManager} from './componentManager'
import {BackendRecovery, BackendLiveness, mayRestartBackend, type BackendHealth} from './backendRecovery'
import {resolveAppProtocolPath} from './appProtocolPath'

protocol.registerSchemesAsPrivileged([
  {
    scheme: 'app',
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true,
    },
  },
])

const __dirname = path.dirname(fileURLToPath(import.meta.url))

process.env.APP_ROOT = path.join(__dirname, '..')

export const VITE_DEV_SERVER_URL = process.env['VITE_DEV_SERVER_URL']
export const MAIN_DIST = path.join(process.env.APP_ROOT, 'dist-electron')
export const RENDERER_DIST = path.join(process.env.APP_ROOT, 'dist')

process.env.VITE_PUBLIC = VITE_DEV_SERVER_URL ? path.join(process.env.APP_ROOT, 'public') : RENDERER_DIST

try {
  if (process.env.E2E_TEST !== '1') {
    // Keep the existing storage path to preserve Chromium/localStorage and user configuration.
    // New Lunarc drafts use explicit v1 keys and default only newly added fields.
    app.setPath('userData', path.join(app.getPath('appData'), 'DaVinci_DLSS5_Studio'))
  }
} catch {
  // Ignore
}

let win: BrowserWindow | null = null
let language: LanguagePreference = 'auto'
try {
  const saved = JSON.parse(fs.readFileSync(path.join(app.getPath('userData'), 'ui-language.json'), 'utf8'))
  if (isLanguagePreference(saved.language)) language = saved.language
} catch { /* Follow the system when no valid UI preference exists. */ }
const nativeText = (text: string) => translateText(text, resolveLocale(language, app.getPreferredSystemLanguages()))
ipcMain.handle('ui:language:get', event =>
  event.sender === win?.webContents && event.senderFrame === win?.webContents.mainFrame ? {preference:language,systemLanguages:app.getPreferredSystemLanguages()} : null)
ipcMain.handle('ui:language:set', (event, value: unknown) => {
  if (event.sender !== win?.webContents || event.senderFrame !== win?.webContents.mainFrame || !isLanguagePreference(value)) return
  language = value
  fs.mkdirSync(app.getPath('userData'), {recursive: true})
  fs.writeFileSync(path.join(app.getPath('userData'), 'ui-language.json'), JSON.stringify({language}), 'utf8')
})
let updates: AppUpdates | null = null
function appUpdates() {
  if (!updates) updates = new AppUpdates(app.getVersion(), app.isPackaged, path.join(app.getPath('userData'), 'updates'), state => {
    if (win && !win.isDestroyed()) win.webContents.send('updates:state', state)
  }, net.fetch.bind(net) as typeof fetch)
  return updates
}
let components: ComponentManager | null = null
function componentManager() {
  const enabled=app.isPackaged || (process.env.E2E_TEST==='1' && process.env.E2E_COMPONENTS==='1')
  if(!enabled)return null
  if(!components)components=new ComponentManager(path.resolve(process.env.APP_ROOT,'..'),
    path.join(app.getPath('userData'),'components'),state=>win?.webContents.send('components:state',state),net.fetch.bind(net) as typeof fetch)
  return components
}
let starting: Promise<boolean>|null=null
async function ensureBackend() {
  if(starting)return starting
  starting=(async()=>{
    if(componentManager() && !componentManager()!.readyForBackend())return false
    const started=await startPythonBackend()
    for(let i=0;started && i<120 && backendStartup.state!=='error';i++) {
      if(await checkServerListening(backendPort)){backendStartup={...backendStartup,state:'ready',message:'后端已连接'};return true}
      await new Promise(r=>setTimeout(r,500))
    }
    if(backendStartup.state==='starting')backendStartup={...backendStartup,state:'error',message:'后端启动超时，请查看日志'}
    return backendStartup.state==='ready'
  })()
  try{return await starting}finally{starting=null}
}
let pyProc: ChildProcess | null = null
let backendStartup: BackendStartup = {state: 'checking', message: '正在检查运行环境'}
const backendPort = process.env.E2E_TEST === '1' ? Number(process.env.E2E_BACKEND_PORT || 18923) : 18923
let shuttingDown=false, recoveryTimer:ReturnType<typeof setInterval>|null=null, healthBusy=false, blockedIncident=''
const backendLiveness=new BackendLiveness()
const recovery=new BackendRecovery(async()=>{
  await stopOwnedBackend()
  if(shuttingDown)throw new Error('工作台正在关闭')
  if(!await ensureBackend())throw new Error(backendStartup.message)
}, state=>{
  if(!win?.isDestroyed())win?.webContents.send('backend-recovery',state)
})

function readBackendHealth():Promise<BackendHealth> {
  return new Promise((resolve,reject)=>{
    const request=http.get(`http://127.0.0.1:${backendPort}/api/backend/health`,
      {headers:{'X-DLSS-Token':process.env.DLSS_LOCAL_TOKEN||''}},response=>{
        let data='';response.on('data',chunk=>{data+=chunk;if(data.length>16384)request.destroy(new Error('后端健康报告过大'))})
        response.on('end',()=>{try{if(response.statusCode!==200)throw new Error(`健康检查 ${response.statusCode}`);resolve(JSON.parse(data))}catch(error){reject(error)}})
        response.on('error',reject)
      })
    request.on('error',reject);request.setTimeout(2000,()=>request.destroy(new Error('后端健康检查超时')))
  })
}

async function stopOwnedBackend(){
  const owned=pyProc
  if(!owned)return
  const exited=new Promise<void>((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('旧后端未能退出，已停止恢复')),12000)
    owned.once('exit',()=>{clearTimeout(timer);resolve()})
    if(owned.exitCode!==null||owned.signalCode!==null){clearTimeout(timer);resolve()}
  })
  killPythonBackend()
  await exited
  // Reusing a listening port here could reconnect to the failed process or another app.
  if(await checkServerListening(backendPort))throw new Error('旧后端端口仍被占用')
}

function startRecoveryMonitor(){
  recoveryTimer=setInterval(async()=>{
    if(shuttingDown||healthBusy||backendStartup.state!=='ready')return
    healthBusy=true
    try{
      const health=await readBackendHealth()
      backendLiveness.observe(health,pyProc?.pid||null)
      if(!health.healthy&&health.incident_id&&health.queue_storage_error&&health.pid===pyProc?.pid&&blockedIncident!==health.incident_id){
        blockedIncident=health.incident_id
        win?.webContents.send('backend-recovery',{phase:'blocked',message:`队列存档失败，已停止自动重启：${health.queue_storage_error}`})
      }
      if(mayRestartBackend(health,pyProc?.pid||null))
        await recovery.recover(health.incident_id!,health.reason||'GPU 处理失败')
    }catch{
      const stalled=backendLiveness.unresponsive(pyProc?.pid||null)
      if(stalled)await recovery.recover(stalled.incident,stalled.reason)
    }
    finally{healthBusy=false}
  },2000)
}

function checkServerListening(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/api/system`, (res) => {
      resolve(res.statusCode === 200)
    })
    req.on('error', () => resolve(false))
    req.setTimeout(500, () => {
      req.destroy()
      resolve(false)
    })
  })
}

async function resolveExistingBackendToken(rootDir: string): Promise<string | null> {
  const fs = await import('node:fs')
  const os = await import('node:os')
  let userDataDir = ''
  try {
    userDataDir = app.getPath('userData')
  } catch {}

  const candidateFiles = [
    path.join(rootDir, '.session_token'),
    userDataDir ? path.join(userDataDir, '.session_token') : '',
    path.join(os.homedir(), 'AppData', 'Roaming', 'DaVinci_DLSS5_Studio', '.session_token'),
    path.join(os.homedir(), '.dlss5', 'session_token'),
  ].filter(Boolean)

  for (const f of candidateFiles) {
    try {
      if (fs.existsSync(f)) {
        const t = fs.readFileSync(f, 'utf-8').trim()
        if (t) return t
      }
    } catch {}
  }
  return null
}

async function startPythonBackend() {
  const rootDir = path.resolve(process.env.APP_ROOT, '..')
  const manager=componentManager()
  const runtimeDir = manager?.runtime() || runtimeDirectory(rootDir, app.isPackaged)
  const standalone = app.isPackaged || fs.existsSync(path.join(runtimeDir, 'manifest.json'))
  const isRunning = await checkServerListening(backendPort)
  if (isRunning) {
    if (standalone) {
      const expected = runtimeFingerprint(rootDir, runtimeDir)
      const matches = await new Promise<boolean>(resolve => {
        const req = http.get(`http://127.0.0.1:${backendPort}/api/system`, res => {
          let data = ''; res.on('data', chunk => {data += chunk; if (data.length > 65536) req.destroy()})
          res.on('end', () => {try {const info = JSON.parse(data); resolve(!!expected && info.service === 'lunarc' && info.runtime_id === expected)} catch {resolve(false)}})
        })
        req.on('error', () => resolve(false)); req.setTimeout(2000, () => {req.destroy(); resolve(false)})
      })
      if (!matches) {
        backendStartup = {state: 'error', message: '另一个版本的工作台正在运行。请先退出旧版，再启动此版本。'}
        return false
      }
    }
    console.log('Python backend server is already running on port 18923.')
    if (!process.env.DLSS_LOCAL_TOKEN) {
      const existingToken = await resolveExistingBackendToken(rootDir)
      if (existingToken) {
        process.env.DLSS_LOCAL_TOKEN = existingToken
        console.log('[Backend] Successfully negotiated token with existing backend.')
      } else {
        console.warn('[Backend] Existing backend detected but no .session_token found!')
      }
    }
    backendStartup = {state: 'ready', message: '已连接本机后端'}
    return true
  }

  const serverScript = path.join(rootDir, 'backend_entry.py')
  // Per-launch session token for the local backend (P0-01): generated here,
  // handed to the backend via env, and exposed to the renderer via preload.
  // An explicit DLSS_LOCAL_TOKEN env (tests, manual runs) always wins.
  const localToken = process.env.DLSS_LOCAL_TOKEN || crypto.randomUUID()
  process.env.DLSS_LOCAL_TOKEN = localToken
  const userData = app.getPath('userData')
  const environment = runtimeEnvironment(rootDir, userData, {...process.env, LUNARC_RUNTIME_DIR: runtimeDir}, standalone)
  if(manager){environment.LUNARC_COMPONENT_MODE='1';environment.LUNARC_COMPONENT_ROOT=path.join(userData,'components');environment.PATH=path.dirname(manager.python())+path.delimiter+environment.PATH}
  const pythonCandidates = manager ? [manager.python()] : runtimeCandidates(rootDir, app.isPackaged, process.env)
  let pythonPath = ''
  let failure = ''
  for (const c of pythonCandidates) {
    try {
      const result = await probeRuntime(c, rootDir, environment)
      backendStartup.checks = result.checks
      if (result.ready) {pythonPath = c; break}
      failure = '运行环境不完整：' + (result.checks as {status: string; label: string; detail: string}[])
        .filter(c => c.status !== 'ok').map(c => `${c.label}：${c.detail}`).join('；')
    } catch (err) {failure = (err as Error).message}
  }
  if (!pythonPath) {
    const msg = standalone ? `运行环境无法启动，请在组件管理中检查安装结果。\n${failure}` : `未找到可用的开发运行环境。\n${failure}`
    backendStartup = {...backendStartup, state: 'error', message: msg}
    console.error('[Backend]', msg)
    // E2E runs must never block on a modal dialog (it would wedge teardown).
    if (process.env.E2E_TEST !== '1') {
      dialog.showErrorBox('后端启动失败', msg)
    }
    return false
  }

  try {
    const logDir = path.join(userData, 'logs')
    fs.mkdirSync(logDir, {recursive: true})
    const logPath = path.join(logDir, 'backend.log')
    const log = fs.createWriteStream(logPath, {flags: 'a'})
    log.on('error', err => console.error('[Backend] log error:', err.message))
    log.write(`\n[${new Date().toISOString()}] Backend starting\n`)
    backendStartup = {...backendStartup, state: 'starting', message: '正在启动后端', logPath}
    const owned = spawn(pythonPath, ['-I', serverScript, '--port', String(backendPort)], {
      cwd: rootDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
      env: { ...environment, DLSS_LOCAL_TOKEN: localToken },
    })
    pyProc = owned
    pyProc.stdout?.pipe(log, {end: false})
    pyProc.stderr?.pipe(log, {end: false})
    console.log('Python backend spawned with PID:', pyProc.pid, 'using:', pythonPath)
    // Async spawn errors (e.g. ENOENT) do NOT throw synchronously — listen.
    owned.on('error', (err) => {
      if(pyProc!==owned||shuttingDown)return
      backendStartup = {...backendStartup, state: 'error', message: `后端启动失败：${err.message}`}
      log.end()
      console.error('[Backend] spawn error:', err)
      if (process.env.E2E_TEST !== '1') {
        dialog.showErrorBox(
          '后端启动失败',
          `无法启动 Python 后端 (${pythonPath})：${(err as Error).message}\n请检查 PYTHON_PATH 后重试。`
        )
      }
      if(pyProc===owned)pyProc = null
    })
    owned.on('exit', (code, signal) => {
      log.end()
      const unexpected=pyProc===owned&&!shuttingDown
      if(!unexpected)return
      pyProc=null
      backendStartup = {...backendStartup, state: 'error', message: `后端已退出（${code ?? signal}），请查看日志`}
      console.log(`[Backend] exited code=${code} signal=${signal}`)
      try {
        win?.webContents.send('backend-exit', { code, signal: String(signal) })
      } catch {}
      void recovery.recover(`exit:${owned.pid}`, '后端意外退出')
    })
    return true
  } catch (err) {
    console.error('Failed to spawn Python backend:', err)
    backendStartup = {...backendStartup, state: 'error', message: String(err)}
    return false
  }
}

app.setName('Lunarc')

function createWindow() {
  win = new BrowserWindow({
    title: 'Lunarc',
    titleBarStyle: 'hidden',
    ...(process.platform !== 'darwin' ? {
      titleBarOverlay: { color: '#202020', symbolColor: '#efeeea', height: 76 },
    } : {}),
    icon: path.join(process.env.VITE_PUBLIC!, 'lunarc.png'),
    width: 1480,
    height: 940,
    minWidth: 1000,
    minHeight: 680,
    backgroundColor: '#191919',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  // Hide default window menu
  win.setMenuBarVisibility(false)

  if (VITE_DEV_SERVER_URL) {
    win.loadURL(VITE_DEV_SERVER_URL)
  } else {
    win.loadURL('app://studio/index.html')
  }
}

// Security: remote debugging port is prohibited in production mode.
if (process.env.E2E_TEST !== '1') {
  if (app.commandLine.hasSwitch('remote-debugging-port')) {
    console.warn('[Security] removing prohibited remote-debugging-port switch.')
    app.commandLine.removeSwitch('remote-debugging-port')
  }
}

// Minimal necessary E2E test interfaces (strictly gated by E2E_TEST === '1')
let e2eMockOpenFile: string | null = process.env.E2E_MOCK_FILE_PATH || null
let e2eMockOpenDir: string | null = process.env.E2E_MOCK_DIR_PATH || null

if (process.env.E2E_TEST === '1') {
  ipcMain.handle('e2e:setMockOpenFilePath', (_event, filePath: string | null) => {
    e2eMockOpenFile = filePath
    return true
  })
  ipcMain.handle('e2e:setMockOpenDirPath', (_event, dirPath: string | null) => {
    e2eMockOpenDir = dirPath
    return true
  })
}

// IPC handlers for native file dialogs
ipcMain.handle('preview:window', event => {
  if (process.platform !== 'win32' || !win || event.sender !== win.webContents) return null
  const handle = win.getNativeWindowHandle()
  return {parent: (handle.length === 8 ? handle.readBigUInt64LE() : BigInt(handle.readUInt32LE())).toString()}
})

ipcMain.handle('dialog:openRuntime', async event => {
  if (!win || event.sender !== win.webContents) return null
  const result = await dialog.showOpenDialog(win, {
    title: nativeText('导入 DLSSNR 运行库'), properties: ['openFile'],
    filters: [{name: 'DLSSNR DLL / ZIP', extensions: ['dll', 'zip']}],
  })
  return result.canceled ? null : result.filePaths[0] || null
})
ipcMain.handle('dialog:openRuntimeDirectory', async event => {
  if (!win || event.sender !== win.webContents) return null
  const result = await dialog.showOpenDialog(win, {
    title: nativeText('选择 DLSSNR 来源目录'), properties: ['openDirectory'],
  })
  return result.canceled ? null : result.filePaths[0] || null
})
ipcMain.handle('dialog:openFile', async () => {
  if (process.env.E2E_TEST === '1' && e2eMockOpenFile) {
    return e2eMockOpenFile
  }
  if (!win) return null
  const result = await dialog.showOpenDialog(win, {
    title: nativeText('选择导入素材 (视频 / 图像)'),
    properties: ['openFile'],
    filters: [
      { name: nativeText('Media Files'), extensions: ['mp4', 'mkv', 'mov', 'avi', 'wmv', 'flv', 'webm', 'm4v', 'mpg', 'mpeg', 'ts', 'mts', 'm2ts', 'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff', 'avif', 'heic', 'heif', 'jxl', 'jxr', 'wdp', 'hdp'] },
      { name: nativeText('Video Files'), extensions: ['mp4', 'mkv', 'mov', 'avi', 'wmv', 'flv', 'webm', 'm4v', 'mpg', 'mpeg', 'ts', 'mts', 'm2ts'] },
      { name: nativeText('Images'), extensions: ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tif', 'tiff', 'avif', 'heic', 'heif', 'jxl', 'jxr', 'wdp', 'hdp'] },
      { name: nativeText('All Files'), extensions: ['*'] },
    ],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

ipcMain.handle('dialog:openDirectory', async () => {
  if (process.env.E2E_TEST === '1' && e2eMockOpenDir) {
    return e2eMockOpenDir
  }
  if (!win) return null
  const result = await dialog.showOpenDialog(win, {
    title: nativeText('选择导出保存目录'),
    properties: ['openDirectory', 'createDirectory'],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

// Authentication token IPC handler (P0-01)
ipcMain.handle('auth:getToken', () => process.env.DLSS_LOCAL_TOKEN || '')
ipcMain.handle('components:get', event => event.sender===win?.webContents && event.senderFrame===win.webContents.mainFrame ? componentManager()?.state||null : null)
ipcMain.handle('components:install', (event,id:unknown) => event.sender===win?.webContents && event.senderFrame===win.webContents.mainFrame ? componentManager()?.install(id) : null)
ipcMain.handle('components:cancel', event => {if(event.sender===win?.webContents && event.senderFrame===win.webContents.mainFrame)componentManager()?.cancel()})
ipcMain.handle('components:startBackend', async event => {
  if(event.sender!==win?.webContents || event.senderFrame!==win.webContents.mainFrame)return null
  await ensureBackend();return backendStartup
})
ipcMain.handle('backend:getStartup', (event) => event.sender === win?.webContents ? backendStartup : null)
ipcMain.handle('updates:get', event => event.sender === win?.webContents && event.senderFrame === win?.webContents.mainFrame ? appUpdates().state : null)
ipcMain.handle('updates:check', (event, preview: unknown) => {
  if ((event.sender !== win?.webContents || event.senderFrame !== win?.webContents.mainFrame) || typeof preview !== 'boolean') return null
  return appUpdates().check(preview)
})
ipcMain.handle('updates:download', event => event.sender === win?.webContents && event.senderFrame === win?.webContents.mainFrame ? appUpdates().download() : null)
ipcMain.handle('updates:cancel', event => {if (event.sender === win?.webContents && event.senderFrame === win?.webContents.mainFrame) appUpdates().cancel()})
ipcMain.handle('updates:openDirectory', async event => {
  if ((event.sender !== win?.webContents || event.senderFrame !== win?.webContents.mainFrame) || appUpdates().state.phase !== 'ready') return
  const directory=appUpdates().state.directory
  if (directory) {const error=await shell.openPath(directory); if (error) throw new Error(error)}
})

// Only the application renderer can update the two supported chrome palettes.
ipcMain.handle('window:setTheme', (event, theme: unknown) => {
  if (!win || event.sender !== win.webContents || (theme !== 'light' && theme !== 'dark')) return
  if (process.platform !== 'darwin') {
    win.setTitleBarOverlay(theme === 'light'
      ? { color: '#f7f6f2', symbolColor: '#242522', height: 76 }
      : { color: '#202020', symbolColor: '#efeeea', height: 76 })
  }
})

function killPythonBackend() {
  // Detach ownership first: its late exit event must not clear a new child.
  const owned=pyProc
  pyProc=null
  if (owned && owned.pid) {
    const pid = owned.pid
    try {
      if (process.platform === 'win32') {
        execFileSync('taskkill', ['/F', '/T', '/PID', String(pid)], { windowsHide: true, stdio: 'ignore', timeout: 10000 })
      } else {
        owned.kill('SIGKILL')
      }
    } catch (error) {
      console.error('[Backend] process-tree cleanup failed:', error)
      owned.kill('SIGKILL')
    }
    pyProc = null
  }
}

app.on('window-all-closed', () => {
  shuttingDown=true;recovery.stop();if(recoveryTimer)clearInterval(recoveryTimer)
  killPythonBackend()
  if (process.platform !== 'darwin') {
    app.quit()
    win = null
  }
})

app.on('before-quit', () => {
  shuttingDown=true;recovery.stop();if(recoveryTimer)clearInterval(recoveryTimer)
  updates?.cancel()
  components?.cancel()
  killPythonBackend()
})

app.on('will-quit', () => {
  killPythonBackend()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

app.whenReady().then(async () => {
  protocol.handle('app', (request) => {
    const filePath = resolveAppProtocolPath(request.url, RENDERER_DIST)
    if (!filePath) return new Response('Not Found', {status: 404})
    return net.fetch(pathToFileURL(filePath).toString())
  })

  const manager=componentManager()
  if(manager){
    if(await checkServerListening(backendPort))process.env.DLSS_LOCAL_TOKEN ||= await resolveExistingBackendToken(path.resolve(process.env.APP_ROOT,'..')) || crypto.randomUUID()
    else process.env.DLSS_LOCAL_TOKEN ||= crypto.randomUUID()
    backendStartup={state:'setup',message:'请先安装基础组件'}
    createWindow()
  }else{
    await ensureBackend()
    createWindow()
  }
  startRecoveryMonitor()
})
