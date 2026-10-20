export type RenderStageKind = 'nr' | 'dlaa' | 'vsr' | 'vfg' | 'truehdr' | 'illustration' | 'animesharp'
export type RenderStage = RenderStageKind | `vsr:${string}`

export interface VideoInfo {
  media_type?: 'video' | 'image'
  has_alpha?: boolean
  path: string
  filename: string
  frames: number
  fps: number
  width: number
  height: number
  duration_seconds: number
  codec_name?: string
  pix_fmt?: string
  bit_depth?: number
  is_hdr?: boolean
  color_transfer?: string
  color_primaries?: string
  color_space?: string
  color_range?: string
}

export type ExportBackend = 'auto' | 'legacy' | 'gpu_transfer'

// Which native DLSS host draws a frame. There is one: the project GPU host,
// native/dlss_gpu_host.dll, for both guidance modes and on either export chain.
// The retired dlssnr_host.dll was 'host_pointer'; the service refuses that value by
// name rather than drawing the frame somewhere else, so the type cannot carry it.
// Absent still means the service's own default, which is the same host.
export type DlssBackend = 'gpu_flow_d3d12'
export type ReconstructionMode = 'single' | 'dual' | 'triple'
export type FusionMode = 'catmull_baseline' | 'data_consistency_experimental' | 'sequential'

export interface PostControls {
  total_gain: number; lightness_gain: number; chroma_gain: number
  low_frequency_gain: number; high_frequency_gain: number
  hue_protection: number; dark_protection: number; highlight_protection: number
  soft_compression: number; roi_kind: 0 | 1 | 2
  roi_bounds: [number, number, number, number]; roi_feather: number
  debug_mode: 0 | 1 | 2 | 3
}
export interface TemporalControls {
  scene_detection: boolean
  mode: 'off' | 'static_low_frequency'
  history_weight: number
}

export interface DLSSSettings {
  neural_mode?: 'nr' | 'dlaa' | 'vsr' | 'nr_dlaa' | 'nr_vsr'
  render_pipeline?: RenderStage[]
  render_stage_options?: Partial<Record<RenderStage, Partial<DLSSSettings>>>
  image_model_variant?: 'denoise' | 'detail'
  image_model_tile?: 0 | 128 | 256 | 512
  image_memory_mode?: 'memory' | 'fast'
  render_stage_enabled?: Partial<Record<RenderStage, boolean>>
  frame_generation_saved_mode?: 'vfg' | 'vfg_dynamic'
  dlaa_preset?: 0 | 5 | 6 | 10 | 11 | 12 | 13
  vsr_quality?: number
  vsr_mode?: 'standard' | 'denoise' | 'deblur' | 'high_bitrate' | 'streaming' | 'bicubic'
  vsr_strength?: number | null
  vsr_output_mode?: 'scale' | 'resolution'
  vsr_scale?: number
  vsr_target?: '720p' | '1080p' | '1440p' | '4k' | '8k' | 'custom'
  vsr_width?: number
  vsr_height?: number
  vsr_runtime_path?: string
  frame_generation?: 'off' | 'vfg' | 'vfg_dynamic'
  frame_multiplier?: 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16
  framegen_target_fps?: number | string
  framegen_scene_detection?: boolean
  framegen_quality?: 0 | 1 | 2
  dlss_dll_path?: string
  preset: number
  style: number
  intensity: number
  local_tone: number
  local_struct: number
  skin_struct: number
  use_auto_mask: number
  ui_correction: number
  guidance_mode: number
  motion_scale_x: number
  motion_scale_y: number
  working_scale: number
  residual_pyramid: boolean
  residual_strength: number
  flow_engine: string
  flow_quality: string
  raft_analysis_edge?: 512 | 720 | 1080
  raft_iterations?: number
  export_backend: ExportBackend
  dlss_backend?: DlssBackend
  multi_pass?: number
  reconstruction_mode: ReconstructionMode
  layer_scales: number[]
  layer_weights: number[]
  layer_execution?: 'sequential' | 'multiscale'
  truehdr_enabled?: boolean
  truehdr_contrast?: number
  truehdr_saturation?: number
  truehdr_middle_gray?: number
  truehdr_peak_nits?: number
  truehdr_debanding?: boolean
  layer_settings?: Partial<Pick<DLSSSettings, 'preset' | 'style' | 'intensity' | 'local_tone' |
    'local_struct' | 'skin_struct' | 'use_auto_mask' | 'ui_correction'>>[]
  fusion_mode: FusionMode
  fusion_beta: number
  fusion_max_iterations: number
  fusion_relative_tolerance: number
  // Experimental HDR10 export opt-in. The backend owns the contract; these two
  // keys only carry the user's request to it. An older GET /api/settings reply
  // that omits them means off / 8.0 — the same as the service defaults — and App
  // merges DEFAULT_SETTINGS under the reply so the UI never sees undefined.
  // Never a preview-path switch: it changes only what an export job asks for.
  experimental_hdr_scrgb: boolean
  // Paper white in scRGB for the whole job (8.0 scRGB = 640 cd/m² at 80 nits per
  // unit). Fixed per export job: changing it mid-job does not move the master.
  hdr_paper_white_scrgb: number
  post_controls?: PostControls
  temporal_controls?: TemporalControls
}

export interface TranscodeSettings {
  codec: string
  container: string
  suffix: string
  rate_mode: string
  cq: number
  video_bitrate: string
  qp: number
  peak_policy: string
  maxrate: string
  bufsize: string
  preset: string
  tune: string
  multipass: string
  gpu: string
  decode_mode: string
  finish_action: string
  gop_mode: string
  gop: number
  b_frames: string
  b_ref_mode: string
  lookahead: string
  lookahead_level: string
  aq_mode: string
  aq_strength: number
  weighted_prediction: boolean
  temporal_filter: string
  split_encode: string
  profile: string
  level: string
  tier: string
  resolution_mode: string
  custom_width: number
  custom_height: number
  custom_factor: number
  keep_aspect: boolean
  scale_algorithm: string
  pixel_format: string
  color_preset: string
  custom_color_space: string
  custom_color_primaries: string
  custom_color_transfer: string
  custom_color_range: string
  audio_mode: string
  audio_bitrate: string
  subtitle_mode: string
  copy_attachments: boolean
  copy_metadata: boolean
  copy_chapters: boolean
  overwrite_output: boolean
  skip_existing: boolean
  extra_args: string
}

export interface RenderJob {
  id: string
  name: string
  input_path: string
  output_path: string
  codec: string
  container: string
  cq: number
  preset: string
  with_audio: boolean
  settings: DLSSSettings
  frame_range?: {start_frame:number;end_frame:number}
  source_fps?: number
  preview_segment?: boolean
  transcode_settings?: TranscodeSettings
  // 'skipped' is its own outcome, not a synonym for 'completed': the destination was
  // left untouched because it already existed, so nothing was rendered or overwritten.
  status: 'pending' | 'running' | 'completed' | 'skipped' | 'cancelled' | 'failed'
  progress: number
  fps: number
  eta: number
  current_frame: number
  total_frames: number
  error?: string | null
  error_traceback?: string | null
  // Which rule skipped a job ('output_exists'), so the row can say why the file it
  // names was never written. Set only when status is 'skipped'.
  skipped_reason?: string | null
  // Which DLSS host ran or refused this job, and that host's own last error. The
  // retired dlssnr_host.dll used to publish a tail of its dlssnr_run.log here, which
  // a GPU-host failure then showed as if it were its own log.
  dlss_host?: { backend: string; dll?: string | null; abi?: number | null } | null
  dlss_host_error?: string | null
}

export interface QueueState {
  jobs: RenderJob[]
  active_id: string | null
  // True while the queue worker thread still owns a round. Between the last row
  // turning terminal and that thread exiting nothing is running, so a row's own
  // status cannot answer whether a reset is safe - only this can.
  queue_running: boolean
}

export interface DiagnosticLogEntry {
  time: string
  level: 'INFO' | 'WARN' | 'ERROR' | 'FATAL' | string
  tag: string
  message: string
  detail?: string
}

export interface SystemInfo {
  gpu_memory?: {sampled_at:number;phase:string;active:boolean;error?:string;known_device_allocations_bytes?:number;
    process?:{local_usage:number;local_budget:number;nonlocal_usage:number;nonlocal_budget:number};
    components?:{owner:string;kind:string;allocated_device_bytes:number;host_staging_bytes?:number;scope?:string}[];
    preview_cache?:{private_plane_bytes?:number};torch?:{allocated_bytes:number;reserved_bytes:number};unattributed?:string[]}
  adapter_memory_used_gb?: number | null
  adapter_memory_total_gb?: number | null
  adapter_memory_scope?: string
  gpu: string | null
  vram_total_gb: number | null
  vram_used_gb: number | null
  gpu_error?: string | null
  vram_scope?: string
  os: string
  python_version: string
  backend_ready: boolean
}

export interface EnvironmentCheck {id: string; label: string; status: 'ok'|'pending'|'missing'|'error'; detail: string; action: string}
export interface EnvironmentReport {checked_at: number; checks: EnvironmentCheck[]; features: EnvironmentCheck[]; isolated?: boolean; standalone?: boolean; scope: string}

// Window Electron API
declare global {
  interface Window {
    electronAPI?: {
      getLanguage?: () => Promise<unknown>
      setLanguage?: (locale: import('../i18n/translate').LanguagePreference) => Promise<void>
      openRuntimeDialog?: () => Promise<string | null>
      openRuntimeDirDialog?: () => Promise<string | null>
      openFileDialog: () => Promise<string | null>
      openDirDialog: () => Promise<string | null>
      getPathForFile: (file: File) => string
      getPreviewWindow?: () => Promise<{parent:string}|null>
      windowOverlay?: boolean
      setWindowTheme?: (theme: 'dark' | 'light') => Promise<void>
      localToken?: string
      getBackendStartup?: () => Promise<{state: string; message: string; logPath?: string; checks?: EnvironmentCheck[]} | null>
      onBackendExit?: (cb: (info: { code: number | null; signal: string }) => void) => (() => void)
      onBackendRecovery?: (cb: (state: {phase:string;message:string}) => void) => (() => void)
      isElectron: boolean
    }
  }
}
