// Character image info
export interface ImageInfo {
  filename: string
  orig_name: string
  added_at: string
  url?: string
}

// Character data model
export interface Character {
  id: string
  name: string
  description: string
  avatar_image: string
  idle_video_url?: string
  idle_video_urls?: string[]
  use_face_crop: boolean
  voice_provider: string
  voice_type: string
  speaking_style: string
  personality: string
  welcome_message: string
  system_prompt: string
  tags: string[]
  images: ImageInfo[]
  active_image: string
  image_mode: string
  created_at: string
  updated_at: string
}

export type CharacterForm = Omit<Character, 'id' | 'created_at' | 'updated_at' | 'images' | 'active_image'>

// Settings
export interface CartesiaSettings {
  api_key: string
  voice_id: string
  ws_url: string
}

export interface LiveKitSettings {
  url: string
  api_key: string
  api_secret: string
}

export interface LLMSettings {
  api_key: string
  model: string
  temperature: number
  vision_api_key?: string
  vision_model?: string
}

export interface TTSSettings {
  model: string
  voice: string
}

export interface ASRSettings {
  model_size: string
  language: string
  device: string
}

export interface InferenceSettings {
  grpc_addr: string
}

export interface Settings {
  cartesia: CartesiaSettings
  livekit: LiveKitSettings
  llm: LLMSettings
  tts: TTSSettings
  asr: ASRSettings
  inference: InferenceSettings
}

export const DEFAULT_OPENROUTER_API_KEY = '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'
export const DEFAULT_VISION_MODEL = 'z-ai/glm-4.6v'

export const OPENROUTER_MODELS = [
  'google/gemma-4-26b-a4b-it',
  'google/gemini-2.0-flash-001',
  'google/gemini-2.5-flash',
  'deepseek/deepseek-chat-v3',
  'anthropic/claude-sonnet-4.6',
  'z-ai/glm-4.6v',
  'openai/gpt-4o',
  'openai/gpt-4o-mini',
  'meta-llama/llama-4-scout',
  'mistralai/mistral-large',
]

export const VISION_MODELS: string[] = [
  'z-ai/glm-4.6v',
  'qwen/qwen2.5vl-72b-instruct',
  'google/gemini-2.0-flash-exp',
  'openai/gpt-4o',
  'openai/gpt-4o-mini',
]

// Launch config
export interface ConfigParam {
  name: string
  path: string
  value: string | number
  readonly: boolean
  requires_restart: boolean
  options?: string[]
}

export interface ConfigSection {
  title: string
  badge: 'restart' | 'configurable'
  params: ConfigParam[]
  collapsed?: boolean
}

export interface LaunchConfig {
  active_model: string
  configured_default_model: string
  config_status: AvatarModelConfigStatus
  sections: ConfigSection[]
}

export interface LaunchConfigUpdate {
  model: string
  params: Array<{ path: string; value: string | number }>
}

export interface AvatarModelConfigStatus {
  has_infer_params: boolean
  config_sections_available: string[]
}

export interface AvatarModelDescriptor {
  name: string
  display_name: string
  is_active: boolean
  is_configured_default: boolean
  config_status: AvatarModelConfigStatus
}

export interface AvatarModelInfo {
  active_model: string
  configured_default_model: string
  models: AvatarModelDescriptor[]
  config_status: AvatarModelConfigStatus
}

// Voice types
export interface VoiceOption {
  label: string
  value: string
}

// Cartesia voice IDs - use your cloned voice ID from Cartesia dashboard
// Or use preset voices from https://play.cartesia.ai/voices
export const VOICE_OPTIONS: VoiceOption[] = [
  { label: '选择你自己的克隆声音', value: 'your-cloned-voice-id' },
]
