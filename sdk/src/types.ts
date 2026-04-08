/** Options for a speech synthesis request */
export interface SpeechOptions {
  /** Voice ID for synthesis */
  voice?: string
  /** Output audio format */
  responseFormat?: 'mp3' | 'wav' | 'ogg' | 'flac' | 'opus'
  /** Speech speed multiplier (default 1.0) */
  speed?: number
}

/** Model listing */
export interface ModelInfo {
  id: string
  object: string
  owned_by: string
}

export interface ModelList {
  object: string
  data: ModelInfo[]
}

/** Operator info from the BSM contract */
export interface OperatorInfo {
  address: string
  model: string
  gpuCount: number
  totalVramMib: number
  gpuModel: string
  endpoint: string
  active: boolean
}

/** Model pricing config from the BSM contract */
export interface ModelConfig {
  maxContextLen: number
  pricePer1kCharacters: bigint
  minGpuVramMib: number
  enabled: boolean
}

/** ShieldedCredits spend authorization */
export interface SpendAuth {
  commitment: string
  serviceId: bigint
  jobIndex: number
  amount: bigint
  operator: string
  nonce: bigint
  expiry: bigint
  signature: string
}

/** Credit account state */
export interface CreditAccount {
  spendingKey: string
  token: string
  balance: bigint
  totalFunded: bigint
  totalSpent: bigint
  nonce: bigint
}

/** Client configuration */
export interface VoiceClientConfig {
  /** Operator HTTP endpoint URL */
  operatorUrl: string

  /** ShieldedCredits contract address */
  shieldedCreditsAddress: string

  /** Chain ID for EIP-712 domain */
  chainId: number

  /** Credit account commitment (keccak256(spendingKey, salt)) */
  commitment: string

  /** Service ID on Tangle */
  serviceId: bigint

  /** Operator's on-chain address (for SpendAuth designation) */
  operatorAddress: string

  /** Spending key private key (ephemeral, for signing SpendAuths) */
  spendingKeyPrivate: string
}

/** STT provider info */
export interface SttProvider {
  backend: string
  model: string
  mode: string
  language: string
}

/** STT provider listing */
export interface SttProviderList {
  providers: SttProvider[]
}

/** Transcription result */
export interface TranscriptionResult {
  text: string
  language: string
  duration: number
}

/** Options for a transcription request */
export interface TranscribeOptions {
  /** Language hint (e.g. "en", "es") */
  language?: string
  /** Model override */
  model?: string
}

/** Error response from operator */
export interface ErrorResponse {
  error: {
    message: string
    type: string
    code: string
  }
}
