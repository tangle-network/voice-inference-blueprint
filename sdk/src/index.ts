import { ethers } from "ethers";
import type {
  SpeechOptions,
  VoiceClientConfig,
  SpendAuth,
  ModelList,
  ErrorResponse,
  SttProvider,
  SttProviderList,
  TranscriptionResult,
  TranscribeOptions,
} from "./types";

export * from "./types";

// EIP-712 constants matching ShieldedCredits.sol
const SPEND_TYPEHASH = ethers.keccak256(
  ethers.toUtf8Bytes(
    "SpendAuthorization(bytes32 commitment,uint64 serviceId,uint8 jobIndex,uint256 amount,address operator,uint256 nonce,uint64 expiry)"
  )
);

const EIP712_DOMAIN = {
  name: "ShieldedCredits",
  version: "1",
};

/**
 * High-level voice inference client that handles SpendAuth signing and request routing.
 *
 * Usage:
 * ```ts
 * const client = createVoiceClient({
 *   operatorUrl: "https://op1.example.com",
 *   shieldedCreditsAddress: "0x...",
 *   chainId: 1,
 *   commitment: "0x...",
 *   serviceId: 1n,
 *   operatorAddress: "0x...",
 *   spendingKeyPrivate: "0x...",
 * });
 *
 * const audioBuffer = await client.synthesize("Hello, world!");
 * // audioBuffer is an ArrayBuffer containing MP3 audio
 * ```
 */
export function createVoiceClient(config: VoiceClientConfig) {
  const spendingWallet = new ethers.Wallet(config.spendingKeyPrivate);
  let currentNonce = 0n;

  /**
   * Sign a SpendAuth for a given amount.
   */
  async function signSpendAuth(
    amount: bigint,
    nonce: bigint,
    expirySeconds: number = 300
  ): Promise<SpendAuth> {
    const expiry = BigInt(Math.floor(Date.now() / 1000)) + BigInt(expirySeconds);

    const domain = {
      ...EIP712_DOMAIN,
      chainId: config.chainId,
      verifyingContract: config.shieldedCreditsAddress,
    };

    const types = {
      SpendAuthorization: [
        { name: "commitment", type: "bytes32" },
        { name: "serviceId", type: "uint64" },
        { name: "jobIndex", type: "uint8" },
        { name: "amount", type: "uint256" },
        { name: "operator", type: "address" },
        { name: "nonce", type: "uint256" },
        { name: "expiry", type: "uint64" },
      ],
    };

    const value = {
      commitment: config.commitment,
      serviceId: config.serviceId,
      jobIndex: 0, // TTS job
      amount: amount,
      operator: config.operatorAddress,
      nonce: nonce,
      expiry: expiry,
    };

    const signature = await spendingWallet.signTypedData(domain, types, value);

    return {
      commitment: config.commitment,
      serviceId: config.serviceId,
      jobIndex: 0,
      amount,
      operator: config.operatorAddress,
      nonce,
      expiry,
      signature,
    };
  }

  /**
   * Estimate cost for a request based on input character count.
   */
  function estimateCost(
    characterCount: number,
    pricePer1kCharacters: bigint
  ): bigint {
    return (BigInt(characterCount) * pricePer1kCharacters) / 1000n;
  }

  /**
   * Send a speech synthesis request with automatic SpendAuth signing.
   * Returns raw audio bytes as an ArrayBuffer.
   */
  async function synthesize(
    input: string,
    options: SpeechOptions & {
      /** Pre-authorized amount. If not set, a default estimate is used. */
      authorizedAmount?: bigint;
      /** Price per 1k characters (from model config) */
      pricePer1kCharacters?: bigint;
    } = {}
  ): Promise<ArrayBuffer> {
    const pricePer1k = options.pricePer1kCharacters ?? 10n;

    const amount =
      options.authorizedAmount ??
      estimateCost(input.length, pricePer1k);

    const spendAuth = await signSpendAuth(amount, currentNonce);
    currentNonce++;

    const body = {
      input,
      voice: options.voice,
      response_format: options.responseFormat ?? "mp3",
      speed: options.speed ?? 1.0,
      spend_auth: {
        commitment: spendAuth.commitment,
        service_id: Number(spendAuth.serviceId),
        job_index: spendAuth.jobIndex,
        amount: spendAuth.amount.toString(),
        operator: spendAuth.operator,
        nonce: Number(spendAuth.nonce),
        expiry: Number(spendAuth.expiry),
        signature: spendAuth.signature,
      },
    };

    const response = await fetch(`${config.operatorUrl}/v1/audio/speech`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error: ErrorResponse = await response.json();
      throw new Error(
        `Speech synthesis failed (${response.status}): ${error.error?.message ?? response.statusText}`
      );
    }

    return response.arrayBuffer();
  }

  /**
   * List available models from the operator.
   */
  async function listModels(): Promise<ModelList> {
    const response = await fetch(`${config.operatorUrl}/v1/models`);
    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * Check operator health.
   */
  async function healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${config.operatorUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Set the current nonce (e.g., after querying the credit account on-chain).
   */
  function setNonce(nonce: bigint) {
    currentNonce = nonce;
  }

  /**
   * List available STT providers from the operator.
   */
  async function listSttProviders(): Promise<SttProvider[]> {
    const response = await fetch(`${config.operatorUrl}/v1/stt/providers`);
    if (!response.ok) {
      throw new Error(`Failed to list STT providers: ${response.statusText}`);
    }
    const data: SttProviderList = await response.json();
    return data.providers;
  }

  /**
   * Transcribe audio using the operator's configured STT backend.
   * Accepts a Blob (browser) or Buffer (Node.js).
   */
  async function transcribe(
    audio: Blob | Buffer,
    options: TranscribeOptions = {}
  ): Promise<TranscriptionResult> {
    const formData = new FormData();

    if (audio instanceof Blob) {
      formData.append("file", audio, "audio.wav");
    } else {
      // Node.js Buffer — wrap in a Blob
      const bytes = new Uint8Array(audio.byteLength);
      bytes.set(audio);
      const blob = new Blob([bytes.buffer], { type: "audio/wav" });
      formData.append("file", blob, "audio.wav");
    }

    if (options.language) {
      formData.append("language", options.language);
    }
    if (options.model) {
      formData.append("model", options.model);
    }

    const response = await fetch(
      `${config.operatorUrl}/v1/audio/transcriptions`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      const error: ErrorResponse = await response.json();
      throw new Error(
        `Transcription failed (${response.status}): ${error.error?.message ?? response.statusText}`
      );
    }

    return response.json();
  }

  return {
    synthesize,
    transcribe,
    listModels,
    listSttProviders,
    healthCheck,
    signSpendAuth,
    estimateCost,
    setNonce,
    get address() {
      return spendingWallet.address;
    },
  };
}

export type VoiceClient = ReturnType<typeof createVoiceClient>;
