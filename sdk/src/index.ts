import { signSpendAuth as signTcloudSpendAuth } from "@tangle-network/tcloud/shielded";
import type { Hex } from "viem";
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

export type * from "./types";

export function createVoiceClient(config: VoiceClientConfig) {
  let currentNonce = 0n;

  async function signSpendAuth(
    amount: bigint,
    nonce: bigint,
    expirySeconds: number = 300
  ): Promise<SpendAuth> {
    const expiry = BigInt(Math.floor(Date.now() / 1000)) + BigInt(expirySeconds);

    return signTcloudSpendAuth(
      { privateKey: config.spendingKeyPrivate as Hex, commitment: config.commitment as Hex },
      { chainId: config.chainId, creditsAddress: config.shieldedCreditsAddress as Hex,
        serviceId: config.serviceId, jobIndex: 0, operator: config.operatorAddress as Hex,
        amount, nonce, expiry },
    );

  }

  function estimateCost(
    characterCount: number,
    pricePer1kCharacters: bigint
  ): bigint {
    return (BigInt(characterCount) * pricePer1kCharacters) / 1000n;
  }

  async function synthesize(
    input: string,
    options: SpeechOptions & {
      authorizedAmount?: bigint;
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
      spend_auth: spendAuth,
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

  async function listModels(): Promise<ModelList> {
    const response = await fetch(`${config.operatorUrl}/v1/models`);
    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }
    return response.json();
  }

  async function healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${config.operatorUrl}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  function setNonce(nonce: bigint) {
    currentNonce = nonce;
  }

  async function listSttProviders(): Promise<SttProvider[]> {
    const response = await fetch(`${config.operatorUrl}/v1/stt/providers`);
    if (!response.ok) {
      throw new Error(`Failed to list STT providers: ${response.statusText}`);
    }
    const data: SttProviderList = await response.json();
    return data.providers;
  }

  async function transcribe(
    audio: Blob | Buffer,
    options: TranscribeOptions = {}
  ): Promise<TranscriptionResult> {
    const formData = new FormData();

    if (audio instanceof Blob) {
      formData.append("file", audio, "audio.wav");
    } else {
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
  };
}

export type VoiceClient = ReturnType<typeof createVoiceClient>;
