import { getAnthropicApiKey, getAnthropicModel } from './anthropicSecrets';
import type { ChatTurn } from './buildPersonaPrompt';

export interface AnthropicChatResult {
  text: string;
  model: string;
  inputTokens?: number;
  outputTokens?: number;
}

/**
 * Call Anthropic Messages API. Never logs the API key or full prompt.
 */
export async function callAnthropicMessages(params: {
  system: string;
  messages: ChatTurn[];
  maxTokens?: number;
}): Promise<AnthropicChatResult> {
  const apiKey = getAnthropicApiKey();
  if (!apiKey) {
    throw new Error(
      'Claude API key is not configured. Open API key settings and save a key from console.anthropic.com.'
    );
  }

  const model = getAnthropicModel();
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      max_tokens: params.maxTokens ?? 1024,
      system: params.system,
      messages: params.messages.map((m) => ({
        role: m.role,
        content: m.content,
      })),
    }),
    signal: AbortSignal.timeout(60_000),
  });

  const raw = await res.text();
  let data: {
    content?: Array<{ type: string; text?: string }>;
    error?: { message?: string };
    usage?: { input_tokens?: number; output_tokens?: number };
  };
  try {
    data = JSON.parse(raw) as typeof data;
  } catch {
    throw new Error(`Anthropic returned a non-JSON response (${res.status}).`);
  }

  if (!res.ok) {
    const msg = (data.error?.message || `Anthropic API error (${res.status})`)
      .replace(/sk-ant-[a-zA-Z0-9_-]+/g, '[redacted]');
    throw new Error(msg);
  }

  const text = (data.content ?? [])
    .filter((b) => b.type === 'text' && b.text)
    .map((b) => b.text!.trim())
    .join('\n')
    .trim();

  if (!text) {
    throw new Error('Claude returned an empty reply.');
  }

  return {
    text,
    model,
    inputTokens: data.usage?.input_tokens,
    outputTokens: data.usage?.output_tokens,
  };
}
