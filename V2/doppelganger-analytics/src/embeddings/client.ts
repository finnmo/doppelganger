/**
 * Call OpenAI or Voyage embedding APIs.
 */

import { EMBEDDING_MODELS, type EmbeddingProvider } from './vectors.js';

export async function embedTexts(
  texts: string[],
  options: {
    provider: EmbeddingProvider;
    apiKey: string;
    model?: string;
  }
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const provider = options.provider;
  const model = options.model ?? EMBEDDING_MODELS[provider].id;

  if (provider === 'openai') {
    return embedOpenAI(texts, options.apiKey, model);
  }
  return embedVoyage(texts, options.apiKey, model);
}

async function embedOpenAI(texts: string[], apiKey: string, model: string): Promise<number[][]> {
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      input: texts.map((t) => t.slice(0, 8000)),
    }),
    signal: AbortSignal.timeout(120_000),
  });

  const data = (await res.json()) as {
    data?: Array<{ embedding: number[]; index: number }>;
    error?: { message?: string };
  };

  if (!res.ok) {
    throw new Error(data.error?.message || `OpenAI embeddings error (${res.status})`);
  }

  const rows = data.data ?? [];
  rows.sort((a, b) => a.index - b.index);
  return rows.map((r) => r.embedding);
}

async function embedVoyage(texts: string[], apiKey: string, model: string): Promise<number[][]> {
  const res = await fetch('https://api.voyageai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      input: texts.map((t) => t.slice(0, 8000)),
      input_type: 'document',
    }),
    signal: AbortSignal.timeout(120_000),
  });

  const data = (await res.json()) as {
    data?: Array<{ embedding: number[]; index?: number }>;
    error?: { message?: string } | string;
  };

  if (!res.ok) {
    const msg =
      typeof data.error === 'string'
        ? data.error
        : data.error?.message || `Voyage embeddings error (${res.status})`;
    throw new Error(msg);
  }

  const rows = data.data ?? [];
  if (rows.length > 0 && rows[0].index != null) {
    rows.sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  }
  return rows.map((r) => r.embedding);
}

/** Query embedding — Voyage wants input_type query for asymmetric retrieval. */
export async function embedQuery(
  text: string,
  options: { provider: EmbeddingProvider; apiKey: string; model?: string }
): Promise<number[]> {
  if (options.provider === 'voyage') {
    const model = options.model ?? EMBEDDING_MODELS.voyage.id;
    const res = await fetch('https://api.voyageai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${options.apiKey}`,
      },
      body: JSON.stringify({
        model,
        input: [text.slice(0, 8000)],
        input_type: 'query',
      }),
      signal: AbortSignal.timeout(30_000),
    });
    const data = (await res.json()) as {
      data?: Array<{ embedding: number[] }>;
      error?: { message?: string } | string;
    };
    if (!res.ok) {
      const msg =
        typeof data.error === 'string'
          ? data.error
          : data.error?.message || `Voyage embeddings error (${res.status})`;
      throw new Error(msg);
    }
    return data.data?.[0]?.embedding ?? [];
  }

  const [vec] = await embedTexts([text], options);
  return vec ?? [];
}
