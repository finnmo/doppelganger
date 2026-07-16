import { NextRequest, NextResponse } from 'next/server';
import {
  clearEmbeddingsApiKey,
  getPublicEmbeddingsSettings,
  saveEmbeddingsApiKey,
} from '@/lib/server/anthropicSecrets';
import { EMBEDDING_MODELS, type EmbeddingProvider } from '@/lib/server/embeddingsVectors';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const PROVIDERS = [
  { id: 'openai' as const, label: EMBEDDING_MODELS.openai.label, model: EMBEDDING_MODELS.openai.id },
  { id: 'voyage' as const, label: EMBEDDING_MODELS.voyage.label, model: EMBEDDING_MODELS.voyage.id },
];

/** Public status — never includes the raw API key. */
export async function GET() {
  try {
    const settings = getPublicEmbeddingsSettings();
    return NextResponse.json({
      ...settings,
      providers: PROVIDERS,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to read embeddings settings';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/**
 * Save / update embeddings API key (OpenAI or Voyage).
 * Body: { apiKey: string, provider: 'openai' | 'voyage', model?: string }
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json().catch(() => null);
    if (!body || typeof body !== 'object') {
      return NextResponse.json({ error: 'Invalid JSON body.' }, { status: 400 });
    }

    const apiKey = typeof body.apiKey === 'string' ? body.apiKey : '';
    const providerRaw = typeof body.provider === 'string' ? body.provider : 'openai';
    const provider: EmbeddingProvider = providerRaw === 'voyage' ? 'voyage' : 'openai';
    const model = typeof body.model === 'string' ? body.model : undefined;

    if (!apiKey.trim()) {
      return NextResponse.json({ error: 'apiKey is required.' }, { status: 400 });
    }

    const result = await saveEmbeddingsApiKey(apiKey, provider, model);
    if (!result.ok) {
      return NextResponse.json({ error: result.error }, { status: 400 });
    }

    return NextResponse.json({
      ok: true,
      ...result.settings,
      providers: PROVIDERS,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to save embeddings key';
    const safe = message.replace(/sk-[a-zA-Z0-9_-]+/g, '[redacted]');
    return NextResponse.json({ error: safe }, { status: 500 });
  }
}

export async function DELETE() {
  try {
    const settings = clearEmbeddingsApiKey();
    return NextResponse.json({
      ok: true,
      ...settings,
      providers: PROVIDERS,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to clear embeddings key';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
