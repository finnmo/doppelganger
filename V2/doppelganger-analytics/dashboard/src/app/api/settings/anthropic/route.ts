import { NextRequest, NextResponse } from 'next/server';
import {
  ANTHROPIC_MODELS,
  clearAnthropicApiKey,
  getPublicAnthropicSettings,
  saveAnthropicApiKey,
  saveAnthropicModelOnly,
} from '@/lib/server/anthropicSecrets';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** Public status — never includes the raw API key. */
export async function GET() {
  try {
    const settings = getPublicAnthropicSettings();
    return NextResponse.json({
      ...settings,
      models: ANTHROPIC_MODELS,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to read settings';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

/**
 * Save / update Claude API key.
 * Body: { apiKey: string, model?: string }
 * Never logs the key; never returns the key.
 */
export async function PUT(request: NextRequest) {
  try {
    const body = await request.json().catch(() => null);
    if (!body || typeof body !== 'object') {
      return NextResponse.json({ error: 'Invalid JSON body.' }, { status: 400 });
    }

    const apiKey = typeof body.apiKey === 'string' ? body.apiKey : '';
    const model = typeof body.model === 'string' ? body.model : undefined;

    // Model-only update when key already stored and apiKey omitted/empty
    if (!apiKey.trim() && model) {
      try {
        const settings = saveAnthropicModelOnly(model);
        return NextResponse.json({ ok: true, ...settings, models: ANTHROPIC_MODELS });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Could not update model';
        return NextResponse.json({ error: message }, { status: 400 });
      }
    }

    const result = await saveAnthropicApiKey(apiKey, model);
    if (!result.ok) {
      return NextResponse.json({ error: result.error }, { status: 400 });
    }

    return NextResponse.json({
      ok: true,
      ...result.settings,
      models: ANTHROPIC_MODELS,
    });
  } catch (err) {
    // Never include request body in error responses
    const message = err instanceof Error ? err.message : 'Failed to save API key';
    const safe = message.replace(/sk-ant-[a-zA-Z0-9_-]+/g, '[redacted]');
    return NextResponse.json({ error: safe }, { status: 500 });
  }
}

/** Remove the encrypted key from disk (env var fallback still applies if set). */
export async function DELETE() {
  try {
    const settings = clearAnthropicApiKey();
    return NextResponse.json({
      ok: true,
      ...settings,
      models: ANTHROPIC_MODELS,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to clear API key';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
