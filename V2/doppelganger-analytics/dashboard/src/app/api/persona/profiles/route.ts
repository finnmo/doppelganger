import { NextResponse } from 'next/server';
import { listPersonaSummaries, resolvePersonaProfilesPath } from '@/lib/server/personaProfiles';
import { getPublicAnthropicSettings } from '@/lib/server/anthropicSecrets';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** List available persona senders (no few-shot payloads). */
export async function GET() {
  try {
    const profiles = listPersonaSummaries();
    const anthropic = getPublicAnthropicSettings();
    return NextResponse.json({
      profiles,
      profileCount: profiles.length,
      profilesPath: resolvePersonaProfilesPath(),
      anthropicConfigured: anthropic.configured,
      model: anthropic.model,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to load personas';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
