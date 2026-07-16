import { NextRequest, NextResponse } from 'next/server';
import { findPersonaProfile } from '@/lib/server/personaProfiles';
import { buildAnthropicPersonaRequest, type ChatTurn } from '@/lib/server/buildPersonaPrompt';
import { callAnthropicMessages } from '@/lib/server/anthropicClient';
import { getPublicAnthropicSettings } from '@/lib/server/anthropicSecrets';
import { getAnalyticsDbReadonly } from '@/lib/server/analyticsDb';
import { retrievePersonaMemories } from '@/lib/server/personaRetrieval';
import { splitReplyBubbles } from '@/lib/server/splitReplyBubbles';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_HISTORY = 60;
const MAX_USER_CHARS = 2000;

function sanitizeHistory(raw: unknown): ChatTurn[] | { error: string } {
  if (!Array.isArray(raw)) return { error: 'messages must be an array.' };
  if (raw.length === 0) return { error: 'messages cannot be empty.' };
  if (raw.length > MAX_HISTORY) {
    return { error: `Too many messages (max ${MAX_HISTORY}).` };
  }

  const out: ChatTurn[] = [];
  for (const item of raw) {
    if (!item || typeof item !== 'object') return { error: 'Invalid message entry.' };
    const role = (item as { role?: string }).role;
    const content = (item as { content?: string }).content;
    if (role !== 'user' && role !== 'assistant') {
      return { error: 'Each message role must be "user" or "assistant".' };
    }
    if (typeof content !== 'string' || !content.trim()) {
      return { error: 'Each message must have non-empty content.' };
    }
    if (content.length > MAX_USER_CHARS) {
      return { error: `Message too long (max ${MAX_USER_CHARS} characters).` };
    }
    out.push({ role, content: content.trim() });
  }

  if (out[out.length - 1].role !== 'user') {
    return { error: 'The last message must be from the user.' };
  }

  return out;
}

/** Only scope RAG/voice to a chat the persona actually participates in. */
function conversationIncludesSender(
  conversationId: string | undefined,
  personaSender: string
): string | undefined {
  if (!conversationId) return undefined;
  const db = getAnalyticsDbReadonly();
  if (!db) return undefined;
  const row = db
    .prepare(
      `
    SELECT 1 AS ok
    FROM messages
    WHERE conversation_id = ?
      AND sender = ?
      AND is_system = 0
    LIMIT 1
  `
    )
    .get(conversationId, personaSender) as { ok: number } | undefined;
  return row ? conversationId : undefined;
}

/**
 * Generate a persona reply with few-shot style + RAG memories + thread context.
 * Body: { sender: string, messages: ChatTurn[], conversationId?: string }
 */
export async function POST(request: NextRequest) {
  try {
    const settings = getPublicAnthropicSettings();
    if (!settings.configured) {
      return NextResponse.json(
        { error: 'Claude API key is not configured. Open API key settings first.' },
        { status: 401 }
      );
    }

    const body = await request.json().catch(() => null);
    if (!body || typeof body !== 'object') {
      return NextResponse.json({ error: 'Invalid JSON body.' }, { status: 400 });
    }

    const sender = typeof body.sender === 'string' ? body.sender.trim() : '';
    if (!sender) {
      return NextResponse.json({ error: 'sender is required.' }, { status: 400 });
    }

    const conversationId =
      typeof body.conversationId === 'string' && body.conversationId.trim()
        ? body.conversationId.trim()
        : undefined;

    const history = sanitizeHistory(body.messages);
    if ('error' in history) {
      return NextResponse.json({ error: history.error }, { status: 400 });
    }

    const profile = findPersonaProfile(sender);
    if (!profile) {
      return NextResponse.json(
        {
          error:
            `No persona profile for "${sender}". Run generate-metrics so personaProfiles.json is created, then reopen the dashboard.`,
        },
        { status: 404 }
      );
    }

    const scopedConversationId = conversationIncludesSender(conversationId, sender);
    const lastUser = [...history].reverse().find((m) => m.role === 'user');
    const query = lastUser?.content ?? '';

    let memories: Awaited<ReturnType<typeof retrievePersonaMemories>>['memories'] = [];
    let retrievalMode: Awaited<ReturnType<typeof retrievePersonaMemories>>['mode'] = 'none';
    const db = getAnalyticsDbReadonly();
    if (db && query) {
      const retrieved = await retrievePersonaMemories(db, {
        personaSender: sender,
        query,
        conversationId: scopedConversationId,
        preferConversationIds: profile.relationshipCard?.sharedConversationIds,
        limit: 8,
      });
      memories = retrieved.memories;
      retrievalMode = retrieved.mode;
    }

    const { system, messages, memoryCount, exampleCount } = buildAnthropicPersonaRequest(
      profile,
      history,
      {
        conversationId: scopedConversationId,
        maxExamples: 12,
        memories,
      }
    );

    const result = await callAnthropicMessages({ system, messages, maxTokens: 700 });
    const bubbles = splitReplyBubbles(result.text);

    return NextResponse.json({
      reply: bubbles.join('\n\n'),
      bubbles,
      sender: profile.sender,
      model: result.model,
      exampleCount,
      memoryCount,
      retrievalMode,
      usage: {
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
      },
    });
  } catch (err) {
    const message = (err instanceof Error ? err.message : 'Persona chat failed').replace(
      /sk-ant-[a-zA-Z0-9_-]+/g,
      '[redacted]'
    );
    const status = message.includes('not configured') ? 401 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
