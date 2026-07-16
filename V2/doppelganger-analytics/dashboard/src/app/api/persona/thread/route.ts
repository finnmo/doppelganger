import { NextRequest, NextResponse } from 'next/server';
import { getAnalyticsDbReadonly } from '@/lib/server/analyticsDb';
import {
  loadConversationMessages,
  messagesToPersonaThread,
} from '@/lib/server/personaRetrieval';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * Seed Persona Chat with the real conversation thread.
 * Query: ?conversationId=&sender=
 */
export async function GET(request: NextRequest) {
  try {
    const conversationId = request.nextUrl.searchParams.get('conversationId')?.trim();
    const sender = request.nextUrl.searchParams.get('sender')?.trim();

    if (!conversationId || !sender) {
      return NextResponse.json(
        { error: 'conversationId and sender are required.' },
        { status: 400 }
      );
    }

    const db = getAnalyticsDbReadonly();
    if (!db) {
      return NextResponse.json({
        messages: [],
        note: 'Analytics database not found — thread seeding unavailable.',
      });
    }

    const limitParam = Number(request.nextUrl.searchParams.get('limit') ?? '40');
    const limit = Number.isFinite(limitParam) ? Math.min(80, Math.max(10, limitParam)) : 40;

    const stored = loadConversationMessages(db, conversationId, limit);
    const thread = messagesToPersonaThread(stored, sender);

    // Drop a trailing assistant message so the UI is ready for the user to type
    // (chat API requires last message to be user when sending).
    const messages = thread.map((t) => ({
      role: t.role,
      content: t.content,
      sender: t.sender,
      timestampMs: t.timestampMs,
      fromHistory: true as const,
    }));

    return NextResponse.json({
      conversationId,
      sender,
      messageCount: messages.length,
      messages,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to load thread';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
