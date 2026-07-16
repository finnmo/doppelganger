/**
 * Group consecutive same-sender messages into texting "turns"
 * (double-/triple-texts within a short gap).
 */

export const BUBBLE_GAP_MS = 90_000;
export const BUBBLE_DELIMITER = ' <<<BUBBLE>>> ';

export interface MessageLike {
  conversation_id: string;
  sender: string;
  content: string;
  timestamp_ms: number;
  is_system?: number;
  source?: string;
}

export interface ReplyTurn {
  conversationId: string;
  sender: string;
  bubbles: string[];
  timestampMs: number;
  source: string;
}

const MEDIA_RE = /sent \d+ photo|sent a voice message|shared a link/i;

export function isMediaPlaceholder(text: string): boolean {
  return MEDIA_RE.test(text);
}

/** Collapse consecutive same-sender messages (within gap) into turns. */
export function groupIntoReplyTurns(rows: MessageLike[]): ReplyTurn[] {
  const turns: ReplyTurn[] = [];
  let current: ReplyTurn | null = null;

  const flush = () => {
    if (current && current.bubbles.length > 0) turns.push(current);
    current = null;
  };

  for (const row of rows) {
    if (row.is_system) {
      flush();
      continue;
    }
    const content = row.content.trim();
    if (!content || isMediaPlaceholder(content)) {
      flush();
      continue;
    }

    if (
      current &&
      current.conversationId === row.conversation_id &&
      current.sender === row.sender &&
      row.timestamp_ms - current.timestampMs <= BUBBLE_GAP_MS
    ) {
      current.bubbles.push(content);
      current.timestampMs = row.timestamp_ms;
    } else {
      flush();
      current = {
        conversationId: row.conversation_id,
        sender: row.sender,
        bubbles: [content],
        timestampMs: row.timestamp_ms,
        source: row.source ?? 'unknown',
      };
    }
  }
  flush();
  return turns;
}

export function joinBubbles(bubbles: string[], maxChars = 500): string {
  const joined = bubbles.join(BUBBLE_DELIMITER);
  if (joined.length <= maxChars) return joined;
  // Prefer keeping whole leading bubbles rather than mid-bubble truncation
  let out = bubbles[0] ?? '';
  for (let i = 1; i < bubbles.length; i++) {
    const next = `${out}${BUBBLE_DELIMITER}${bubbles[i]}`;
    if (next.length > maxChars) break;
    out = next;
  }
  return out.length <= maxChars ? out : `${out.slice(0, maxChars - 1)}…`;
}

export interface FewShotPair {
  context: string;
  reply: string;
  conversationId: string;
  source: string;
  bubbleCount: number;
}

/**
 * Partner turn → persona turn pairs. Multi-bubble persona replies are joined
 * with <<<BUBBLE>>> so few-shots teach when they actually split.
 */
export function extractReplyPairs(
  rows: MessageLike[],
  personaSender: string,
  options?: {
    limit?: number;
    /** If set, only pairs where the preceding turn is from this sender. */
    contextSender?: string;
    maxContextChars?: number;
    maxReplyChars?: number;
  }
): FewShotPair[] {
  const limit = options?.limit ?? 80;
  const maxContext = options?.maxContextChars ?? 400;
  const maxReply = options?.maxReplyChars ?? 500;
  const turns = groupIntoReplyTurns(rows);
  const pairs: FewShotPair[] = [];

  for (let i = 1; i < turns.length; i++) {
    const prev = turns[i - 1];
    const cur = turns[i];
    if (cur.conversationId !== prev.conversationId) continue;
    if (cur.sender !== personaSender) continue;
    if (prev.sender === personaSender) continue;
    if (options?.contextSender && prev.sender !== options.contextSender) continue;

    const context = prev.bubbles.join(' ').slice(0, maxContext);
    if (context.length < 2) continue;

    pairs.push({
      context,
      reply: joinBubbles(cur.bubbles, maxReply),
      conversationId: cur.conversationId,
      source: cur.source,
      bubbleCount: cur.bubbles.length,
    });
    if (pairs.length >= limit * 3) break;
  }

  return diversifyPairs(pairs, limit);
}

function diversifyPairs(pairs: FewShotPair[], limit: number): FewShotPair[] {
  const byConv = new Map<string, FewShotPair[]>();
  for (const p of pairs) {
    const list = byConv.get(p.conversationId) ?? [];
    list.push(p);
    byConv.set(p.conversationId, list);
  }
  const diversified: FewShotPair[] = [];
  const queues = [...byConv.values()];
  let i = 0;
  while (diversified.length < limit && queues.some((q) => q.length > 0)) {
    const q = queues[i % queues.length];
    if (q.length > 0) diversified.push(q.shift()!);
    i++;
  }
  return diversified;
}
