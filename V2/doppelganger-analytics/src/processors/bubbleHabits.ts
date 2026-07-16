/**
 * How often a sender double-/triple-texts (multi-bubble turns),
 * and when those splits tend to happen.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import {
  BUBBLE_DELIMITER,
  BUBBLE_GAP_MS,
  groupIntoReplyTurns,
  type MessageLike,
} from './replyTurns.js';

export interface BubbleContextualSample {
  /** Partner message(s) right before the multi-bubble turn */
  context: string;
  bubbles: string[];
}

export interface BubbleHabits {
  /** Average number of consecutive bubbles per turn */
  avgBubblesPerTurn: number;
  /** Median bubble count among multi-bubble turns only */
  medianBubblesWhenMulti: number;
  /** Share of turns that are 2+ bubbles */
  multiBubbleRate: number;
  /** Sample multi-bubble turns (joined) for prompting */
  sampleTurns: string[][];
  /** Partner→multi-bubble pairs so the model sees triggers, not orphan stacks */
  contextualSamples: BubbleContextualSample[];
  /** Prompt-ready note — conditional, not a quota */
  styleSummary: string;
}

interface MsgRow {
  conversation_id: string;
  sender: string;
  content: string;
  timestamp_ms: number;
  is_system: number;
}

function median(nums: number[]): number {
  if (nums.length === 0) return 1;
  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? Math.round(((sorted[mid - 1] + sorted[mid]) / 2) * 10) / 10
    : sorted[mid];
}

function describeTriggers(samples: BubbleContextualSample[]): string {
  if (samples.length === 0) return '';
  const bits: string[] = [];
  let questions = 0;
  let planning = 0;
  let reaction = 0;
  for (const s of samples) {
    const joined = s.bubbles.join(' ');
    if (/\?/.test(joined)) questions += 1;
    if (/\b(time|when|where|meet|friday|saturday|sunday|tomorrow|tonight|free|plans?)\b/i.test(joined)) {
      planning += 1;
    }
    if (/^(omg|wait|haha|lol|lmao|no way|dude|bro)\b/i.test(s.bubbles[0] ?? '')) {
      reaction += 1;
    }
  }
  const n = samples.length;
  if (questions / n >= 0.25) bits.push('follow-up questions');
  if (planning / n >= 0.2) bits.push('plans/logistics');
  if (reaction / n >= 0.15) bits.push('reactions + a second thought');
  if (bits.length === 0) return 'separate thoughts that do not fit in one text';
  return bits.join(', ');
}

export function buildBubbleHabits(db: DatabaseType, sender: string): BubbleHabits | null {
  const rows = db
    .prepare(
      `
    SELECT conversation_id, sender, content, timestamp_ms, is_system
    FROM messages
    WHERE content IS NOT NULL AND TRIM(content) != ''
      AND conversation_id IN (
        SELECT DISTINCT conversation_id FROM messages WHERE sender = ?
      )
    ORDER BY conversation_id, timestamp_ms
    LIMIT 12000
  `
    )
    .all(sender) as MsgRow[];

  if (rows.length < 30) return null;

  const turns = groupIntoReplyTurns(rows as MessageLike[]);
  const theirTurns = turns.filter((t) => t.sender === sender);
  if (theirTurns.length < 10) return null;

  const sizes = theirTurns.map((t) => t.bubbles.length);
  const avgBubblesPerTurn =
    Math.round((sizes.reduce((a, b) => a + b, 0) / sizes.length) * 100) / 100;
  const multiSizes = sizes.filter((n) => n >= 2);
  const multiBubbleRate = Math.round((multiSizes.length / sizes.length) * 1000) / 1000;
  const medianBubblesWhenMulti = median(multiSizes);

  const sampleTurns = theirTurns
    .filter(
      (t) =>
        t.bubbles.length >= 2 &&
        t.bubbles.length <= 5 &&
        t.bubbles.every((b) => b.length >= 2 && b.length <= 160)
    )
    .slice(0, 6)
    .map((t) => t.bubbles);

  const contextualSamples: BubbleContextualSample[] = [];
  for (let i = 1; i < turns.length && contextualSamples.length < 8; i++) {
    const prev = turns[i - 1];
    const cur = turns[i];
    if (cur.conversationId !== prev.conversationId) continue;
    if (cur.sender !== sender || prev.sender === sender) continue;
    if (cur.bubbles.length < 2 || cur.bubbles.length > 5) continue;
    if (!cur.bubbles.every((b) => b.length >= 2 && b.length <= 160)) continue;
    const context = prev.bubbles.join(' ').trim();
    if (context.length < 2 || context.length > 280) continue;
    contextualSamples.push({ context, bubbles: cur.bubbles });
  }

  const pct = Math.round(multiBubbleRate * 100);
  const singlePct = 100 - pct;
  const typical = Math.max(2, Math.min(4, Math.round(medianBubblesWhenMulti || 2)));
  const triggers = describeTriggers(contextualSamples);

  let styleSummary: string;
  if (multiBubbleRate >= 0.2) {
    styleSummary =
      `${sender} splits into multiple texts on ~${pct}% of turns` +
      ` (typical size when they do: ~${typical} bubbles; overall avg ~${avgBubblesPerTurn.toFixed(1)}). ` +
      `Use <<<BUBBLE>>> only when this reply naturally has separate chunks` +
      (triggers ? ` — often for ${triggers}` : '') +
      `. About ~${singlePct}% of their turns are a single bubble — quick acks, simple yes/no, and short answers stay one text. ` +
      `Never pad with extra bubbles just to hit a rate; mirror the few-shot examples for this message.`;
  } else {
    styleSummary =
      `${sender} usually sends one message at a time (multi-bubble ~${pct}% of turns). ` +
      `Default to a single bubble. Only split with <<<BUBBLE>>> when a second thought clearly belongs as its own text.`;
  }

  return {
    avgBubblesPerTurn,
    medianBubblesWhenMulti,
    multiBubbleRate,
    sampleTurns,
    contextualSamples,
    styleSummary,
  };
}

/** Split a model reply into text bubbles. */
export function splitReplyBubbles(text: string): string[] {
  const raw = text.trim();
  if (!raw) return [];

  if (raw.includes('<<<BUBBLE>>>')) {
    return raw
      .split('<<<BUBBLE>>>')
      .map((b) => b.trim())
      .filter(Boolean)
      .slice(0, 5);
  }

  // Fallback: blank-line separated short paragraphs that look like texts
  const parts = raw
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter(Boolean);
  if (parts.length >= 2 && parts.length <= 4 && parts.every((p) => p.length <= 220 && !p.includes('\n\n'))) {
    return parts;
  }

  return [raw];
}

export { BUBBLE_GAP_MS, BUBBLE_DELIMITER };
