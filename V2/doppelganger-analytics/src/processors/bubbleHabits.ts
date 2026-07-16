/**
 * How often a sender double-/triple-texts (multi-bubble turns).
 */

import type { Database as DatabaseType } from 'better-sqlite3';

/** Max gap between consecutive same-sender messages to count as one "turn". */
const BUBBLE_GAP_MS = 90_000;

export interface BubbleHabits {
  /** Average number of consecutive bubbles per turn */
  avgBubblesPerTurn: number;
  /** Share of turns that are 2+ bubbles */
  multiBubbleRate: number;
  /** Sample multi-bubble turns (joined) for prompting */
  sampleTurns: string[][];
  /** Prompt-ready note */
  styleSummary: string;
}

interface MsgRow {
  conversation_id: string;
  content: string;
  timestamp_ms: number;
}

export function buildBubbleHabits(db: DatabaseType, sender: string): BubbleHabits | null {
  const rows = db
    .prepare(
      `
    SELECT conversation_id, content, timestamp_ms
    FROM messages
    WHERE sender = ?
      AND is_system = 0
      AND content IS NOT NULL
      AND TRIM(content) != ''
    ORDER BY conversation_id, timestamp_ms
    LIMIT 8000
  `
    )
    .all(sender) as MsgRow[];

  if (rows.length < 30) return null;

  const turns: string[][] = [];
  let current: string[] = [];
  let prev: MsgRow | null = null;

  const flush = () => {
    if (current.length > 0) turns.push(current);
    current = [];
  };

  for (const row of rows) {
    if (
      prev &&
      prev.conversation_id === row.conversation_id &&
      row.timestamp_ms - prev.timestamp_ms <= BUBBLE_GAP_MS
    ) {
      current.push(row.content.trim());
    } else {
      flush();
      current = [row.content.trim()];
    }
    prev = row;
  }
  flush();

  if (turns.length < 10) return null;

  const sizes = turns.map((t) => t.length);
  const avgBubblesPerTurn =
    Math.round((sizes.reduce((a, b) => a + b, 0) / sizes.length) * 100) / 100;
  const multiBubbleRate =
    Math.round((sizes.filter((n) => n >= 2).length / sizes.length) * 1000) / 1000;

  const sampleTurns = turns
    .filter((t) => t.length >= 2 && t.every((b) => b.length >= 2 && b.length <= 160))
    .slice(0, 6);

  const styleSummary =
    multiBubbleRate >= 0.2
      ? `${sender} often sends ${avgBubblesPerTurn.toFixed(1)} short bubbles in a row (~${Math.round(multiBubbleRate * 100)}% of turns are multi-bubble). Prefer 1–3 separate text bubbles when they would, split with the delimiter <<<BUBBLE>>>.`
      : `${sender} usually sends one message at a time (multi-bubble ~${Math.round(multiBubbleRate * 100)}% of turns). Use a single bubble unless a follow-up clearly fits; if splitting, use <<<BUBBLE>>>.`;

  return {
    avgBubblesPerTurn,
    multiBubbleRate,
    sampleTurns,
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
