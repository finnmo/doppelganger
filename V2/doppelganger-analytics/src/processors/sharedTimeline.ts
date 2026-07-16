/**
 * Shared timeline / pinned life facts from high-signal messages with you.
 */

import type { Database as DatabaseType } from 'better-sqlite3';

const PLACE_HINTS = [
  'melbourne', 'sydney', 'brisbane', 'perth', 'bali', 'japan', 'thailand',
  'london', 'airport', 'uni', 'campus', 'gym', 'work', 'home', 'cafe',
];

const EVENTISH =
  /\b(trip|flew|flight|birthday|wedding|graduat|moved|moving|started|job|interview|concert|festival|holiday|vacation|hospital|surgery|pregnant|engaged|married|break.?up|broke up|new place|apartment|house)\b/i;

export interface TimelineFact {
  date: string;
  text: string;
  conversationId: string;
  score: number;
}

export interface SharedTimeline {
  withPerson: string;
  facts: TimelineFact[];
  /** Prompt-ready block */
  summary: string;
}

function scoreMessage(content: string): number {
  let score = 0;
  const lower = content.toLowerCase();
  if (EVENTISH.test(content)) score += 3;
  for (const p of PLACE_HINTS) {
    if (lower.includes(p)) score += 1.2;
  }
  if (/\b(remember|that time|when we|last year|years ago)\b/i.test(content)) score += 2;
  const words = content.trim().split(/\s+/).length;
  if (words >= 8 && words <= 60) score += 1;
  if (words < 4) score -= 2;
  if (/sent \d+ photo|shared a link/i.test(content)) score -= 5;
  return score;
}

export function buildSharedTimeline(
  db: DatabaseType,
  personaSender: string,
  selfSender: string,
  limit = 12
): SharedTimeline | null {
  if (!personaSender || !selfSender || personaSender === selfSender) return null;

  const rows = db
    .prepare(
      `
    SELECT conversation_id, content, timestamp_ms
    FROM messages
    WHERE sender = ?
      AND is_system = 0
      AND content IS NOT NULL
      AND TRIM(content) != ''
      AND conversation_id IN (
        SELECT conversation_id FROM messages WHERE sender = ?
        INTERSECT
        SELECT conversation_id FROM messages WHERE sender = ?
      )
    ORDER BY timestamp_ms DESC
    LIMIT 4000
  `
    )
    .all(personaSender, personaSender, selfSender) as Array<{
    conversation_id: string;
    content: string;
    timestamp_ms: number;
  }>;

  if (rows.length < 20) return null;

  const scored = rows
    .map((r) => ({
      date: new Date(r.timestamp_ms).toISOString().slice(0, 10),
      text: r.content.trim().slice(0, 180),
      conversationId: r.conversation_id,
      score: scoreMessage(r.content),
    }))
    .filter((f) => f.score >= 3.5)
    .sort((a, b) => b.score - a.score || b.date.localeCompare(a.date));

  const facts: TimelineFact[] = [];
  const seen = new Set<string>();
  for (const f of scored) {
    const key = f.text.toLowerCase().slice(0, 60);
    if (seen.has(key)) continue;
    seen.add(key);
    facts.push(f);
    if (facts.length >= limit) break;
  }

  if (facts.length < 3) return null;

  // Chronological for the prompt
  const chrono = [...facts].sort((a, b) => a.date.localeCompare(b.date));
  const summary = [
    `Shared timeline with ${selfSender} (pinned life facts — stay consistent, do not invent conflicting events):`,
    ...chrono.map((f, i) => `${i + 1}. [${f.date}] ${personaSender}: ${f.text}`),
  ].join('\n');

  return {
    withPerson: selfSender,
    facts: chrono,
    summary,
  };
}
