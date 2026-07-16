/**
 * Turn-taking habits: questions back, double-text, mirror energy, etc.
 */

import type { Database as DatabaseType } from 'better-sqlite3';

const BUBBLE_GAP_MS = 90_000;

export interface TurnTakingHabits {
  questionRate: number;
  /** Share of their messages that follow their own message quickly (double-text) */
  doubleTextRate: number;
  /** Share of replies that are longer than the message they responded to */
  expandsOnPartnerRate: number;
  /** Share of replies shorter than partner message */
  compressesRate: number;
  avgReplyLatencyMs: number | null;
  styleSummary: string;
}

interface Row {
  conversation_id: string;
  sender: string;
  content: string;
  timestamp_ms: number;
  is_system: number;
}

function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export function buildTurnTakingHabits(
  db: DatabaseType,
  sender: string
): TurnTakingHabits | null {
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
    .all(sender) as Row[];

  if (rows.length < 40) return null;

  let theirMsgs = 0;
  let questions = 0;
  let doubleTexts = 0;
  let replyPairs = 0;
  let replyQuestions = 0;
  let expands = 0;
  let compresses = 0;
  const latencies: number[] = [];
  const followUpSamples: string[] = [];

  let prev: Row | null = null;
  for (const row of rows) {
    if (row.is_system) {
      prev = null;
      continue;
    }

    if (row.sender === sender) {
      theirMsgs += 1;
      if (/\?/.test(row.content)) questions += 1;

      if (
        prev &&
        prev.conversation_id === row.conversation_id &&
        prev.sender === sender &&
        row.timestamp_ms - prev.timestamp_ms <= BUBBLE_GAP_MS
      ) {
        doubleTexts += 1;
      }

      if (
        prev &&
        prev.conversation_id === row.conversation_id &&
        prev.sender !== sender
      ) {
        replyPairs += 1;
        const pw = wordCount(prev.content);
        const rw = wordCount(row.content);
        if (rw > pw * 1.15) expands += 1;
        if (rw < pw * 0.7) compresses += 1;
        if (/\?/.test(row.content)) {
          replyQuestions += 1;
          if (followUpSamples.length < 8 && rw >= 3 && row.content.length <= 140) {
            followUpSamples.push(row.content.trim());
          }
        }
        const lag = row.timestamp_ms - prev.timestamp_ms;
        if (lag > 0 && lag < 24 * 3600_000) latencies.push(lag);
      }
    }
    prev = row;
  }

  if (theirMsgs < 20) return null;

  const questionRate = Math.round((questions / theirMsgs) * 1000) / 1000;
  const questionRateOnReplies =
    replyPairs > 0 ? Math.round((replyQuestions / replyPairs) * 1000) / 1000 : questionRate;
  const doubleTextRate = Math.round((doubleTexts / theirMsgs) * 1000) / 1000;
  const expandsOnPartnerRate =
    replyPairs > 0 ? Math.round((expands / replyPairs) * 1000) / 1000 : 0;
  const compressesRate =
    replyPairs > 0 ? Math.round((compresses / replyPairs) * 1000) / 1000 : 0;
  const avgReplyLatencyMs =
    latencies.length > 0
      ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)
      : null;

  const bits: string[] = [];
  bits.push(`Turn-taking habits for ${sender}:`);
  // Use reply-turn question rate — raw message rate is diluted by acks/bubbles
  bits.push(
    `when replying to someone, ~${Math.round(questionRateOnReplies * 100)}% of their reply turns include a question.`
  );
  if (questionRateOnReplies >= 0.25) {
    bits.push(
      'In casual chat they fairly often tack on a follow-up question — do that when it fits this message, not every turn.'
    );
  } else if (questionRateOnReplies >= 0.12 || followUpSamples.length >= 2) {
    bits.push(
      'They sometimes ask a follow-up (~' +
        Math.round(questionRateOnReplies * 100) +
        '% of reply turns) — only when it fits, not as a default.'
    );
  } else {
    bits.push(
      'They are not frequent question-askers on reply turns — keep the thread going with a reaction or detail when needed, without forcing a question.'
    );
  }
  if (followUpSamples.length) {
    bits.push(
      `Example follow-ups they have sent: ${followUpSamples
        .slice(0, 4)
        .map((s) => `“${s}”`)
        .join(' · ')}`
    );
  }
  bits.push(
    `double-texts ~${Math.round(doubleTextRate * 100)}% of the time` +
      (doubleTextRate >= 0.15
        ? ' — split only when the reply has separate thoughts; otherwise one bubble.'
        : ' — usually one bubble unless a second thought clearly belongs alone.')
  );
  if (replyPairs >= 10) {
    if (expandsOnPartnerRate >= 0.35) {
      bits.push('Often expands on what the other person said (longer reply than their message).');
    } else if (compressesRate >= 0.4) {
      bits.push('Often replies shorter than the other person — punchy, not essays.');
    } else {
      bits.push("Reply length usually tracks the other person's energy.");
    }
  }
  if (avgReplyLatencyMs != null && avgReplyLatencyMs < 60_000) {
    bits.push('Typically replies quickly in active threads.');
  }

  return {
    questionRate: questionRateOnReplies,
    doubleTextRate,
    expandsOnPartnerRate,
    compressesRate,
    avgReplyLatencyMs,
    styleSummary: bits.join(' '),
  };
}
