// Builds per-sender "style profiles" from existing analytics tables.
// These profiles feed a future LLM persona layer — they do not generate replies.

import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';
import { buildRelationshipCard, extractWithYouFewShot, inferSelfSender, type RelationshipCard } from './relationshipCard.js';
import { buildConversationVoices, buildPlatformVoices, type ConversationVoice, type PlatformVoice } from './conversationVoice.js';
import { buildBubbleHabits, type BubbleHabits } from './bubbleHabits.js';
import { buildTurnTakingHabits, type TurnTakingHabits } from './turnTakingHabits.js';
import { buildSharedTimeline, type SharedTimeline } from './sharedTimeline.js';
import { extractReplyPairs } from './replyTurns.js';
import { STOP_WORDS, buildConversationNameBlocklist } from '../utils/messageFilters.js';

interface SenderRow {
  sender: string;
  message_count: number;
  avg_word_count: number;
  avg_emoji_count: number;
}

/** Extract cross-sender reply pairs for few-shot LLM prompting (full reply turns). */
function extractFewShotForSender(
  db: import('better-sqlite3').Database,
  sender: string,
  limit: number
): PersonaStyleProfile['fewShotExamples'] {
  const rows = db.prepare(`
    SELECT conversation_id, sender, content, timestamp_ms, source, is_system
    FROM messages
    WHERE content IS NOT NULL AND TRIM(content) != ''
      AND conversation_id IN (
        SELECT DISTINCT conversation_id FROM messages WHERE sender = ?
      )
    ORDER BY conversation_id, timestamp_ms
  `).all(sender) as Array<{
    conversation_id: string;
    sender: string;
    content: string;
    timestamp_ms: number;
    source: string;
    is_system: number;
  }>;

  return extractReplyPairs(rows, sender, { limit, maxReplyChars: 550 }).map((p) => ({
    context: p.context,
    reply: p.reply,
    conversationId: p.conversationId,
    source: p.source,
  }));
}

export interface PersonaStyleProfile {
  sender: string;
  messageCount: number;
  vocabulary: {
    topWords: Array<{ word: string; count: number }>;
    avgWordsPerMessage: number;
    avgEmojiPerMessage: number;
  };
  sentiment: {
    avgCompound: number;
    positiveRatio: number;
    negativeRatio: number;
  };
  responsiveness: {
    medianReplyMs: number | null;
    p90ReplyMs: number | null;
    label: string;
  };
  starters: {
    conversationStarts: number;
    startRatio: number;
  };
  /** Short natural-language summary for LLM system prompts. */
  styleSummary: string;
  /** Real (context → their reply) pairs for few-shot prompting (~85% path). */
  fewShotExamples: Array<{
    context: string;
    reply: string;
    conversationId: string;
    source: string;
    withYou?: boolean;
  }>;
  /** Reply pairs specifically when responding to the account holder. */
  withYouFewShotExamples?: Array<{
    context: string;
    reply: string;
    conversationId: string;
    source: string;
    withYou: true;
  }>;
  /** Platforms this sender appears on. */
  sources: string[];
  /** How they relate to / text the inferred account holder ("you"). */
  relationshipCard?: RelationshipCard | null;
  /** Style notes scoped to specific chats (DM vs group). */
  conversationVoices?: ConversationVoice[];
  /** Style notes scoped by messaging platform (Instagram vs WhatsApp, etc.). */
  platformVoices?: PlatformVoice[];
  /** Double-text / multi-bubble habits. */
  bubbleHabits?: BubbleHabits | null;
  /** Question-back / expand-compress / latency habits. */
  turnTakingHabits?: TurnTakingHabits | null;
  /** Pinned shared life facts with you. */
  sharedTimeline?: SharedTimeline | null;
}

function latencyLabel(ms: number | null): string {
  if (ms === null) return 'unknown';
  if (ms < 30_000) return 'very fast (<30s)';
  if (ms < 300_000) return 'quick (under 5m)';
  if (ms < 3_600_000) return 'moderate (minutes to an hour)';
  return 'slow (often hours+)';
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * p));
  return sorted[idx];
}

export async function computePersonaMetrics(): Promise<void> {
  progressReporter.start('Computing persona style profiles...');
  const db = await getDb();

  try {
    const senders = db.prepare(`
      SELECT
        m.sender,
        COUNT(*) AS message_count,
        AVG(tm.word_count) AS avg_word_count,
        AVG(tm.emoji_count) AS avg_emoji_count
      FROM messages m
      LEFT JOIN text_metrics tm ON tm.message_id = m.id
      WHERE m.is_system = 0 AND m.content IS NOT NULL AND TRIM(m.content) != ''
      GROUP BY m.sender
      HAVING message_count >= 5
      ORDER BY message_count DESC
    `).all() as SenderRow[];

    if (senders.length === 0) {
      writeDashData('personaProfiles.json', { profiles: [], note: 'Not enough messages to build profiles.' });
      progressReporter.success('Persona profiles skipped (insufficient data)');
      return;
    }

    const sentimentBySender = db.prepare(`
      SELECT
        m.sender,
        AVG(s.compound) AS avg_compound,
        SUM(CASE WHEN s.compound > 0.05 THEN 1 ELSE 0 END) AS positive_count,
        SUM(CASE WHEN s.compound < -0.05 THEN 1 ELSE 0 END) AS negative_count,
        COUNT(*) AS n
      FROM messages m
      JOIN sentiment s ON s.message_id = m.id
      WHERE m.is_system = 0
      GROUP BY m.sender
    `).all() as Array<{
      sender: string;
      avg_compound: number;
      positive_count: number;
      negative_count: number;
      n: number;
    }>;

    const sentimentMap = new Map(sentimentBySender.map(r => [r.sender, r]));

    const latencyRows = db.prepare(`
      SELECT m.sender, rt.latency_ms
      FROM response_times rt
      JOIN messages m ON m.id = rt.to_message_id
      WHERE rt.latency_ms > 0
    `).all() as Array<{ sender: string; latency_ms: number }>;

    const latencyBySender = new Map<string, number[]>();
    for (const row of latencyRows) {
      const list = latencyBySender.get(row.sender) ?? [];
      list.push(row.latency_ms);
      latencyBySender.set(row.sender, list);
    }

    const starterRows = db.prepare(`
      SELECT starter AS sender, COUNT(*) AS starts
      FROM (
        SELECT
          conversation_id,
          sender AS starter,
          timestamp_ms,
          ROW_NUMBER() OVER (PARTITION BY conversation_id, date(timestamp_ms / 1000, 'unixepoch', 'localtime') ORDER BY timestamp_ms) AS rn
        FROM messages
        WHERE is_system = 0
      )
      WHERE rn = 1
      GROUP BY starter
    `).all() as Array<{ sender: string; starts: number }>;
    const starterMap = new Map(starterRows.map(r => [r.sender, r.starts]));

    const selfSender = inferSelfSender(db);
    if (selfSender) {
      console.log(`  Inferred account holder (you): ${selfSender}`);
    }

    const participantRows = db.prepare(`
      SELECT DISTINCT sender, conversation_id
      FROM messages
      WHERE sender IS NOT NULL AND TRIM(sender) != ''
    `).all() as Array<{ sender: string; conversation_id: string }>;
    const nameBlocklist = buildConversationNameBlocklist(participantRows);

    const profiles: PersonaStyleProfile[] = [];

    let processed = 0;
    for (const sender of senders) {
      if (++processed % 25 === 0 || processed === senders.length) {
        console.log(`  Persona profiles: ${processed}/${senders.length} senders`);
      }
      const words = db.prepare(`
        SELECT m.content, m.conversation_id
        FROM messages m
        WHERE m.sender = ? AND m.is_system = 0 AND m.content IS NOT NULL
      `).all(sender.sender) as Array<{ content: string; conversation_id: string }>;

      const freq = new Map<string, number>();
      for (const row of words) {
        const blocked = nameBlocklist.get(row.conversation_id) ?? new Set<string>();
        for (const token of row.content.toLowerCase().match(/[a-z']{3,}/g) ?? []) {
          if (STOP_WORDS.has(token) || blocked.has(token)) continue;
          freq.set(token, (freq.get(token) ?? 0) + 1);
        }
      }
      const topWords = [...freq.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 25)
        .map(([word, count]) => ({ word, count }));

      const sent = sentimentMap.get(sender.sender);
      const latencies = (latencyBySender.get(sender.sender) ?? []).sort((a, b) => a - b);
      const median = latencies.length ? percentile(latencies, 0.5) : null;
      const p90 = latencies.length ? percentile(latencies, 0.9) : null;
      const starts = starterMap.get(sender.sender) ?? 0;
      const startRatio = sender.message_count > 0 ? starts / sender.message_count : 0;

      const avgWords = Math.round((sender.avg_word_count ?? 0) * 10) / 10;
      const avgEmoji = Math.round((sender.avg_emoji_count ?? 0) * 100) / 100;
      const topWordStr = topWords.slice(0, 8).map(w => w.word).join(', ') || 'n/a';

      const styleSummary = [
        `${sender.sender} averages ~${avgWords} words/message (many short acks, but also fuller multi-sentence texts when chatting).`,
        `Emoji rate ~${avgEmoji}/message.`,
        `Frequent words: ${topWordStr}.`,
        sent
          ? `Tone skews ${sent.avg_compound > 0.1 ? 'positive' : sent.avg_compound < -0.1 ? 'negative' : 'neutral'} (compound ${sent.avg_compound.toFixed(2)}).`
          : 'Sentiment data limited.',
        `Typical reply speed: ${latencyLabel(median)}.`,
        startRatio > 0.15
          ? 'Often starts conversations and keeps them going.'
          : 'More often responds than initiates, but still engages when in a thread.',
      ].join(' ');

      const fewShotExamples = extractFewShotForSender(db, sender.sender, 80);
      const withYouFewShotExamples =
        selfSender && selfSender !== sender.sender
          ? extractWithYouFewShot(db, sender.sender, selfSender, 40)
          : [];
      const sources = [
        ...new Set(
          (
            db.prepare(`
              SELECT DISTINCT source FROM messages WHERE sender = ?
            `).all(sender.sender) as Array<{ source: string }>
          ).map(r => r.source)
        )
      ];

      const relationshipCard =
        selfSender && selfSender !== sender.sender
          ? buildRelationshipCard(db, sender.sender, selfSender)
          : null;

      const conversationVoices = buildConversationVoices(db, sender.sender);
      const platformVoices = buildPlatformVoices(db, sender.sender);
      const bubbleHabits = buildBubbleHabits(db, sender.sender);
      const turnTakingHabits = buildTurnTakingHabits(db, sender.sender);
      const sharedTimeline =
        selfSender && selfSender !== sender.sender
          ? buildSharedTimeline(db, sender.sender, selfSender, 12)
          : null;

      profiles.push({
        sender: sender.sender,
        messageCount: sender.message_count,
        vocabulary: {
          topWords,
          avgWordsPerMessage: avgWords,
          avgEmojiPerMessage: avgEmoji
        },
        sentiment: {
          avgCompound: sent ? Math.round(sent.avg_compound * 1000) / 1000 : 0,
          positiveRatio: sent && sent.n ? sent.positive_count / sent.n : 0,
          negativeRatio: sent && sent.n ? sent.negative_count / sent.n : 0
        },
        responsiveness: {
          medianReplyMs: median,
          p90ReplyMs: p90,
          label: latencyLabel(median)
        },
        starters: {
          conversationStarts: starts,
          startRatio: Math.round(startRatio * 1000) / 1000
        },
        styleSummary,
        fewShotExamples,
        withYouFewShotExamples,
        sources,
        relationshipCard,
        conversationVoices,
        platformVoices,
        bubbleHabits,
        turnTakingHabits,
        sharedTimeline,
      });
    }

    writeDashData('personaProfiles.json', {
      generatedAt: new Date().toISOString(),
      profileCount: profiles.length,
      inferredSelf: selfSender,
      profiles,
      capabilityNote:
        'Each profile includes styleSummary + fewShotExamples + withYouFewShotExamples + relationshipCard (with-you register) + conversationVoices + platformVoices. ' +
        'See docs/PERSONA_SIMULATION.md.',
      accuracyTarget: '85%',
      nextStep: 'Evaluate with held-out replies; optional fine-tune for Tier 4.'
    });

    progressReporter.success(`Persona profiles exported for ${profiles.length} senders`);
  } finally {
    await closeDb(db);
  }
}
