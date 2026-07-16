/**
 * Per-conversation and per-platform voice profiles:
 * same person texts differently in a DM vs a group, and on Instagram vs WhatsApp.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import { sourceLabel } from '../utils/platformSource.js';

const MIN_MESSAGES = 25;
const MAX_VOICES_PER_SENDER = 20;
const MIN_PLATFORM_MESSAGES = 40;

export interface ConversationVoice {
  conversationId: string;
  source: string;
  participantCount: number;
  messageCount: number;
  avgWordsPerMessage: number;
  avgEmojiPerMessage: number;
  /** short / medium / long */
  lengthLabel: string;
  /** dm | small_group | group */
  chatType: 'dm' | 'small_group' | 'group';
  /** One-line voice note for the system prompt */
  styleSummary: string;
}

export interface PlatformVoice {
  source: string;
  messageCount: number;
  conversationCount: number;
  avgWordsPerMessage: number;
  avgEmojiPerMessage: number;
  lengthLabel: string;
  /** Share of their messages on this platform that are in 1:1 chats */
  dmShare: number;
  styleSummary: string;
}

function lengthLabel(avgWords: number): string {
  if (avgWords < 4) return 'very short';
  if (avgWords < 10) return 'short';
  if (avgWords < 25) return 'medium';
  return 'long';
}

function chatType(participantCount: number): ConversationVoice['chatType'] {
  if (participantCount <= 2) return 'dm';
  if (participantCount <= 5) return 'small_group';
  return 'group';
}

function chatTypeLabel(t: ConversationVoice['chatType']): string {
  if (t === 'dm') return '1:1 DM';
  if (t === 'small_group') return 'small group chat';
  return 'larger group chat';
}

/**
 * Build conversation-scoped voice notes for a sender (top chats by their message volume).
 */
export function buildConversationVoices(
  db: DatabaseType,
  sender: string
): ConversationVoice[] {
  const rows = db
    .prepare(
      `
    SELECT
      m.conversation_id AS conversationId,
      MAX(m.source) AS source,
      COUNT(*) AS messageCount,
      AVG(COALESCE(tm.word_count, 0)) AS avgWords,
      AVG(COALESCE(tm.emoji_count, 0)) AS avgEmoji,
      (
        SELECT COUNT(DISTINCT m2.sender)
        FROM messages m2
        WHERE m2.conversation_id = m.conversation_id
          AND m2.is_system = 0
      ) AS participantCount
    FROM messages m
    LEFT JOIN text_metrics tm ON tm.message_id = m.id
    WHERE m.sender = ?
      AND m.is_system = 0
      AND m.content IS NOT NULL
      AND TRIM(m.content) != ''
    GROUP BY m.conversation_id
    HAVING messageCount >= ?
    ORDER BY messageCount DESC
    LIMIT ?
  `
    )
    .all(sender, MIN_MESSAGES, MAX_VOICES_PER_SENDER) as Array<{
    conversationId: string;
    source: string;
    messageCount: number;
    avgWords: number;
    avgEmoji: number;
    participantCount: number;
  }>;

  return rows.map((row) => {
    const avgWords = Math.round((row.avgWords ?? 0) * 10) / 10;
    const avgEmoji = Math.round((row.avgEmoji ?? 0) * 100) / 100;
    const len = lengthLabel(avgWords);
    const type = chatType(row.participantCount || 2);
    const emojiBit =
      avgEmoji >= 0.4
        ? 'uses emoji often here'
        : avgEmoji < 0.05
          ? 'rarely uses emoji here'
          : 'uses emoji sparingly here';

    const styleSummary = [
      `In this ${chatTypeLabel(type)} (${row.participantCount} people),`,
      `${sender} tends to send ${len} messages (~${avgWords} words)`,
      `and ${emojiBit}.`,
      type === 'group' || type === 'small_group'
        ? 'Voice may be more public/performative than in private DMs.'
        : 'Voice is their private 1:1 register with this person.',
    ].join(' ');

    return {
      conversationId: row.conversationId,
      source: row.source || 'unknown',
      participantCount: row.participantCount || 2,
      messageCount: row.messageCount,
      avgWordsPerMessage: avgWords,
      avgEmojiPerMessage: avgEmoji,
      lengthLabel: len,
      chatType: type,
      styleSummary,
    };
  });
}

/**
 * Aggregate how this sender texts on each messaging platform.
 */
export function buildPlatformVoices(db: DatabaseType, sender: string): PlatformVoice[] {
  const rows = db
    .prepare(
      `
    SELECT
      m.source AS source,
      COUNT(*) AS messageCount,
      COUNT(DISTINCT m.conversation_id) AS conversationCount,
      AVG(COALESCE(tm.word_count, 0)) AS avgWords,
      AVG(COALESCE(tm.emoji_count, 0)) AS avgEmoji
    FROM messages m
    LEFT JOIN text_metrics tm ON tm.message_id = m.id
    WHERE m.sender = ?
      AND m.is_system = 0
      AND m.content IS NOT NULL
      AND TRIM(m.content) != ''
      AND m.source IS NOT NULL
      AND TRIM(m.source) != ''
    GROUP BY m.source
    HAVING messageCount >= ?
    ORDER BY messageCount DESC
  `
    )
    .all(sender, MIN_PLATFORM_MESSAGES) as Array<{
    source: string;
    messageCount: number;
    conversationCount: number;
    avgWords: number;
    avgEmoji: number;
  }>;

  if (rows.length === 0) return [];

  const dmShareStmt = db.prepare(`
    SELECT COUNT(*) AS n
    FROM messages m
    WHERE m.sender = ?
      AND m.source = ?
      AND m.is_system = 0
      AND m.content IS NOT NULL
      AND TRIM(m.content) != ''
      AND (
        SELECT COUNT(DISTINCT m2.sender)
        FROM messages m2
        WHERE m2.conversation_id = m.conversation_id
          AND m2.is_system = 0
      ) <= 2
  `);

  return rows.map((row) => {
    const avgWords = Math.round((row.avgWords ?? 0) * 10) / 10;
    const avgEmoji = Math.round((row.avgEmoji ?? 0) * 100) / 100;
    const len = lengthLabel(avgWords);
    const dmCount = (
      dmShareStmt.get(sender, row.source) as { n: number } | undefined
    )?.n ?? 0;
    const dmShare =
      row.messageCount > 0 ? Math.round((dmCount / row.messageCount) * 1000) / 1000 : 0;

    const emojiBit =
      avgEmoji >= 0.4
        ? 'uses emoji often'
        : avgEmoji < 0.05
          ? 'rarely uses emoji'
          : 'uses emoji sparingly';

    const platform = sourceLabel(row.source);
    const styleSummary = [
      `On ${platform}, ${sender} tends to send ${len} texts (~${avgWords} words/message)`,
      `and ${emojiBit}`,
      `(~${row.messageCount} messages across ${row.conversationCount} chats`,
      dmShare >= 0.6
        ? '— mostly 1:1).'
        : dmShare <= 0.35
          ? '— often groups).'
          : ').',
      `Prefer this ${platform} register over how they sound on other apps when the open chat is ${platform}.`,
    ].join(' ');

    return {
      source: row.source,
      messageCount: row.messageCount,
      conversationCount: row.conversationCount,
      avgWordsPerMessage: avgWords,
      avgEmojiPerMessage: avgEmoji,
      lengthLabel: len,
      dmShare,
      styleSummary,
    };
  });
}

export function findConversationVoice(
  voices: ConversationVoice[] | undefined,
  conversationId: string | undefined
): ConversationVoice | null {
  if (!voices?.length || !conversationId) return null;
  return voices.find((v) => v.conversationId === conversationId) ?? null;
}

export function findPlatformVoice(
  voices: PlatformVoice[] | undefined,
  source: string | null | undefined
): PlatformVoice | null {
  if (!voices?.length || !source) return null;
  return voices.find((v) => v.source === source) ?? null;
}
