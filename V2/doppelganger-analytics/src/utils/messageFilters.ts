/**
 * Shared message-filtering helpers. System notifications (media sends,
 * reactions, call events) should be excluded from text/sentiment/word analysis.
 *
 * Prefer the `is_system` column set by platform importers. Content heuristics
 * remain as a fallback for legacy rows and synthetic attachment text.
 */

import { parseConversationId } from './platformSource.js';

// Meta/Instagram-style auto-generated phrases. Kept deliberately specific —
// e.g. no bare "sent", which would drop real messages like "I sent you the file".
const SYSTEM_MESSAGE_MARKERS = [
  'sent an attachment',
  'sent a photo',
  'sent a video',
  'reacted to your message',
  'reacted to a message',
  'liked a message',
  'started a call',
  'ended the call',
  'missed call',
  'missed a call',
  'changed the group photo',
  'named the group',
  'left the group',
  'removed a message',
  'sent a voice message',
  'shared a link',
  'gamepigeon',
];

/** Regex fallbacks for importer synthetic text and Messenger wording variants. */
const SYSTEM_MESSAGE_PATTERNS = [
  /\bsent\s+\d+\s+photos?\b/i,
  /\bsent\s+\d+\s+videos?\b/i,
  /\breacted\b.*\bto\b.*\b(your|a)\s+message\b/i,
  /\bremoved\s+.*\s+from\s+the\s+group\b/i,
];

/**
 * Tokens that almost always come from platform notifications, not real vocabulary.
 * Used by word-frequency analysis only (not full-message filtering).
 */
export const NOTIFICATION_STOP_WORDS = new Set<string>([
  'sent', 'photo', 'photos', 'video', 'videos', 'reacted', 'reaction', 'reactions',
  'message', 'messages', 'attachment', 'attachments', 'liked', 'shared', 'share',
  'voice', 'removed', 'unsent', 'sticker', 'stickers', 'gamepigeon', 'move',
  'named', 'changed', 'added', 'call', 'missed', 'started', 'ended',
]);

/**
 * True for empty/very short content and known system notifications —
 * i.e. messages that should be excluded from text/sentiment/word analysis.
 *
 * Pass `isSystem` (from messages.is_system) when available so non-Instagram
 * platforms can mark system events without matching English phrase heuristics.
 */
export function isSystemMessage(
  content: string | null | undefined,
  isSystem?: number | boolean | null
): boolean {
  if (isSystem === 1 || isSystem === true) return true;
  if (!content) return true;
  const text = content.trim().toLowerCase();
  if (text.length < 3) return true;
  if (text.startsWith('__')) return true;
  if (SYSTEM_MESSAGE_MARKERS.some(marker => text.includes(marker))) return true;
  return SYSTEM_MESSAGE_PATTERNS.some(pattern => pattern.test(text));
}

/**
 * Common words excluded from word-frequency analysis. Single canonical set so
 * word charts stay consistent.
 */
/** Tokenize a display name into lowercase word tokens for per-conversation blocklists. */
export function tokenizeParticipantName(name: string): string[] {
  return name
    .toLowerCase()
    .replace(/[^\w\s'’]/g, ' ')
    .split(/\s+/)
    .map(t => t.trim())
    .filter(t => t.length >= 2);
}

/** Tokenize a conversation folder/id label (e.g. ellu_radhu_guys). */
export function tokenizeConversationLabel(rawId: string): string[] {
  return rawId
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/[\s_]+/)
    .map(t => t.trim())
    .filter(t => t.length >= 2);
}

/**
 * Map conversation_id → Set of name tokens to exclude from word-frequency analysis.
 * Built from distinct senders and conversation labels (names are labels, not vocabulary).
 */
export function buildConversationNameBlocklist(
  rows: Array<{ sender: string; conversation_id: string }>
): Map<string, Set<string>> {
  const map = new Map<string, Set<string>>();
  for (const { sender, conversation_id } of rows) {
    if (!sender?.trim()) continue;
    let tokens = map.get(conversation_id);
    if (!tokens) {
      tokens = new Set();
      map.set(conversation_id, tokens);
    }
    for (const token of tokenizeParticipantName(sender)) {
      tokens.add(token);
    }
    const { rawId } = parseConversationId(conversation_id);
    for (const token of tokenizeConversationLabel(rawId)) {
      tokens.add(token);
    }
  }
  return map;
}

/** Trim and cap message snippets for dashboard example tooltips. */
export function trimMessageSnippet(text: string, maxLen = 140): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= maxLen) return clean;
  return `${clean.slice(0, maxLen - 1)}…`;
}

export const STOP_WORDS = new Set<string>([
  'the', 'and', 'that', 'this', 'but', 'for', 'not', 'you', 'are', 'was', 'will',
  'can', 'with', 'have', 'had', 'they', 'them', 'their', 'there', 'then',
  'than', 'when', 'what', 'who', 'why', 'how', 'where', 'which', 'some', 'any',
  'all', 'one', 'two', 'see', 'get', 'got', 'make', 'made', 'take', 'come', 'came',
  'went', 'said', 'say', 'know', 'think', 'like', 'just', 'now', 'way', 'may',
  'use', 'its', 'only', 'over', 'also', 'back', 'after', 'first', 'well', 'work',
  'life', 'day', 'part', 'year', 'much', 'good', 'new', 'old', 'great', 'right',
  'still', 'own', 'long', 'same', 'want', 'look', 'find', 'give', 'here', 'both',
  'last', 'most', 'very', 'call', 'even', 'need', 'feel', 'try', 'ask', 'turn',
  'move', 'live', 'seem', 'show', 'help', 'talk', 'let', 'put', 'would', 'could',
  'should', 'his', 'her', 'him', 'she', 'your',
]);
