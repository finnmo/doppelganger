/**
 * Shared message-filtering helpers. System notifications (media sends,
 * reactions, call events) should be excluded from text/sentiment/word analysis.
 *
 * Prefer the `is_system` column set by platform importers. Content heuristics
 * remain as a fallback for legacy rows and synthetic attachment text.
 */

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
  'missed a call'
];

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
  return SYSTEM_MESSAGE_MARKERS.some(marker => text.includes(marker));
}

/**
 * Common words excluded from word-frequency analysis. Single canonical set so
 * word charts stay consistent.
 */
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
  'should', 'his', 'her', 'him', 'she'
]);
