// Platform / messaging-source helpers shared by import + processors.

export const KNOWN_SOURCES = [
  'instagram',
  'messenger',
  'whatsapp',
  'imessage',
  'meta'
] as const;

export type KnownSource = (typeof KNOWN_SOURCES)[number];

const SOURCE_PREFIX_RE = /^(instagram|messenger|whatsapp|imessage|meta):(.+)$/;

/** Prefix conversation ids so Instagram "Alice" and WhatsApp "Alice" never collide. */
export function namespacedConversationId(source: string, id: string): string {
  if (SOURCE_PREFIX_RE.test(id)) return id;
  return `${source}:${id}`;
}

export function parseConversationId(conversationId: string): {
  source: string | null;
  rawId: string;
} {
  const match = conversationId.match(SOURCE_PREFIX_RE);
  if (match) return { source: match[1], rawId: match[2] };
  return { source: null, rawId: conversationId };
}

export function displayConversationId(conversationId: string): string {
  return parseConversationId(conversationId).rawId;
}

export function sourceLabel(source: string | null | undefined): string {
  switch (source) {
    case 'instagram':
      return 'Instagram';
    case 'messenger':
      return 'Messenger';
    case 'whatsapp':
      return 'WhatsApp';
    case 'imessage':
      return 'iMessage';
    case 'meta':
      return 'Meta';
    default:
      return source ? source.charAt(0).toUpperCase() + source.slice(1) : 'Unknown';
  }
}
