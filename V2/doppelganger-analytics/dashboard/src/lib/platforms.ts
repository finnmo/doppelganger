/** Messaging platforms supported in the dashboard. */

export const PLATFORM_SOURCES = [
  'instagram',
  'messenger',
  'whatsapp',
  'imessage',
  'meta',
] as const;

export type PlatformSource = (typeof PLATFORM_SOURCES)[number] | string;

const SOURCE_PREFIX_RE = /^(instagram|messenger|whatsapp|imessage|meta):(.+)$/;

export function parseConversationId(conversationId: string): {
  source: string | null;
  rawId: string;
} {
  const match = conversationId.match(SOURCE_PREFIX_RE);
  if (match) return { source: match[1], rawId: match[2] };
  return { source: null, rawId: conversationId };
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

/** Soft, distinct colors per platform — readable on white cards. */
export function platformStyles(source: string | null | undefined): {
  bg: string;
  text: string;
  ring: string;
  dot: string;
} {
  switch (source) {
    case 'instagram':
      return {
        bg: 'bg-rose-50',
        text: 'text-rose-800',
        ring: 'ring-rose-200',
        dot: 'bg-rose-500',
      };
    case 'messenger':
      return {
        bg: 'bg-sky-50',
        text: 'text-sky-800',
        ring: 'ring-sky-200',
        dot: 'bg-sky-500',
      };
    case 'whatsapp':
      return {
        bg: 'bg-emerald-50',
        text: 'text-emerald-800',
        ring: 'ring-emerald-200',
        dot: 'bg-emerald-500',
      };
    case 'imessage':
      return {
        bg: 'bg-blue-50',
        text: 'text-blue-800',
        ring: 'ring-blue-200',
        dot: 'bg-blue-500',
      };
    default:
      return {
        bg: 'bg-slate-50',
        text: 'text-slate-700',
        ring: 'ring-slate-200',
        dot: 'bg-slate-400',
      };
  }
}

export function resolveSource(
  conversationId: string,
  explicitSource?: string | null
): string {
  if (explicitSource) return explicitSource;
  return parseConversationId(conversationId).source ?? 'unknown';
}
