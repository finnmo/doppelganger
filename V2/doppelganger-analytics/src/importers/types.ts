// Platform-neutral import model. Every messaging platform (Instagram,
// Messenger, WhatsApp, ...) is supported by one PlatformImporter that parses
// that platform's export into these normalized shapes. Everything downstream
// (database schema, processors, dashboard) only sees normalized data.

export interface NormalizedAttachment {
  kind: 'photo' | 'video' | 'audio';
  uri: string;
  creationTimestamp?: number;
  backupUri?: string;
}

export interface NormalizedReaction {
  /** Reaction emoji/content, already decoded to proper UTF-8. */
  emoji: string;
  actor: string;
  /** Seconds since epoch when the platform provides it; 0 when unknown. */
  timestamp: number;
}

export interface NormalizedShare {
  link?: string;
  text?: string;
}

export interface NormalizedMessage {
  /** Display name of the sender, already decoded. */
  sender: string;
  timestampMs: number;
  /** Message text, already decoded. Null for pure media/system entries. */
  text: string | null;
  attachments: NormalizedAttachment[];
  reactions: NormalizedReaction[];
  share?: NormalizedShare;
  /**
   * True when the platform marks this as a system/notification event
   * (calls, reactions-as-messages, etc.). Importers set this; processors
   * should prefer it over content heuristics when present.
   */
  isSystem?: boolean;
}

export interface NormalizedConversation {
  /** Stable conversation identifier within the export (e.g. folder name). */
  id: string;
  messages: NormalizedMessage[];
  /**
   * Platform source for DB `messages.source` (e.g. 'instagram', 'messenger').
   * When omitted, the importer's `id` is used.
   */
  source?: string;
}

/**
 * One importer per platform. `detect` inspects an extracted export directory
 * and claims it when the layout matches; `parse` converts it into normalized
 * conversations. Adding a platform = implementing this interface and
 * registering it in registry.ts.
 */
export interface PlatformImporter {
  /** Stable id stored in the messages.source column (e.g. 'instagram'). */
  id: string;
  displayName: string;
  detect(rootDir: string): Promise<boolean>;
  parse(rootDir: string): Promise<NormalizedConversation[]>;
}
