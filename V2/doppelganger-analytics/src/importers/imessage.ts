// iMessage importer — reads Apple's chat.db (Messages SQLite database).
// Copy chat.db from ~/Library/Messages/chat.db (read-only) before importing.

import fs from 'fs';
import path from 'path';
import Database from 'better-sqlite3';
import type { NormalizedConversation, NormalizedMessage, PlatformImporter } from './types.js';

const APPLE_EPOCH_MS = 978_307_200_000;

const IMESSAGE_DB_NAMES = ['chat.db', 'Chat.db', 'messages.db'];

/** Extract readable text from attributedBody blobs on Ventura+ when text is NULL. */
export function extractAttributedBodyText(blob: Buffer | null | undefined): string | null {
  if (!blob || blob.length === 0) return null;

  const utf8 = blob.toString('utf8');
  const candidates = utf8
    .split(/[\x00-\x08\x0b\x0c\x0e-\x1f]/)
    .map(s => s.trim())
    .filter(s => s.length >= 2 && /[a-zA-Z0-9]/.test(s) && !/^NS[A-Z]/.test(s) && !/^__k/.test(s));

  if (candidates.length === 0) return null;
  candidates.sort((a, b) => b.length - a.length);
  const best = candidates[0];
  return best.length >= 2 ? best : null;
}

function appleDateToMs(raw: number | null | undefined): number {
  if (raw == null || raw === 0) return 0;
  // Nanoseconds since 2001-01-01 (modern macOS).
  if (raw > 1e15) return Math.floor(raw / 1_000_000) + APPLE_EPOCH_MS;
  // Microseconds.
  if (raw > 1e12) return raw / 1000 + APPLE_EPOCH_MS;
  // Seconds.
  if (raw < 1e11) return raw * 1000 + APPLE_EPOCH_MS;
  // Milliseconds.
  return raw + APPLE_EPOCH_MS;
}

function findChatDb(rootDir: string): string | null {
  for (const name of IMESSAGE_DB_NAMES) {
    const candidate = path.join(rootDir, name);
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      return candidate;
    }
  }
  return null;
}

interface ChatRow {
  chat_id: number;
  chat_identifier: string;
  display_name: string | null;
}

interface MessageRow {
  message_id: number;
  chat_id: number;
  text: string | null;
  attributed_body: Buffer | null;
  date: number;
  is_from_me: number;
  handle: string | null;
  is_system: number;
  item_type: number;
}

const SYSTEM_ITEM_TYPES = new Set([1, 2, 3]); // tapbacks, group actions, etc.

export const imessageImporter: PlatformImporter = {
  id: 'imessage',
  displayName: 'iMessage',

  async detect(rootDir: string): Promise<boolean> {
    return findChatDb(rootDir) !== null;
  },

  async parse(rootDir: string): Promise<NormalizedConversation[]> {
    const dbPath = findChatDb(rootDir);
    if (!dbPath) return [];

    const db = new Database(dbPath, { readonly: true, fileMustExist: true });

    try {
      const chats = db.prepare(`
        SELECT ROWID AS chat_id, chat_identifier, display_name
        FROM chat
      `).all() as ChatRow[];

      const msgStmt = db.prepare(`
        SELECT
          m.ROWID AS message_id,
          cmj.chat_id AS chat_id,
          m.text,
          m.attributedBody AS attributed_body,
          m.date,
          m.is_from_me,
          h.id AS handle,
          m.is_system_message AS is_system,
          m.item_type
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        WHERE cmj.chat_id = ?
        ORDER BY m.date ASC
      `);

      const conversations: NormalizedConversation[] = [];

      for (const chat of chats) {
        const rows = msgStmt.all(chat.chat_id) as MessageRow[];
        if (rows.length === 0) continue;

        const id = chat.display_name?.trim() || chat.chat_identifier || `chat_${chat.chat_id}`;
        console.log(`📁 Processing iMessage chat: ${id}`);

        const messages: NormalizedMessage[] = [];
        for (const row of rows) {
          const text = row.text?.trim()
            || extractAttributedBodyText(row.attributed_body)
            || null;

          const sender = row.is_from_me
            ? 'Me'
            : (row.handle?.trim() || 'Unknown');

          const isSystem = Boolean(row.is_system)
            || SYSTEM_ITEM_TYPES.has(row.item_type)
            || (text !== null && /^(You )?reacted to|^Loved |^Liked |^Disliked |^Laughed at |^Emphasized |^Questioned /.test(text));

          messages.push({
            sender,
            timestampMs: appleDateToMs(row.date),
            text,
            attachments: [],
            reactions: [],
            isSystem
          });
        }

        if (messages.length > 0) {
          conversations.push({ id, messages, source: 'imessage' });
        }
      }

      return conversations;
    } finally {
      db.close();
    }
  }
};
