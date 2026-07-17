// WhatsApp plain-text export (_chat.txt) importer.
// Supports common locale formats from "Export chat" on iOS/Android.

import fs from 'fs';
import path from 'path';
import type { NormalizedConversation, NormalizedMessage, PlatformImporter } from './types.js';

const WHATSAPP_SYSTEM_MARKERS = [
  'end-to-end encrypted',
  'security code changed',
  'created group',
  'added you',
  'left',
  'changed the subject',
  'changed this group',
  'changed the group',
  'removed',
  'joined using',
  'deleted this message',
  'message was deleted',
  'missed voice call',
  'missed video call',
  'calling',
  'media omitted',
  'document omitted',
  'sticker omitted',
  'gif omitted',
  'audio omitted',
  'video omitted',
  'image omitted',
  'contact card omitted',
  'waiting for this message',
  'messages and calls are'
];

// [12/31/23, 10:30:45 PM] Sender: text
const BRACKET_LINE =
  /^\u200e?\[(\d{1,2}[\/.]\d{1,2}[\/.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]\s([^:]*?):\s([\s\S]*)$/i;

// 31.12.23, 22:30 - Sender: text  (also without brackets)
const DASH_LINE =
  /^\u200e?(\d{1,2}[\/.]\d{1,2}[\/.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\s+-\s([^:]*?):\s([\s\S]*)$/i;

// System lines without a sender colon: [date, time] notification text
const BRACKET_SYSTEM =
  /^\u200e?\[(\d{1,2}[\/.]\d{1,2}[\/.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\]\s([\s\S]+)$/i;

const DASH_SYSTEM =
  /^\u200e?(\d{1,2}[\/.]\d{1,2}[\/.]\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\s+-\s([\s\S]+)$/i;

function isWhatsAppSystemText(text: string): boolean {
  const lower = text.trim().toLowerCase();
  return WHATSAPP_SYSTEM_MARKERS.some(marker => lower.includes(marker));
}

function parseDateParts(datePart: string, timePart: string): number | null {
  const dateBits = datePart.split(/[\/.]/).map(s => s.trim());
  if (dateBits.length !== 3) return null;

  let day: number;
  let month: number;
  let year = parseInt(dateBits[2], 10);
  if (year < 100) year += 2000;

  const a = parseInt(dateBits[0], 10);
  const b = parseInt(dateBits[1], 10);
  // Heuristic: if first segment > 12 it must be day-first (EU).
  if (a > 12) {
    day = a;
    month = b;
  } else if (b > 12) {
    month = a;
    day = b;
  } else {
    // Ambiguous — prefer day-first (common in EU exports).
    day = a;
    month = b;
  }

  const timeClean = timePart.trim().toUpperCase();
  const ampm = timeClean.match(/\s*(AM|PM)$/);
  const timeOnly = ampm ? timeClean.replace(/\s*(AM|PM)$/, '').trim() : timeClean;
  const timeBits = timeOnly.split(':').map(s => parseInt(s, 10));
  if (timeBits.length < 2 || timeBits.some(n => Number.isNaN(n))) return null;

  let hours = timeBits[0];
  const minutes = timeBits[1];
  const seconds = timeBits[2] ?? 0;

  if (ampm) {
    const suffix = ampm[1];
    if (suffix === 'PM' && hours < 12) hours += 12;
    if (suffix === 'AM' && hours === 12) hours = 0;
  }

  const ms = new Date(year, month - 1, day, hours, minutes, seconds).getTime();
  return Number.isNaN(ms) ? null : ms;
}

interface ParsedLine {
  timestampMs: number;
  sender: string;
  text: string;
  isSystem: boolean;
}

function parseMessageLine(line: string): ParsedLine | null {
  const trimmed = line.replace(/^\uFEFF/, '').trimEnd();
  if (!trimmed) return null;

  let match = trimmed.match(BRACKET_LINE);
  if (match) {
    const ts = parseDateParts(match[1], match[2]);
    if (ts === null) return null;
    const sender = match[3].trim() || 'System';
    const text = match[4].trim();
    return {
      timestampMs: ts,
      sender,
      text,
      isSystem: isWhatsAppSystemText(text) || sender === 'System'
    };
  }

  match = trimmed.match(DASH_LINE);
  if (match) {
    const ts = parseDateParts(match[1], match[2]);
    if (ts === null) return null;
    const sender = match[3].trim() || 'System';
    const text = match[4].trim();
    return {
      timestampMs: ts,
      sender,
      text,
      isSystem: isWhatsAppSystemText(text) || sender === 'System'
    };
  }

  match = trimmed.match(BRACKET_SYSTEM);
  if (match) {
    const ts = parseDateParts(match[1], match[2]);
    if (ts === null) return null;
    const text = match[3].trim();
    return {
      timestampMs: ts,
      sender: 'System',
      text,
      isSystem: true
    };
  }

  match = trimmed.match(DASH_SYSTEM);
  if (match) {
    const ts = parseDateParts(match[1], match[2]);
    if (ts === null) return null;
    const text = match[3].trim();
    // Only treat as system if there's no "Name: message" tail.
    if (text.includes(':')) return null;
    return {
      timestampMs: ts,
      sender: 'System',
      text,
      isSystem: true
    };
  }

  return null;
}

async function findChatTxtFiles(rootDir: string): Promise<string[]> {
  const found: string[] = [];

  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > 4) return;
    let entries: fs.Dirent[];
    try {
      entries = await fs.promises.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isFile() && entry.name.endsWith('.txt')) {
        const lower = entry.name.toLowerCase();
        if (lower === '_chat.txt' || lower.includes('chat') || lower.includes('whatsapp')) {
          found.push(full);
        }
      } else if (entry.isDirectory()) {
        await walk(full, depth + 1);
      }
    }
  }

  await walk(rootDir, 0);
  return found;
}

function parseChatFile(filePath: string): NormalizedMessage[] {
  const raw = fs.readFileSync(filePath, 'utf-8');
  const lines = raw.split(/\r?\n/);
  const messages: NormalizedMessage[] = [];
  let current: NormalizedMessage | null = null;

  for (const line of lines) {
    const parsed = parseMessageLine(line);
    if (parsed) {
      if (current) messages.push(current);
      current = {
        sender: parsed.sender,
        timestampMs: parsed.timestampMs,
        text: parsed.text || null,
        attachments: [],
        reactions: [],
        isSystem: parsed.isSystem
      };
      continue;
    }

    // Continuation of a multiline message.
    if (current && line.trim()) {
      current.text = current.text ? `${current.text}\n${line}` : line;
    }
  }
  if (current) messages.push(current);

  return messages;
}

function conversationIdFromPath(filePath: string): string {
  const base = path.basename(filePath, path.extname(filePath));
  if (base === '_chat' || base.toLowerCase() === 'chat') {
    return 'WhatsApp Chat';
  }
  return base;
}

export const whatsappImporter: PlatformImporter = {
  id: 'whatsapp',
  displayName: 'WhatsApp',

  async detect(rootDir: string): Promise<boolean> {
    const files = await findChatTxtFiles(rootDir);
    return files.length > 0;
  },

  async parse(rootDir: string): Promise<NormalizedConversation[]> {
    const files = await findChatTxtFiles(rootDir);
    const conversations: NormalizedConversation[] = [];

    for (const filePath of files) {
      const id = conversationIdFromPath(filePath);
      console.log(`📁 Processing WhatsApp chat: ${id}`);
      const messages = parseChatFile(filePath);
      if (messages.length > 0) {
        conversations.push({ id, messages, source: 'whatsapp' });
      }
    }

    return conversations;
  }
};
