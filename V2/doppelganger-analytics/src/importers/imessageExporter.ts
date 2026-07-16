// iMessage TXT export importer — parses output from reagentx/imessage-exporter.
// Format per message block:
//   Line 1: timestamp (optional read receipt)
//   Line 2: sender
//   Line 3+: body (multi-line allowed)
//   Blank line between blocks
//
// Single-line announcements (e.g. "Mum added X to the conversation.") are skipped.

import fs from 'fs';
import path from 'path';
import type { NormalizedConversation, NormalizedMessage, PlatformImporter } from './types.js';

const MONTHS = 'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec';

/** Matches imessage-exporter timestamp prefix (double space before hour). */
export const IMESSAGE_TIMESTAMP_RE = new RegExp(
  `^\\s*(?:${MONTHS})\\s+\\d{1,2},\\s+\\d{4}\\s+\\d{1,2}:\\d{2}:\\d{2}\\s+(?:AM|PM)` +
    `(?:\\s+\\(Read by .+\\))?$`,
  'i'
);

const TIMESTAMP_PREFIX_RE = new RegExp(
  `^(\\s*)(?:(${MONTHS})\\s+(\\d{1,2}),\\s+(\\d{4})\\s+(\\d{1,2}):(\\d{2}):(\\d{2})\\s+(AM|PM))` +
    `(?:\\s+\\(Read by [^)]+\\))?(.*)$`,
  'i'
);

const ANNOUNCEMENT_RE =
  /\b(added .+ to the conversation|removed .+ from the conversation|left the conversation|changed (?:their phone number|the group photo|the chat background)|named the conversation|kept an audio message|unsent a message)\b/i;

const TAPBACK_LINE_RE = /^\s*(?:Tapbacks:\s*)?(?:Loved|Liked|Disliked|Laughed|Emphasized|Questioned) by\b/i;

const ATTACHMENT_RE = /^(?:attachments\/|\S+\.(?:heic|jpeg|jpg|png|gif|mov|mp4|m4a|caf|pdf|svg))$/i;

function parseTimestamp(month: string, day: string, year: string, hour: string, minute: string, second: string, ampm: string): number {
  const months: Record<string, number> = {
    jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
    jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11
  };
  let h = parseInt(hour, 10);
  const m = parseInt(minute, 10);
  const s = parseInt(second, 10);
  const y = parseInt(year, 10);
  const d = parseInt(day, 10);
  const suffix = ampm.toUpperCase();
  if (suffix === 'PM' && h < 12) h += 12;
  if (suffix === 'AM' && h === 12) h = 0;
  return new Date(y, months[month.toLowerCase().slice(0, 3)], d, h, m, s).getTime();
}

function isTimestampLine(line: string): boolean {
  return IMESSAGE_TIMESTAMP_RE.test(line.trim());
}

function stripTapbackLines(body: string): string {
  const lines = body.split('\n');
  const kept: string[] = [];
  let inTapbacks = false;
  for (const line of lines) {
    if (/^\s*Tapbacks:\s*$/i.test(line)) {
      inTapbacks = true;
      continue;
    }
    if (inTapbacks && TAPBACK_LINE_RE.test(line)) continue;
    if (inTapbacks && line.trim() === '') {
      inTapbacks = false;
      continue;
    }
    if (TAPBACK_LINE_RE.test(line)) continue;
    inTapbacks = false;
    kept.push(line);
  }
  return kept.join('\n').trim();
}

function isAnnouncement(text: string): boolean {
  return ANNOUNCEMENT_RE.test(text);
}

export function parseIMessageExporterText(raw: string): NormalizedMessage[] {
  const lines = raw.split(/\r?\n/);
  const messages: NormalizedMessage[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }

    const tsMatch = line.match(TIMESTAMP_PREFIX_RE);
    if (!tsMatch) {
      i++;
      continue;
    }

    const [, , month, day, year, hour, minute, second, ampm, inlineRest] = tsMatch;
    const timestampMs = parseTimestamp(month, day, year, hour, minute, second, ampm);
    const inlineText = (inlineRest || '').trim();

    // Single-line announcement or system event
    if (inlineText) {
      i++;
      continue;
    }

    i++;
    if (i >= lines.length) break;

    const senderLine = lines[i].trim();
    i++;
    if (!senderLine || isTimestampLine(senderLine)) continue;

    const sender = senderLine;

    const bodyLines: string[] = [];
    while (i < lines.length) {
      const next = lines[i];
      if (!next.trim()) {
        i++;
        break;
      }
      if (isTimestampLine(next)) break;
      bodyLines.push(next);
      i++;
    }

    let text = stripTapbackLines(bodyLines.join('\n').trim());
    if (!text) continue;
    if (isAnnouncement(text)) continue;

    const attachments: NormalizedMessage['attachments'] = [];
    if (ATTACHMENT_RE.test(text.split('\n')[0])) {
      const first = text.split('\n')[0];
      const kind = /\.(mov|mp4|m4a|caf)$/i.test(first) ? 'video' as const
        : /\.(heic|jpeg|jpg|png|gif|svg)$/i.test(first) ? 'photo' as const
        : 'photo' as const;
      attachments.push({ kind, uri: first });
      if (text === first) text = `[attachment: ${path.basename(first)}]`;
    }

    messages.push({
      sender,
      timestampMs,
      text,
      attachments,
      reactions: [],
      isSystem: false
    });
  }

  return messages;
}

async function findTxtFiles(rootDir: string): Promise<string[]> {
  const found: string[] = [];

  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > 3) return;
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
        if (lower === '_chat.txt' || lower.includes('whatsapp')) continue;
        found.push(full);
      } else if (entry.isDirectory()) {
        await walk(full, depth + 1);
      }
    }
  }

  await walk(rootDir, 0);
  return found;
}

function looksLikeIMessageExport(filePath: string): boolean {
  try {
    const sample = fs.readFileSync(filePath, 'utf-8').slice(0, 4000);
    const lines = sample.split(/\r?\n/).filter(l => l.trim());
    let hits = 0;
    for (const line of lines.slice(0, 30)) {
      if (IMESSAGE_TIMESTAMP_RE.test(line.trim())) hits++;
    }
    return hits >= 2;
  } catch {
    return false;
  }
}

function conversationIdFromPath(rootDir: string, filePath: string): string {
  const base = path.basename(filePath, '.txt');
  const rel = path.relative(rootDir, path.dirname(filePath));
  if (rel && rel !== '.') {
    return `imessage:${rel}/${base}`;
  }
  return `imessage:${base}`;
}

export const imessageExporterImporter: PlatformImporter = {
  id: 'imessage',
  displayName: 'iMessage (exporter)',

  async detect(rootDir: string): Promise<boolean> {
    // chat.db is handled by imessageImporter — skip if present
    for (const name of ['chat.db', 'Chat.db', 'messages.db']) {
      if (fs.existsSync(path.join(rootDir, name))) return false;
    }

    const files = await findTxtFiles(rootDir);
    if (files.length === 0) return false;

    const imessageFiles = files.filter(looksLikeIMessageExport);
    return imessageFiles.length >= 1;
  },

  async parse(rootDir: string): Promise<NormalizedConversation[]> {
    const files = (await findTxtFiles(rootDir)).filter(looksLikeIMessageExport);
    const conversations: NormalizedConversation[] = [];

    for (const filePath of files) {
      const id = conversationIdFromPath(rootDir, filePath);
      const raw = fs.readFileSync(filePath, 'utf-8');
      const messages = parseIMessageExporterText(raw);
      if (messages.length > 0) {
        messages.sort((a, b) => a.timestampMs - b.timestampMs);
        conversations.push({ id, messages, source: 'imessage' });
      }
    }

    return conversations;
  }
};
