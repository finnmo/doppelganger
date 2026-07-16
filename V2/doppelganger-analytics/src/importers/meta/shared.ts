// Shared Meta (Instagram / Facebook Messenger) JSON parsing.
// Both platforms use the same message_*.json shape from "Download Your Information".

import fs from 'fs';
import path from 'path';
import { decodeInstagramUnicode } from '../../utils/unicodeDecoder.js';
import type { NormalizedAttachment, NormalizedConversation, NormalizedMessage } from '../types.js';

export interface MetaRawMessage {
  sender_name: string;
  timestamp_ms: number;
  content?: string;
  photos?: Array<{ uri: string; creation_timestamp: number; backup_uri?: string }>;
  videos?: Array<{ uri: string; creation_timestamp: number; backup_uri?: string }>;
  audio_files?: Array<{ uri: string; creation_timestamp?: number }>;
  reactions?: Array<{ reaction: string; actor: string; timestamp?: number }>;
  share?: {
    link?: string;
    share_text?: string;
    original_content_owner?: string;
    profile_share_name?: string;
    profile_share_username?: string;
  };
  type?: string;
  is_unsent?: boolean;
}

const META_MESSAGE_BUCKETS = [
  'inbox',
  'archived_threads',
  'filtered_threads',
  'message_requests'
] as const;

const META_SYSTEM_MARKERS = [
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
  'added',
  'removed'
];

function isMessageFile(name: string): boolean {
  return name.startsWith('message_') && name.endsWith('.json');
}

export function inferMetaSource(absoluteConvPath: string, rootDir: string): 'instagram' | 'messenger' {
  const rel = path.relative(rootDir, absoluteConvPath).toLowerCase();
  if (rel.includes('instagram')) return 'instagram';
  if (rel.includes('facebook')) return 'messenger';
  // Legacy flat exports often sit under messages/inbox only — treat as Messenger
  // when the surrounding tree mentions facebook, otherwise Instagram (original default).
  const rootLower = rootDir.toLowerCase();
  if (rootLower.includes('facebook')) return 'messenger';
  if (rootLower.includes('instagram')) return 'instagram';
  return 'instagram';
}

function isMetaSystemText(text: string | null): boolean {
  if (!text) return false;
  const lower = text.trim().toLowerCase();
  return META_SYSTEM_MARKERS.some(marker => lower.includes(marker));
}

export function normalizeMetaMessage(msg: MetaRawMessage): NormalizedMessage {
  const attachments: NormalizedAttachment[] = [];
  for (const photo of msg.photos ?? []) {
    attachments.push({
      kind: 'photo',
      uri: photo.uri,
      creationTimestamp: photo.creation_timestamp,
      backupUri: photo.backup_uri
    });
  }
  for (const video of msg.videos ?? []) {
    attachments.push({
      kind: 'video',
      uri: video.uri,
      creationTimestamp: video.creation_timestamp,
      backupUri: video.backup_uri
    });
  }
  for (const audio of msg.audio_files ?? []) {
    attachments.push({
      kind: 'audio',
      uri: audio.uri,
      creationTimestamp: audio.creation_timestamp
    });
  }

  const text = msg.content ? decodeInstagramUnicode(msg.content) : null;

  return {
    sender: decodeInstagramUnicode(msg.sender_name),
    timestampMs: msg.timestamp_ms,
    text: msg.is_unsent ? null : text,
    attachments,
    reactions: (msg.reactions ?? []).map(r => ({
      emoji: decodeInstagramUnicode(r.reaction),
      actor: decodeInstagramUnicode(r.actor),
      timestamp: r.timestamp ?? 0
    })),
    share: msg.share
      ? {
          link: msg.share.link ? decodeInstagramUnicode(msg.share.link) : undefined,
          text: msg.share.share_text ? decodeInstagramUnicode(msg.share.share_text) : undefined
        }
      : undefined,
    isSystem: isMetaSystemText(text) || msg.type === 'Call'
  };
}

/** Find every conversation directory containing message_*.json under a Meta tree. */
export async function findMetaConversationDirs(rootDir: string): Promise<string[]> {
  const found = new Set<string>();

  async function walk(dirPath: string, depth: number): Promise<void> {
    if (depth > 12) return;
    let entries: fs.Dirent[];
    try {
      entries = await fs.promises.readdir(dirPath, { withFileTypes: true });
    } catch {
      return;
    }

    if (entries.some(e => e.isFile() && isMessageFile(e.name))) {
      found.add(dirPath);
      return;
    }

    for (const entry of entries) {
      if (entry.isDirectory()) {
        await walk(path.join(dirPath, entry.name), depth + 1);
      }
    }
  }

  // Fast path: known Meta layout messages/<bucket>/<conversation>/
  for (const bucket of META_MESSAGE_BUCKETS) {
    const bucketDir = path.join(rootDir, 'messages', bucket);
    if (!fs.existsSync(bucketDir)) continue;
    const convs = await fs.promises.readdir(bucketDir, { withFileTypes: true });
    for (const conv of convs) {
      if (conv.isDirectory()) {
        await walk(path.join(bucketDir, conv.name), 0);
      }
    }
  }

  // Nested exports: your_instagram_activity/messages/inbox, your_facebook_activity/...
  const nestedMarkers = ['your_instagram_activity', 'your_facebook_activity', 'instagram', 'facebook'];
  async function walkNested(dir: string, depth: number): Promise<void> {
    if (depth > 4) return;
    let entries: fs.Dirent[];
    try {
      entries = await fs.promises.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (!entry.isDirectory()) continue;
      const full = path.join(dir, entry.name);
      const nameLower = entry.name.toLowerCase();
      if (nestedMarkers.some(m => nameLower.includes(m)) || entry.name === 'messages') {
        await walk(full, 0);
        await walkNested(full, depth + 1);
      }
    }
  }
  await walkNested(rootDir, 0);

  // Fallback: full recursive search (legacy instagram-only flat trees).
  if (found.size === 0) {
    await walk(rootDir, 0);
  }

  return [...found];
}

export async function parseMetaConversationDir(
  convDir: string,
  rootDir: string
): Promise<{ id: string; source: 'instagram' | 'messenger'; messages: NormalizedMessage[] }> {
  const conversationId = decodeInstagramUnicode(path.basename(convDir));
  const source = inferMetaSource(convDir, rootDir);
  const messages: NormalizedMessage[] = [];
  const entries = await fs.promises.readdir(convDir, { withFileTypes: true });

  const messageFiles = entries
    .filter(e => e.isFile() && isMessageFile(e.name))
    .map(e => e.name)
    .sort((a, b) => {
      const na = parseInt(a.replace(/\D/g, ''), 10) || 0;
      const nb = parseInt(b.replace(/\D/g, ''), 10) || 0;
      return na - nb;
    });

  for (const fileName of messageFiles) {
    const raw = JSON.parse(
      await fs.promises.readFile(path.join(convDir, fileName), 'utf-8')
    );
    for (const msg of (raw.messages ?? []) as MetaRawMessage[]) {
      if (!msg.sender_name || !msg.timestamp_ms) continue;
      messages.push(normalizeMetaMessage(msg));
    }
  }

  return { id: conversationId, source, messages };
}

export async function parseMetaExport(rootDir: string): Promise<
  Array<NormalizedConversation & { source: 'instagram' | 'messenger' }>
> {
  const dirs = await findMetaConversationDirs(rootDir);
  const conversations: Array<NormalizedConversation & { source: 'instagram' | 'messenger' }> = [];

  for (const convDir of dirs) {
    const parsed = await parseMetaConversationDir(convDir, rootDir);
    if (parsed.messages.length === 0) continue;
    console.log(`📁 Processing conversation (${parsed.source}): ${parsed.id}`);
    conversations.push({
      id: parsed.id,
      messages: parsed.messages,
      source: parsed.source
    });
  }

  return conversations;
}

export async function detectMetaJsonExport(rootDir: string): Promise<boolean> {
  const dirs = await findMetaConversationDirs(rootDir);
  return dirs.length > 0;
}
