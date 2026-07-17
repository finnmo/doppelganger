// src/importer.ts
// Platform-neutral import pipeline: extract the archive, detect which
// platform's export it is (src/importers/registry.ts), parse it into the
// normalized model, and insert into SQLite. No platform-specific parsing
// lives here — that belongs in a PlatformImporter.
import fs from 'fs';
import os from 'os';
import path from 'path';
import unzipper from 'unzipper';
import { getDb, closeDb } from './db/client.js';
import { isDevMode, applyDevModeLimit } from './utils/devMode.js';
import { detectPlatform, supportedPlatforms } from './importers/registry.js';
import type { NormalizedMessage } from './importers/types.js';
import { namespacedConversationId } from './utils/platformSource.js';
import { recordImportComplete } from './pipeline/freshness.js';
import chalk from 'chalk';

// Re-exported for backwards compatibility; the raw Instagram shape now lives
// with the Instagram importer.
export type { InstagramRawMessage as RawMessage } from './importers/instagram.js';

async function prepareInboxDir(sourcePath: string): Promise<string> {
  // Unique per import — a shared fixed path races when two imports overlap
  // (or when cleanup fails mid-rmdir) and can drop whole conversation folders.
  const tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'doppelganger-inbox-'));

  if (sourcePath.endsWith('.zip')) {
    const directory = await unzipper.Open.file(sourcePath);
    await directory.extract({ path: tempDir });
  } else {
    const stat = await fs.promises.stat(sourcePath);
    if (stat.isFile()) {
      await fs.promises.copyFile(sourcePath, path.join(tempDir, path.basename(sourcePath)));
    } else {
      await fs.promises.cp(sourcePath, tempDir, { recursive: true });
    }
  }

  return tempDir;
}

/**
 * Fallback text for attachment-only messages so downstream text/classification
 * steps see them. Platform-neutral: derived from normalized attachments.
 */
function syntheticContent(msg: NormalizedMessage): string | null {
  const photos = msg.attachments.filter(a => a.kind === 'photo').length;
  const videos = msg.attachments.filter(a => a.kind === 'video').length;
  const audio = msg.attachments.filter(a => a.kind === 'audio').length;

  if (photos > 0) return `${msg.sender} sent ${photos} photo${photos > 1 ? 's' : ''}`;
  if (videos > 0) return `${msg.sender} sent ${videos} video${videos > 1 ? 's' : ''}`;
  if (audio > 0) return `${msg.sender} sent a voice message`;
  if (msg.share) return msg.share.text ?? `${msg.sender} shared a link`;
  return null;
}

export async function importArchive(sourcePath: string): Promise<void> {
  const db = await getDb();
  const tempDir = await prepareInboxDir(sourcePath);

  try {
    const importer = await detectPlatform(tempDir);
    if (!importer) {
      throw new Error(
        `Could not detect the export format. Supported platforms: ${supportedPlatforms()}. ` +
        'Make sure the archive is an unmodified data export (for Instagram: JSON format with Messages selected).'
      );
    }
    console.log(chalk.blue(`🔍 Detected platform: ${importer.displayName}`));

    const conversations = await importer.parse(tempDir);

    // Flatten all messages from all conversations. Namespace conversation ids
    // by source so the same contact name on Instagram and WhatsApp stays distinct.
    const allMessages: Array<NormalizedMessage & { conversation_id: string; source: string }> = [];
    for (const conv of conversations) {
      const source = conv.source ?? importer.id;
      const conversationId = namespacedConversationId(source, conv.id);
      for (const msg of conv.messages) {
        allMessages.push({ ...msg, conversation_id: conversationId, source });
      }
    }

    // Apply global dev mode limit to total messages
    const limitedMessages = isDevMode() ? applyDevModeLimit(allMessages, 'messages') : allMessages;

    // Replace only the platforms present in this import — keep other sources
    // so Instagram + iMessage can coexist in one database.
    const sourcesInBatch = [...new Set(limitedMessages.map(m => m.source))];
    console.log(
      chalk.yellow(
        `🗑️  Replacing existing data for: ${sourcesInBatch.join(', ')} (other platforms kept)...`
      )
    );

    db.pragma('foreign_keys = OFF');
    const placeholders = sourcesInBatch.map(() => '?').join(',');
    const sourceFilter = `source IN (${placeholders})`;

    db.prepare(`
      DELETE FROM sentiment WHERE message_id IN (SELECT id FROM messages WHERE ${sourceFilter})
    `).run(...sourcesInBatch);
    db.prepare(`
      DELETE FROM text_metrics WHERE message_id IN (SELECT id FROM messages WHERE ${sourceFilter})
    `).run(...sourcesInBatch);
    db.prepare(`
      DELETE FROM emotions WHERE message_id IN (SELECT id FROM messages WHERE ${sourceFilter})
    `).run(...sourcesInBatch);
    db.prepare(`
      DELETE FROM message_reactions WHERE message_id IN (SELECT id FROM messages WHERE ${sourceFilter})
    `).run(...sourcesInBatch);
    db.prepare(`
      DELETE FROM message_photos WHERE message_id IN (SELECT id FROM messages WHERE ${sourceFilter})
    `).run(...sourcesInBatch);
    db.prepare(`
      DELETE FROM message_videos WHERE message_id IN (SELECT id FROM messages WHERE ${sourceFilter})
    `).run(...sourcesInBatch);
    db.prepare(`
      DELETE FROM message_audio WHERE message_id IN (SELECT id FROM messages WHERE ${sourceFilter})
    `).run(...sourcesInBatch);
    // response_times / conversations are conversation-scoped; drop rows whose
    // conversation_id is being replaced or still belongs to a removed source.
    db.prepare(`
      DELETE FROM response_times WHERE conversation_id IN (
        SELECT DISTINCT conversation_id FROM messages WHERE ${sourceFilter}
      )
    `).run(...sourcesInBatch);
    db.prepare(`
      DELETE FROM conversations WHERE conversation_id IN (
        SELECT DISTINCT conversation_id FROM messages WHERE ${sourceFilter}
      )
    `).run(...sourcesInBatch);
    db.prepare(`DELETE FROM messages WHERE ${sourceFilter}`).run(...sourcesInBatch);
    db.pragma('foreign_keys = ON');

    const remaining = db.prepare('SELECT COUNT(*) as count FROM messages').get() as { count: number };
    console.log(
      chalk.green(
        `✅ Cleared ${sourcesInBatch.join(', ')} — ${remaining.count} messages from other platforms remain`
      )
    );

    const insertMessage = db.prepare(`
      INSERT INTO messages (conversation_id, sender, timestamp_ms, content, has_photos, has_videos, has_audio, has_share, share_link, source, is_system)
      VALUES (@conversation_id, @sender, @timestamp_ms, @content, @has_photos, @has_videos, @has_audio, @has_share, @share_link, @source, @is_system)
    `);

    const insertPhoto = db.prepare(`
      INSERT INTO message_photos (message_id, uri, creation_timestamp, backup_uri)
      VALUES (@message_id, @uri, @creation_timestamp, @backup_uri)
    `);

    const insertVideo = db.prepare(`
      INSERT INTO message_videos (message_id, uri, creation_timestamp, backup_uri)
      VALUES (@message_id, @uri, @creation_timestamp, @backup_uri)
    `);

    const insertAudio = db.prepare(`
      INSERT INTO message_audio (message_id, uri, creation_timestamp)
      VALUES (@message_id, @uri, @creation_timestamp)
    `);

    const insertReaction = db.prepare(`
      INSERT INTO message_reactions (message_id, reaction, actor, timestamp)
      VALUES (@message_id, @reaction, @actor, @timestamp)
    `);

    const insertMany = db.transaction((messages: Array<NormalizedMessage & { conversation_id: string; source: string }>) => {
      let count = 0;
      for (const msg of messages) {
        const photos = msg.attachments.filter(a => a.kind === 'photo');
        const videos = msg.attachments.filter(a => a.kind === 'video');
        const audio = msg.attachments.filter(a => a.kind === 'audio');

        const synthetic = msg.text == null ? syntheticContent(msg) : null;
        const content = msg.text ?? synthetic;
        const isSystem = msg.isSystem || synthetic != null;

        const result = insertMessage.run({
          conversation_id: msg.conversation_id,
          sender: msg.sender,
          timestamp_ms: msg.timestampMs,
          content,
          has_photos: photos.length > 0 ? 1 : 0,
          has_videos: videos.length > 0 ? 1 : 0,
          has_audio: audio.length > 0 ? 1 : 0,
          has_share: msg.share ? 1 : 0,
          share_link: msg.share?.link ?? null,
          source: msg.source,
          is_system: isSystem ? 1 : 0
        });

        const messageId = result.lastInsertRowid;

        for (const photo of photos) {
          insertPhoto.run({
            message_id: messageId,
            uri: photo.uri,
            creation_timestamp: photo.creationTimestamp ?? null,
            backup_uri: photo.backupUri ?? null
          });
        }
        for (const video of videos) {
          insertVideo.run({
            message_id: messageId,
            uri: video.uri,
            creation_timestamp: video.creationTimestamp ?? null,
            backup_uri: video.backupUri ?? null
          });
        }
        for (const item of audio) {
          insertAudio.run({
            message_id: messageId,
            uri: item.uri,
            creation_timestamp: item.creationTimestamp ?? null
          });
        }
        for (const reaction of msg.reactions) {
          insertReaction.run({
            message_id: messageId,
            reaction: reaction.emoji,
            actor: reaction.actor,
            timestamp: reaction.timestamp
          });
        }

        count++;
      }
      return count;
    });

    const conversationIds = [...new Set(limitedMessages.map(m => m.conversation_id))];
    const inserted = insertMany(limitedMessages);

    console.log(
      chalk.green(
        `✅ Imported ${inserted} messages from ${conversationIds.length} conversations (${importer.displayName})`
      )
    );
    console.log(chalk.blue(`📊 Conversations: ${conversationIds.slice(0, 8).join(', ')}${conversationIds.length > 8 ? '…' : ''}`));

    recordImportComplete(db, sourcesInBatch);
    console.log(chalk.gray('📌 Recorded import timestamp (run generate to refresh analytics).'));

  } finally {
    await closeDb(db);
    try {
      await fs.promises.rm(tempDir, { recursive: true, force: true, maxRetries: 3 });
    } catch (err) {
      console.warn(chalk.yellow(`Warning: could not remove temp dir ${tempDir}:`), err);
    }
  }
}
