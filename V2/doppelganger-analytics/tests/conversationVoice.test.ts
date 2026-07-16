import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  buildConversationVoices,
  buildPlatformVoices,
  findConversationVoice,
  findPlatformVoice,
} from '../src/processors/conversationVoice.js';

describe('conversationVoice', () => {
  let db: Database.Database;
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'conv-voice-'));
    db = new Database(path.join(tmp, 't.db'));
    db.exec(`
      CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        conversation_id TEXT,
        sender TEXT,
        content TEXT,
        timestamp_ms INTEGER,
        is_system INTEGER DEFAULT 0,
        source TEXT DEFAULT 'instagram'
      );
      CREATE TABLE text_metrics (
        message_id INTEGER PRIMARY KEY,
        word_count INTEGER,
        emoji_count REAL
      );
    `);
  });

  afterEach(() => {
    db.close();
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  test('builds distinct DM vs group voice summaries', () => {
    const insertMsg = db.prepare(
      `INSERT INTO messages (conversation_id, sender, content, timestamp_ms, source)
       VALUES (?, ?, ?, ?, 'instagram')`
    );
    const insertTm = db.prepare(
      `INSERT INTO text_metrics (message_id, word_count, emoji_count) VALUES (?, ?, ?)`
    );

    // DM: short messages
    for (let i = 0; i < 30; i++) {
      const a = insertMsg.run('dm1', 'Alex', 'ok', i);
      insertTm.run(Number(a.lastInsertRowid), 1, 0);
      insertMsg.run('dm1', 'Sam', 'cool', i);
    }

    // Group: longer + emoji
    for (let i = 0; i < 30; i++) {
      const a = insertMsg.run(
        'group1',
        'Alex',
        'haha that was so funny everyone should see this lol 😂',
        1000 + i
      );
      insertTm.run(Number(a.lastInsertRowid), 12, 1);
      insertMsg.run('group1', 'Sam', 'yes', 1000 + i);
      insertMsg.run('group1', 'Pat', 'lol', 1000 + i);
      insertMsg.run('group1', 'Jo', 'same', 1000 + i);
    }

    const voices = buildConversationVoices(db, 'Alex');
    expect(voices.length).toBeGreaterThanOrEqual(2);

    const dm = findConversationVoice(voices, 'dm1');
    const group = findConversationVoice(voices, 'group1');
    expect(dm?.chatType).toBe('dm');
    expect(group?.chatType).toBe('small_group');
    expect(dm?.styleSummary).toMatch(/1:1 DM/i);
    expect(group?.styleSummary).toMatch(/group/i);
    expect(group!.avgWordsPerMessage).toBeGreaterThan(dm!.avgWordsPerMessage);
  });

  test('builds per-platform voice cards', () => {
    const insertMsg = db.prepare(
      `INSERT INTO messages (conversation_id, sender, content, timestamp_ms, source)
       VALUES (?, ?, ?, ?, ?)`
    );
    const insertTm = db.prepare(
      `INSERT INTO text_metrics (message_id, word_count, emoji_count) VALUES (?, ?, ?)`
    );

    for (let i = 0; i < 50; i++) {
      const a = insertMsg.run('instagram:dm', 'Alex', 'lol ok', i, 'instagram');
      insertTm.run(Number(a.lastInsertRowid), 2, 0);
      insertMsg.run('instagram:dm', 'Sam', 'hi', i, 'instagram');
    }
    for (let i = 0; i < 50; i++) {
      const a = insertMsg.run(
        'whatsapp:dm',
        'Alex',
        'hey how are you doing today hope you are well',
        1000 + i,
        'whatsapp'
      );
      insertTm.run(Number(a.lastInsertRowid), 10, 0.2);
      insertMsg.run('whatsapp:dm', 'Sam', 'good', 1000 + i, 'whatsapp');
    }

    const platforms = buildPlatformVoices(db, 'Alex');
    expect(platforms.map((p) => p.source).sort()).toEqual(['instagram', 'whatsapp']);
    const ig = findPlatformVoice(platforms, 'instagram');
    const wa = findPlatformVoice(platforms, 'whatsapp');
    expect(ig?.styleSummary).toMatch(/Instagram/i);
    expect(wa?.styleSummary).toMatch(/WhatsApp/i);
    expect(wa!.avgWordsPerMessage).toBeGreaterThan(ig!.avgWordsPerMessage);
  });
});
