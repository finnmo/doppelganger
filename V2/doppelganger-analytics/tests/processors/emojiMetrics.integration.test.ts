import fs from 'fs';
import os from 'os';
import path from 'path';
import Database from 'better-sqlite3';
import { migrate } from '../../src/db/schema.js';

let tmpDir: string;
let dbPath: string;
let dashDir: string;

function seed(db: Database.Database) {
  const insertMsg = db.prepare(`
    INSERT INTO messages (id, conversation_id, sender, timestamp_ms, content, has_photos, has_videos, has_audio, has_share, share_link)
    VALUES (@id, @c, @s, @t, @content, 0, 0, 0, 0, NULL)
  `);
  const msgs = [
    { id: 1, c: 'c1', s: 'A', t: 1000, content: 'hello 😀😀' },
    { id: 2, c: 'c1', s: 'B', t: 2000, content: 'hey 😂' },
    { id: 3, c: 'c1', s: 'A', t: 3000, content: 'no emojis here' },
    { id: 4, c: 'c2', s: 'A', t: 4000, content: 'other chat 😀' }
  ];
  const tx = db.transaction(() => msgs.forEach(m => insertMsg.run(m)));
  tx();

  // text_metrics is the authoritative per-message emoji count source.
  const insertMetric = db.prepare(`
    INSERT INTO text_metrics (message_id, word_count, emoji_count, url_count)
    VALUES (@id, @w, @e, @u)
  `);
  const metrics = [
    { id: 1, w: 1, e: 2, u: 0 },
    { id: 2, w: 1, e: 1, u: 0 },
    { id: 3, w: 3, e: 0, u: 0 },
    { id: 4, w: 2, e: 1, u: 0 }
  ];
  const tx2 = db.transaction(() => metrics.forEach(m => insertMetric.run(m)));
  tx2();
}

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'emoji-test-'));
  dbPath = path.join(tmpDir, 'fixture.db');
  dashDir = path.join(tmpDir, 'dash-data');
  const db = new Database(dbPath);
  migrate(db);
  seed(db);
  db.close();

  process.env.DOPPELGANGER_DB_PATH = dbPath;
  process.env.DOPPELGANGER_DASH_DIR = dashDir;
});

afterAll(async () => {
  const { closeAllConnections } = await import('../../src/db/client.js');
  await closeAllConnections();
  delete process.env.DOPPELGANGER_DB_PATH;
  delete process.env.DOPPELGANGER_DASH_DIR;
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('computeEmojiMetrics (integration)', () => {
  test('per-conversation per-sender counts match text_metrics and include top emojis', async () => {
    const { computeEmojiMetrics } = await import('../../src/processors/emojiMetrics.js');
    await computeEmojiMetrics();

    const output = JSON.parse(fs.readFileSync(path.join(dashDir, 'emojiMetrics.json'), 'utf-8'));

    expect(output.summary.totalEmojis).toBe(4);
    expect(output.summary.totalSenders).toBe(2);

    const rows: Array<{
      conversation_id: string;
      sender: string;
      emoji_count: number;
      top_emojis: Array<{ emoji: string; count: number }>;
    }> = output.senderEmojis;

    // Only sender+conversation pairs that actually used emojis are exported.
    expect(rows).toHaveLength(3);

    const aInC1 = rows.find(r => r.conversation_id === 'c1' && r.sender === 'A')!;
    expect(aInC1.emoji_count).toBe(2);
    expect(aInC1.top_emojis[0]).toEqual({ emoji: '😀', count: 2 });

    const bInC1 = rows.find(r => r.conversation_id === 'c1' && r.sender === 'B')!;
    expect(bInC1.emoji_count).toBe(1);

    const aInC2 = rows.find(r => r.conversation_id === 'c2' && r.sender === 'A')!;
    expect(aInC2.emoji_count).toBe(1);

    // Filtered (per-conversation) counts sum to the global total.
    expect(rows.reduce((sum, r) => sum + r.emoji_count, 0)).toBe(output.summary.totalEmojis);
  });
});
