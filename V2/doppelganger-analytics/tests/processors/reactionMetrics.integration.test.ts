import fs from 'fs';
import os from 'os';
import path from 'path';
import Database from 'better-sqlite3';
import { migrate } from '../../src/db/schema.js';

// Isolated fixture DB + output dir, wired via env before importing the
// processor (which reads them through getDb / writeDashData).
let tmpDir: string;
let dbPath: string;
let dashDir: string;

function seed(db: Database.Database) {
  const insertMsg = db.prepare(`
    INSERT INTO messages (id, conversation_id, sender, timestamp_ms, content, has_photos, has_videos, has_audio, has_share, share_link)
    VALUES (@id, @c, @s, @t, @content, 0, 0, 0, 0, NULL)
  `);
  const msgs = [
    { id: 1, c: 'c1', s: 'A', t: 1000, content: 'hi' },
    { id: 2, c: 'c1', s: 'B', t: 2000, content: 'hey' },
    { id: 3, c: 'c2', s: 'X', t: 3000, content: 'yo' }
  ];
  const tx = db.transaction(() => msgs.forEach(m => insertMsg.run(m)));
  tx();

  const insertReaction = db.prepare(`
    INSERT INTO message_reactions (message_id, reaction, actor, timestamp)
    VALUES (@message_id, @reaction, @actor, @timestamp)
  `);
  const reactions = [
    // c1: message 1 (from A) gets ❤️ and ❤ (variation-selector variants) from B
    { message_id: 1, reaction: '❤️', actor: 'B', timestamp: 0 },
    { message_id: 1, reaction: '❤', actor: 'B', timestamp: 0 },
    { message_id: 2, reaction: '😂', actor: 'A', timestamp: 0 }, // 😂 on B's message from A
    // c2: one reaction
    { message_id: 3, reaction: '❤️', actor: 'Y', timestamp: 0 }
  ];
  const tx2 = db.transaction(() => reactions.forEach(r => insertReaction.run(r)));
  tx2();
}

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'reaction-test-'));
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

describe('computeReactionMetrics (integration)', () => {
  test('aggregates real reactions with per-conversation totals and merges heart variants', async () => {
    const { computeReactionMetrics } = await import('../../src/processors/reactionMetrics.js');
    await computeReactionMetrics();

    const output = JSON.parse(fs.readFileSync(path.join(dashDir, 'reactionMetrics.json'), 'utf-8'));

    expect(output.summary.totalReactions).toBe(4);

    // ❤️ and ❤ collapse into one emoji group of count 3
    const heart = output.reactionSummaries.find((r: any) => r.emoji === '❤️' || r.emoji === '❤');
    expect(heart.count).toBe(3);

    // Per-conversation totals sum to the global total
    const byConv: Array<{
      conversation_id: string;
      count: number;
      emoji_counts: Array<{ emoji: string; count: number }>;
      senders: Array<{ sender: string; reactionsGiven: number; reactionsReceived: number }>;
    }> = output.reactionsByConversation;
    expect(byConv.reduce((s, c) => s + c.count, 0)).toBe(4);
    expect(byConv.find(c => c.conversation_id === 'c1')!.count).toBe(3);
    expect(byConv.find(c => c.conversation_id === 'c2')!.count).toBe(1);

    // Per-conversation emoji/sender breakdowns for filtered dashboard views
    const c1 = byConv.find(c => c.conversation_id === 'c1')!;
    expect(c1.emoji_counts.reduce((s, e) => s + e.count, 0)).toBe(3);
    const heartInC1 = c1.emoji_counts.find(e => e.emoji === '❤️' || e.emoji === '❤');
    expect(heartInC1?.count).toBe(2);
    expect(c1.senders.find(s => s.sender === 'B')?.reactionsGiven).toBe(2);
    expect(c1.senders.find(s => s.sender === 'A')?.reactionsReceived).toBe(2);
  });

  test('produces identical output on a second run (deterministic)', async () => {
    const { computeReactionMetrics } = await import('../../src/processors/reactionMetrics.js');
    await computeReactionMetrics();
    const first = fs.readFileSync(path.join(dashDir, 'reactionMetrics.json'), 'utf-8');
    await computeReactionMetrics();
    const second = fs.readFileSync(path.join(dashDir, 'reactionMetrics.json'), 'utf-8');
    expect(first).toBe(second);
  });
});
