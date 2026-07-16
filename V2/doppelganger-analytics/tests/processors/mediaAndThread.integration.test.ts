import fs from 'fs';
import os from 'os';
import path from 'path';
import Database from 'better-sqlite3';
import { migrate } from '../../src/db/schema.js';

// Isolated fixture DB + output dir, wired via env before importing processors.
let tmpDir: string;
let dbPath: string;
let dashDir: string;

const MIN = 60_000;
const base = 1_700_000_000_000;

function seed(db: Database.Database) {
  const insertMsg = db.prepare(`
    INSERT INTO messages (id, conversation_id, sender, timestamp_ms, content, has_photos, has_videos, has_audio, has_share, share_link)
    VALUES (@id, @c, @s, @t, @content, @hp, @hv, @ha, @hsh, @link)
  `);
  const msg = (o: Partial<{ id: number; c: string; s: string; t: number; content: string; hp: number; hv: number; ha: number; hsh: number; link: string }>) =>
    insertMsg.run({
      id: o.id, c: o.c ?? 'c1', s: o.s ?? 'A', t: o.t ?? base, content: o.content ?? null,
      hp: o.hp ?? 0, hv: o.hv ?? 0, ha: o.ha ?? 0, hsh: o.hsh ?? 0, link: o.link ?? null
    });

  const tx = db.transaction(() => {
    // A rapid burst of 3 messages (gaps < 5 min) then a long gap then 2 more
    msg({ id: 1, s: 'A', t: base, content: 'hey' });
    msg({ id: 2, s: 'B', t: base + 1 * MIN, content: 'yo' });
    msg({ id: 3, s: 'A', t: base + 2 * MIN, content: 'sup', hp: 1 });           // 1 photo
    msg({ id: 4, s: 'B', t: base + 60 * MIN, content: 'later', hv: 1 });         // new burst (gap>5m); 1 video
    msg({ id: 5, s: 'A', t: base + 61 * MIN, content: 'A sent an attachment' }); // attachment via content
    msg({ id: 6, s: 'B', t: base + 62 * MIN, content: 'voice', ha: 1 });         // audio
    msg({ id: 7, s: 'A', t: base + 63 * MIN, content: 'link', hsh: 1, link: 'https://x.com/p' }); // share
  });
  tx();

  db.prepare('INSERT INTO message_photos (message_id, uri, creation_timestamp) VALUES (3, \'p.jpg\', 0)').run();
  db.prepare('INSERT INTO message_videos (message_id, uri, creation_timestamp) VALUES (4, \'v.mp4\', 0)').run();
  db.prepare('INSERT INTO message_audio (message_id, uri, creation_timestamp) VALUES (6, \'a.aac\', 0)').run();
}

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'media-thread-test-'));
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

describe('enhancedMediaProcessor (integration)', () => {
  test('counts photos/videos/attachments from the real tables and flags', async () => {
    const { generateEnhancedMediaData } = await import('../../src/processors/enhancedMediaProcessor.js');
    await generateEnhancedMediaData();
    const out = JSON.parse(fs.readFileSync(path.join(dashDir, 'mediaMetrics.json'), 'utf-8'));

    expect(out.media_type_distribution).toEqual({ photos: 1, videos: 1, attachments: 1 });
    // photo(3) + video(4) + attachment(5) = 3 media messages out of 7 total
    expect(out.summary.total_media_messages).toBe(3);
    expect(Math.round(out.summary.media_percentage)).toBe(43); // 3/7
  });
});

describe('attachmentTypeMetrics (integration)', () => {
  test('classifies audio and link (share) from flags', async () => {
    const { computeAttachmentTypeMetrics } = await import('../../src/processors/attachmentTypeMetrics.js');
    await computeAttachmentTypeMetrics();
    const rows = JSON.parse(fs.readFileSync(path.join(dashDir, 'attachmentTypeMetrics.json'), 'utf-8'));
    const byType: Record<string, number> = {};
    for (const r of rows) byType[r.type] = (byType[r.type] || 0) + r.count;

    expect(byType.image).toBe(1);
    expect(byType.video).toBe(1);
    expect(byType.audio).toBe(1);
    expect(byType.link).toBe(1);
    expect(byType.document).toBe(1); // the 'sent an attachment' message
  });
});

describe('threadAnalysisMetrics (integration)', () => {
  test('reconstructs threads from timing bursts', async () => {
    const { computeThreadAnalysisMetrics } = await import('../../src/processors/threadAnalysisMetrics.js');
    await computeThreadAnalysisMetrics();
    const out = JSON.parse(fs.readFileSync(path.join(dashDir, 'threadAnalysis.json'), 'utf-8'));

    // Two bursts: [1,2,3] (depth 3) and [4,5,6,7] (depth 4)
    expect(out.summary.total_threads).toBe(2);
    expect(out.summary.max_depth).toBe(4);
    expect(out.summary.threaded_conversations).toBe(1);
  });
});
