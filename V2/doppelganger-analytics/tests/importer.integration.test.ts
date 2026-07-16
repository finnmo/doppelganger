import fs from 'fs';
import os from 'os';
import path from 'path';

// End-to-end test of the platform-neutral import pipeline using a minimal
// on-disk Instagram export layout (messages/inbox/<conv>/message_1.json).

let tmpDir: string;
let exportDir: string;
let dbPath: string;

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'importer-test-'));
  dbPath = path.join(tmpDir, 'fixture.db');
  exportDir = path.join(tmpDir, 'export');

  const convDir = path.join(exportDir, 'messages', 'inbox', 'alice_123');
  fs.mkdirSync(convDir, { recursive: true });
  fs.writeFileSync(path.join(convDir, 'message_1.json'), JSON.stringify({
    participants: [{ name: 'Alice' }, { name: 'Bob' }],
    messages: [
      {
        sender_name: 'Alice',
        timestamp_ms: 2000,
        content: 'hello there',
        reactions: [{ reaction: '\u00e2\u009d\u00a4', actor: 'Bob' }]
      },
      {
        sender_name: 'Bob',
        timestamp_ms: 1000,
        photos: [{ uri: 'photos/img.jpg', creation_timestamp: 1 }]
      }
    ]
  }));

  process.env.DOPPELGANGER_DB_PATH = dbPath;
});

afterAll(async () => {
  const { closeAllConnections } = await import('../src/db/client.js');
  await closeAllConnections();
  delete process.env.DOPPELGANGER_DB_PATH;
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('importArchive (platform-neutral pipeline)', () => {
  test('detects Instagram layout and inserts normalized rows with source', async () => {
    const { importArchive } = await import('../src/importer.js');
    await importArchive(exportDir);

    const { getDb } = await import('../src/db/client.js');
    const db = await getDb();

    const messages = db.prepare(
      'SELECT conversation_id, sender, content, has_photos, source, is_system FROM messages ORDER BY timestamp_ms'
    ).all() as Array<{
      conversation_id: string;
      sender: string;
      content: string | null;
      has_photos: number;
      source: string;
      is_system: number;
    }>;

    expect(messages).toHaveLength(2);
    expect(messages.every(m => m.source === 'instagram')).toBe(true);
    expect(messages.every(m => m.is_system === 0)).toBe(true);
    expect(messages.every(m => m.conversation_id === 'instagram:alice_123')).toBe(true);

    // Attachment-only message gets synthetic content and the media flag.
    expect(messages[0].sender).toBe('Bob');
    expect(messages[0].has_photos).toBe(1);
    expect(messages[0].content).toBe('Bob sent 1 photo');

    expect(messages[1].content).toBe('hello there');

    // Photo row and decoded reaction present.
    const photoCount = (db.prepare('SELECT COUNT(*) AS n FROM message_photos').get() as { n: number }).n;
    expect(photoCount).toBe(1);
    const reaction = db.prepare('SELECT reaction, actor FROM message_reactions').get() as { reaction: string; actor: string };
    expect(reaction.actor).toBe('Bob');
    expect(reaction.reaction).toBe('❤'); // malformed \u00e2\u009d\u00a4 decoded
  });

  test('unrecognized layouts fail with a clear message', async () => {
    const emptyDir = path.join(tmpDir, 'not-an-export');
    fs.mkdirSync(path.join(emptyDir, 'random'), { recursive: true });
    fs.writeFileSync(path.join(emptyDir, 'random', 'notes.txt'), 'hi');

    const { importArchive } = await import('../src/importer.js');
    await expect(importArchive(emptyDir)).rejects.toThrow(/Could not detect the export format/);
  });

  test('ZIP import extracts every conversation folder', async () => {
    const { execFileSync } = await import('child_process');
    const zipRoot = path.join(tmpDir, 'zip-export');
    for (const id of ['chat_one', 'chat_two']) {
      const dir = path.join(zipRoot, 'messages', 'inbox', id);
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, 'message_1.json'), JSON.stringify({
        messages: [
          { sender_name: 'A', timestamp_ms: 1000, content: `hi ${id}` },
          { sender_name: 'B', timestamp_ms: 2000, content: `yo ${id}` }
        ]
      }));
    }
    const zipPath = path.join(tmpDir, 'two-chats.zip');
    execFileSync('zip', ['-qr', zipPath, '.'], { cwd: zipRoot });

    // Fresh DB so this case does not see rows from the folder-import test.
    const zipDb = path.join(tmpDir, 'zip.db');
    process.env.DOPPELGANGER_DB_PATH = zipDb;
    const { closeAllConnections } = await import('../src/db/client.js');
    await closeAllConnections();

    const { importArchive } = await import('../src/importer.js');
    await importArchive(zipPath);

    const { getDb } = await import('../src/db/client.js');
    const db = await getDb();
    const rows = db.prepare(
      'SELECT conversation_id, COUNT(*) AS n FROM messages GROUP BY conversation_id ORDER BY conversation_id'
    ).all() as Array<{ conversation_id: string; n: number }>;

    expect(rows).toEqual([
      { conversation_id: 'instagram:chat_one', n: 2 },
      { conversation_id: 'instagram:chat_two', n: 2 }
    ]);

    // Restore DB path for any later tests in this file.
    process.env.DOPPELGANGER_DB_PATH = dbPath;
    await closeAllConnections();
  });

  test('detects WhatsApp _chat.txt and parses bracket-format messages', async () => {
    const waDir = path.join(tmpDir, 'whatsapp-export');
    fs.mkdirSync(waDir, { recursive: true });
    fs.writeFileSync(path.join(waDir, '_chat.txt'), [
      '[15/03/24, 10:30:45] Alice: hello there',
      '[15/03/24, 10:31:00] Bob: hi back',
      '[15/03/24, 10:31:15] Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.',
      '[15/03/24, 10:32:00] Alice: sounds good'
    ].join('\n'));

    const waDb = path.join(tmpDir, 'wa.db');
    process.env.DOPPELGANGER_DB_PATH = waDb;
    const { closeAllConnections } = await import('../src/db/client.js');
    await closeAllConnections();

    const { importArchive } = await import('../src/importer.js');
    await importArchive(waDir);

    const { getDb } = await import('../src/db/client.js');
    const db = await getDb();
    const messages = db.prepare(
      'SELECT sender, content, source, is_system FROM messages ORDER BY timestamp_ms'
    ).all() as Array<{ sender: string; content: string | null; source: string; is_system: number }>;

    expect(messages).toHaveLength(4);
    expect(messages.every(m => m.source === 'whatsapp')).toBe(true);
    expect(messages[0].content).toBe('hello there');
    expect(messages[1].content).toBe('hi back');
    expect(messages[2].is_system).toBe(1);
    expect(messages[3].content).toBe('sounds good');

    process.env.DOPPELGANGER_DB_PATH = dbPath;
    await closeAllConnections();
  });

  test('detects iMessage chat.db and maps handles', async () => {
    const imDir = path.join(tmpDir, 'imessage-export');
    fs.mkdirSync(imDir, { recursive: true });
    const chatDbPath = path.join(imDir, 'chat.db');

    const Database = (await import('better-sqlite3')).default;
    const db = new Database(chatDbPath);
    db.exec(`
      CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
      CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
      CREATE TABLE message (
        ROWID INTEGER PRIMARY KEY,
        text TEXT,
        date INTEGER,
        is_from_me INTEGER,
        handle_id INTEGER,
        is_system_message INTEGER DEFAULT 0,
        item_type INTEGER DEFAULT 0,
        attributedBody BLOB
      );
      CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
    `);
    db.prepare('INSERT INTO chat (chat_identifier, display_name) VALUES (?, ?)').run('+15551234', 'Sam');
    db.prepare('INSERT INTO handle (id) VALUES (?)').run('+15551234');
    // Nanoseconds since Apple epoch (2001-01-01), matching real chat.db magnitudes.
    const t1 = 738_000_000_000_000_000;
    const t2 = 738_000_000_001_000_000;
    db.prepare(`
      INSERT INTO message (text, date, is_from_me, handle_id, is_system_message, item_type)
      VALUES (?, ?, ?, ?, 0, 0)
    `).run('hey from sam', t1, 0, 1);
    db.prepare(`
      INSERT INTO message (text, date, is_from_me, handle_id, is_system_message, item_type)
      VALUES (?, ?, ?, ?, 0, 0)
    `).run('reply from me', t2, 1, null);
    db.prepare('INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 1)').run();
    db.prepare('INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 2)').run();
    db.close();

    const imDb = path.join(tmpDir, 'im.db');
    process.env.DOPPELGANGER_DB_PATH = imDb;
    const { closeAllConnections } = await import('../src/db/client.js');
    await closeAllConnections();

    const { importArchive } = await import('../src/importer.js');
    await importArchive(imDir);

    const { getDb } = await import('../src/db/client.js');
    const sqlDb = await getDb();
    const messages = sqlDb.prepare(
      'SELECT sender, content, source FROM messages ORDER BY timestamp_ms'
    ).all() as Array<{ sender: string; content: string | null; source: string }>;

    expect(messages).toHaveLength(2);
    expect(messages.every(m => m.source === 'imessage')).toBe(true);
    expect(messages[0].sender).toBe('+15551234');
    expect(messages[0].content).toBe('hey from sam');
    expect(messages[1].sender).toBe('Me');
    expect(messages[1].content).toBe('reply from me');

    process.env.DOPPELGANGER_DB_PATH = dbPath;
    await closeAllConnections();
  });

  test('tags Messenger conversations separately from Instagram paths', async () => {
    const metaDir = path.join(tmpDir, 'meta-mixed');
    const igDir = path.join(metaDir, 'your_instagram_activity', 'messages', 'inbox', 'ig_friend');
    const fbDir = path.join(metaDir, 'your_facebook_activity', 'messages', 'inbox', 'fb_friend');
    for (const [dir, label] of [[igDir, 'ig'], [fbDir, 'fb']] as const) {
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(path.join(dir, 'message_1.json'), JSON.stringify({
        messages: [{ sender_name: 'A', timestamp_ms: 1000, content: `hello ${label}` }]
      }));
    }

    const metaDb = path.join(tmpDir, 'meta-mixed.db');
    process.env.DOPPELGANGER_DB_PATH = metaDb;
    const { closeAllConnections } = await import('../src/db/client.js');
    await closeAllConnections();

    const { importArchive } = await import('../src/importer.js');
    await importArchive(metaDir);

    const { getDb } = await import('../src/db/client.js');
    const db = await getDb();
    const rows = db.prepare(
      'SELECT conversation_id, source, content FROM messages ORDER BY conversation_id'
    ).all() as Array<{ conversation_id: string; source: string; content: string | null }>;

    expect(rows).toHaveLength(2);
    const ig = rows.find(r => r.conversation_id === 'instagram:ig_friend');
    const fb = rows.find(r => r.conversation_id === 'messenger:fb_friend');
    expect(ig?.source).toBe('instagram');
    expect(fb?.source).toBe('messenger');

    process.env.DOPPELGANGER_DB_PATH = dbPath;
    await closeAllConnections();
  });

  test('importing a second platform keeps the first (merge by source)', async () => {
    const mergeDb = path.join(tmpDir, 'merge.db');
    process.env.DOPPELGANGER_DB_PATH = mergeDb;
    const { closeAllConnections } = await import('../src/db/client.js');
    await closeAllConnections();

    const igDir = path.join(tmpDir, 'merge-ig');
    const convDir = path.join(igDir, 'messages', 'inbox', 'ig_only');
    fs.mkdirSync(convDir, { recursive: true });
    fs.writeFileSync(path.join(convDir, 'message_1.json'), JSON.stringify({
      messages: [{ sender_name: 'A', timestamp_ms: 1000, content: 'from ig' }]
    }));

    const { importArchive } = await import('../src/importer.js');
    await importArchive(igDir);

    const waDir = path.join(tmpDir, 'merge-wa');
    fs.mkdirSync(waDir, { recursive: true });
    fs.writeFileSync(path.join(waDir, '_chat.txt'), [
      '[15/03/24, 10:30:45] Alice: from wa'
    ].join('\n'));
    await importArchive(waDir);

    const { getDb } = await import('../src/db/client.js');
    const db = await getDb();
    const bySource = db.prepare(
      'SELECT source, COUNT(*) AS n FROM messages GROUP BY source ORDER BY source'
    ).all() as Array<{ source: string; n: number }>;

    expect(bySource).toEqual([
      { source: 'instagram', n: 1 },
      { source: 'whatsapp', n: 1 }
    ]);

    process.env.DOPPELGANGER_DB_PATH = dbPath;
    await closeAllConnections();
  });
});
