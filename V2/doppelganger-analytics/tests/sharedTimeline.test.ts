import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { buildSharedTimeline } from '../src/processors/sharedTimeline.js';

describe('sharedTimeline', () => {
  let db: Database.Database;
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'timeline-'));
    db = new Database(path.join(tmp, 't.db'));
    db.exec(`
      CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        conversation_id TEXT,
        sender TEXT,
        content TEXT,
        timestamp_ms INTEGER,
        is_system INTEGER DEFAULT 0
      );
    `);
  });

  afterEach(() => {
    db.close();
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  test('pins high-signal shared life facts', () => {
    const insert = db.prepare(
      `INSERT INTO messages (conversation_id, sender, content, timestamp_ms) VALUES (?, ?, ?, ?)`
    );
    const base = Date.parse('2022-01-01');
    for (let i = 0; i < 25; i++) {
      insert.run('dm', 'Finn', `hey ${i}`, base + i * 1000);
      insert.run('dm', 'Tia', `ok ${i}`, base + i * 1000 + 1);
    }
    insert.run(
      'dm',
      'Tia',
      'Remember that melbourne trip last year when we almost missed the flight',
      base + 50_000
    );
    insert.run('dm', 'Finn', 'hahaha yes', base + 51_000);
    insert.run(
      'dm',
      'Tia',
      'Also when we went to that birthday wedding thing in sydney it was chaos',
      base + 60_000
    );
    insert.run('dm', 'Finn', 'true', base + 61_000);
    insert.run(
      'dm',
      'Tia',
      'Thinking about moving apartments near campus after graduation',
      base + 70_000
    );

    const timeline = buildSharedTimeline(db, 'Tia', 'Finn', 8);
    expect(timeline).not.toBeNull();
    expect(timeline!.summary).toContain('Shared timeline');
    expect(timeline!.facts.length).toBeGreaterThanOrEqual(2);
    expect(timeline!.facts.some((f) => /melbourne|sydney|apartment|flight/i.test(f.text))).toBe(
      true
    );
  });
});
