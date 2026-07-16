import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { buildBubbleHabits, splitReplyBubbles } from '../src/processors/bubbleHabits.js';

describe('bubbleHabits', () => {
  let db: Database.Database;
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'bubbles-'));
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

  test('detects multi-bubble turns', () => {
    const insert = db.prepare(
      `INSERT INTO messages (conversation_id, sender, content, timestamp_ms) VALUES (?, ?, ?, ?)`
    );
    let t = 1_000_000;
    for (let i = 0; i < 20; i++) {
      insert.run('c1', 'Alex', `bubble a ${i}`, t);
      t += 5_000;
      insert.run('c1', 'Alex', `bubble b ${i}`, t);
      t += 5_000;
      insert.run('c1', 'Sam', 'ok', t);
      t += 120_000;
    }

    const habits = buildBubbleHabits(db, 'Alex');
    expect(habits).not.toBeNull();
    expect(habits!.multiBubbleRate).toBeGreaterThan(0.5);
    expect(habits!.avgBubblesPerTurn).toBeGreaterThan(1.5);
    expect(habits!.styleSummary).toContain('<<<BUBBLE>>>');
  });

  test('splitReplyBubbles parses delimiter and blank lines', () => {
    expect(splitReplyBubbles('one <<<BUBBLE>>> two <<<BUBBLE>>> three')).toEqual([
      'one',
      'two',
      'three',
    ]);
    expect(splitReplyBubbles('just one')).toEqual(['just one']);
  });
});
