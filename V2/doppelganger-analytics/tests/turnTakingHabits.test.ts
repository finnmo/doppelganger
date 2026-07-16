import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { buildTurnTakingHabits } from '../src/processors/turnTakingHabits.js';

describe('turnTakingHabits', () => {
  let db: Database.Database;
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'turns-'));
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

  test('computes question and double-text rates', () => {
    const insert = db.prepare(
      `INSERT INTO messages (conversation_id, sender, content, timestamp_ms) VALUES (?, ?, ?, ?)`
    );
    let t = 1;
    for (let i = 0; i < 30; i++) {
      insert.run('c1', 'Sam', `hey whats up ${i}`, t++);
      insert.run('c1', 'Alex', `not much you? ${i}`, t++);
      insert.run('c1', 'Alex', `also saw this lol`, t++);
      t += 100;
    }

    const habits = buildTurnTakingHabits(db, 'Alex');
    expect(habits).not.toBeNull();
    expect(habits!.questionRate).toBeGreaterThan(0.2);
    expect(habits!.doubleTextRate).toBeGreaterThan(0.2);
    expect(habits!.styleSummary).toContain('Turn-taking habits');
  });
});
