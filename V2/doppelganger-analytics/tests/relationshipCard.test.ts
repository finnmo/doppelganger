import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  buildRelationshipCard,
  inferSelfSender,
} from '../src/processors/relationshipCard.js';

describe('relationshipCard', () => {
  let db: Database.Database;
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rel-card-'));
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
      CREATE TABLE sentiment (
        message_id INTEGER PRIMARY KEY,
        compound REAL
      );
    `);
  });

  afterEach(() => {
    db.close();
    fs.rmSync(tmp, { recursive: true, force: true });
    delete process.env.DOPPELGANGER_SELF_NAME;
  });

  function seed() {
    const insert = db.prepare(
      `INSERT INTO messages (conversation_id, sender, content, timestamp_ms, is_system)
       VALUES (?, ?, ?, ?, 0)`
    );
    const sent = db.prepare(`INSERT INTO sentiment (message_id, compound) VALUES (?, ?)`);

    // Finn in many dyads → inferred self
    for (let i = 0; i < 5; i++) {
      insert.run(`dm_${i}`, 'Finn Morris', `hi from finn ${i}`, i * 10);
      insert.run(`dm_${i}`, 'Other Person', `yo ${i}`, i * 10 + 1);
    }

    // Tia + Finn main thread
    const lines = [
      ['Tia Shannon', 'hey finn want coffee in Melbourne?'],
      ['Finn Morris', 'yeah down'],
      ['Tia Shannon', 'babe also Sam is coming'],
      ['Finn Morris', 'cool'],
      ['Tia Shannon', 'love the cafe near campus'],
      ['Finn Morris', 'same'],
      ['Tia Shannon', 'hi Finn — trip photos later'],
      ['Finn Morris', 'send them'],
      ['Tia Shannon', 'Sam said sydney was wild'],
      ['Finn Morris', 'haha'],
      ['Tia Shannon', 'babe see you at the gym'],
      ['Finn Morris', 'ok'],
      ['Tia Shannon', 'morning Finn'],
      ['Finn Morris', 'morning'],
      ['Tia Shannon', 'coffee again?'],
      ['Finn Morris', 'sure'],
    ];
    let id = 100;
    for (const [sender, content] of lines) {
      const info = insert.run('main', sender, content, id);
      if (sender === 'Tia Shannon') {
        sent.run(Number(info.lastInsertRowid), 0.4);
      }
      id += 1;
    }
  }

  test('inferSelfSender picks person in most dyads', () => {
    seed();
    expect(inferSelfSender(db)).toBe('Finn Morris');
  });

  test('DOPPELGANGER_SELF_NAME overrides inference', () => {
    seed();
    process.env.DOPPELGANGER_SELF_NAME = 'Tia Shannon';
    expect(inferSelfSender(db)).toBe('Tia Shannon');
  });

  test('buildRelationshipCard extracts address, places, people, tone, and with-you register', () => {
    seed();
    const card = buildRelationshipCard(db, 'Tia Shannon', 'Finn Morris');
    expect(card).not.toBeNull();
    if (!card) return;

    expect(card.withPerson).toBe('Finn Morris');
    expect(card.addressForms.some((a) => /finn|babe|love/i.test(a))).toBe(true);
    expect(card.recurringPlaces.some((p) => /melbourne|cafe|gym|sydney|campus|coffee/i.test(p))).toBe(
      true
    );
    expect(card.summary).toContain('Relationship with Finn Morris');
    expect(card.registerSummary).toContain('How Tia Shannon texts Finn Morris');
    expect(card.sharedConversationIds.length).toBeGreaterThan(0);
    expect(typeof card.questionBackRate).toBe('number');
    expect(card.avgWordsWithYou).toBeGreaterThan(0);
  });
});
