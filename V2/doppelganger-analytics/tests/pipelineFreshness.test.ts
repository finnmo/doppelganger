import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import {
  readPipelineFreshness,
  recordGenerateComplete,
  recordImportComplete,
} from '../src/pipeline/freshness.js';

describe('pipeline freshness', () => {
  let db: Database.Database;
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'fresh-'));
    db = new Database(path.join(tmp, 't.db'));
    db.exec(`CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);`);
  });

  afterEach(() => {
    db.close();
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  test('marks stale after import without generate', () => {
    recordImportComplete(db, ['whatsapp'], '2026-07-16T01:00:00.000Z');
    const f = readPipelineFreshness(db);
    expect(f.stale).toBe(true);
    expect(f.lastImportSources).toEqual(['whatsapp']);
  });

  test('clears stale after generate catches up', () => {
    recordImportComplete(db, ['instagram'], '2026-07-16T01:00:00.000Z');
    recordGenerateComplete(db, '2026-07-16T02:00:00.000Z');
    expect(readPipelineFreshness(db).stale).toBe(false);
  });

  test('stale again when a newer import lands', () => {
    recordImportComplete(db, ['instagram'], '2026-07-16T01:00:00.000Z');
    recordGenerateComplete(db, '2026-07-16T02:00:00.000Z');
    recordImportComplete(db, ['imessage'], '2026-07-16T03:00:00.000Z');
    const f = readPipelineFreshness(db);
    expect(f.stale).toBe(true);
    expect(f.reason).toMatch(/newer import/i);
  });
});
