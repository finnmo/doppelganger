// src/db/client.ts
import Database, { Database as DatabaseType } from 'better-sqlite3';
import path from 'path';
import os from 'os';
import fs from 'fs';
import { migrate } from './schema.js';

// better-sqlite3 is synchronous, so a single shared connection is both
// simpler and faster than pooling. getDb/closeDb keep async signatures so
// callers don't need to change.
let db: DatabaseType | null = null;

/**
 * Returns the shared Database instance, creating the file in the user's
 * home dir and running migrations on first open.
 */
export async function getDb(): Promise<DatabaseType> {
  if (!db) {
    // Tests point this at a fixture database via DOPPELGANGER_DB_PATH.
    const override = process.env.DOPPELGANGER_DB_PATH;
    let dbPath: string;
    if (override) {
      dbPath = override;
    } else {
      const baseDir = path.join(os.homedir(), '.doppelgaenger-analytics');
      if (!fs.existsSync(baseDir)) {
        fs.mkdirSync(baseDir, { recursive: true });
      }
      dbPath = path.join(baseDir, 'doppelgaenger-analytics.db');
    }

    db = new Database(dbPath, {
      fileMustExist: false,
      timeout: 5000
    });
    db.pragma('journal_mode = WAL');
    db.pragma('busy_timeout = 5000');
    migrate(db);
  }
  return db;
}

/**
 * No-op kept for API compatibility: the shared connection stays open until
 * closeAllConnections() is called.
 */
export async function closeDb(_db: DatabaseType): Promise<void> {}

/**
 * Closes the shared connection (checkpointing the WAL).
 */
export async function closeAllConnections(): Promise<void> {
  if (db) {
    try {
      db.close();
    } catch (err) {
      console.warn('Error closing database connection:', err);
    }
    db = null;
  }
}
