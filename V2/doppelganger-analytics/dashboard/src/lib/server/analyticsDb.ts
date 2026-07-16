/**
 * Read-only access to the analytics SQLite DB from Next.js API routes.
 */

import Database, { type Database as DatabaseType } from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';

let cached: { path: string; db: DatabaseType } | null = null;

export function resolveAnalyticsDbPath(): string {
  if (process.env.DOPPELGANGER_DB_PATH) {
    return path.resolve(process.env.DOPPELGANGER_DB_PATH);
  }
  return path.join(os.homedir(), '.doppelgaenger-analytics', 'doppelgaenger-analytics.db');
}

/** Open (or reuse) a read-only connection. Returns null if the DB file is missing. */
export function getAnalyticsDbReadonly(): DatabaseType | null {
  const dbPath = resolveAnalyticsDbPath();
  if (!fs.existsSync(dbPath)) return null;

  if (cached && cached.path === dbPath) {
    try {
      // Probe that the connection is still alive
      cached.db.prepare('SELECT 1').get();
      return cached.db;
    } catch {
      cached = null;
    }
  }

  try {
    const db = new Database(dbPath, { readonly: true, fileMustExist: true });
    db.pragma('busy_timeout = 3000');
    cached = { path: dbPath, db };
    return db;
  } catch {
    return null;
  }
}

export function closeAnalyticsDb(): void {
  if (cached) {
    try {
      cached.db.close();
    } catch {
      // ignore
    }
    cached = null;
  }
}
