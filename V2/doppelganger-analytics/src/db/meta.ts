import { Database } from 'better-sqlite3';

/**
 * Small helpers over the `meta` key-value table. Used to record which version
 * of a derived-data engine produced the current rows, so a change to that
 * engine can invalidate and recompute stale data automatically.
 */

export function getMeta(db: Database, key: string): string | null {
  const row = db.prepare('SELECT value FROM meta WHERE key = ?').get(key) as
    | { value: string }
    | undefined;
  return row ? row.value : null;
}

export function setMeta(db: Database, key: string, value: string): void {
  db.prepare(
    `INSERT INTO meta (key, value) VALUES (?, ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value`
  ).run(key, value);
}
