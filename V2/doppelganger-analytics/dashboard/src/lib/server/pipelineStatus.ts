/**
 * Pipeline freshness status for the dashboard (import vs generate).
 */

import fs from 'fs';
import path from 'path';
import { getAnalyticsDbReadonly, resolveAnalyticsDbPath } from './analyticsDb';
import { resolvePersonaProfilesPath } from './personaProfiles';

const META_LAST_IMPORT_AT = 'last_import_at';
const META_LAST_IMPORT_SOURCES = 'last_import_sources';
const META_LAST_GENERATE_AT = 'last_generate_at';

export interface PipelineStatus {
  dbPath: string;
  dbExists: boolean;
  messageCount: number;
  platforms: string[];
  lastImportAt: string | null;
  lastImportSources: string[];
  lastGenerateAt: string | null;
  personaProfilesPath: string | null;
  personaGeneratedAt: string | null;
  stale: boolean;
  reason: string | null;
  generateHint: string;
}

function readMeta(db: import('better-sqlite3').Database, key: string): string | null {
  try {
    const row = db.prepare('SELECT value FROM meta WHERE key = ?').get(key) as
      | { value: string }
      | undefined;
    return row?.value ?? null;
  } catch {
    return null;
  }
}

export function getPipelineStatus(): PipelineStatus {
  const dbPath = resolveAnalyticsDbPath();
  const dbExists = fs.existsSync(dbPath);
  const personaProfilesPath = resolvePersonaProfilesPath();
  let personaGeneratedAt: string | null = null;
  if (personaProfilesPath) {
    try {
      const raw = JSON.parse(fs.readFileSync(personaProfilesPath, 'utf8')) as {
        generatedAt?: string;
      };
      personaGeneratedAt = raw.generatedAt ?? null;
      if (!personaGeneratedAt) {
        personaGeneratedAt = new Date(fs.statSync(personaProfilesPath).mtimeMs).toISOString();
      }
    } catch {
      personaGeneratedAt = null;
    }
  }

  const generateHint = 'npm run generate-metrics';

  if (!dbExists) {
    return {
      dbPath,
      dbExists: false,
      messageCount: 0,
      platforms: [],
      lastImportAt: null,
      lastImportSources: [],
      lastGenerateAt: null,
      personaProfilesPath,
      personaGeneratedAt,
      stale: false,
      reason: null,
      generateHint,
    };
  }

  const db = getAnalyticsDbReadonly();
  let messageCount = 0;
  let platforms: string[] = [];
  let lastImportAt: string | null = null;
  let lastImportSources: string[] = [];
  let lastGenerateAt: string | null = null;

  if (db) {
    try {
      messageCount = (
        db.prepare(`SELECT COUNT(*) AS n FROM messages WHERE is_system = 0`).get() as { n: number }
      ).n;
      platforms = (
        db
          .prepare(
            `SELECT DISTINCT source AS source FROM messages WHERE source IS NOT NULL ORDER BY source`
          )
          .all() as Array<{ source: string }>
      ).map((r) => r.source);
      lastImportAt = readMeta(db, META_LAST_IMPORT_AT);
      const sourcesRaw = readMeta(db, META_LAST_IMPORT_SOURCES);
      lastImportSources = sourcesRaw
        ? sourcesRaw
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
        : [];
      lastGenerateAt = readMeta(db, META_LAST_GENERATE_AT);
    } catch {
      // schema may be incomplete
    }
  }

  // Fall back to file mtimes when meta is missing (older DBs)
  if (!lastImportAt && dbExists) {
    lastImportAt = new Date(fs.statSync(dbPath).mtimeMs).toISOString();
  }
  if (!lastGenerateAt && personaGeneratedAt) {
    lastGenerateAt = personaGeneratedAt;
  }

  let stale = false;
  let reason: string | null = null;

  if (messageCount > 0 && !personaProfilesPath) {
    stale = true;
    reason = 'Message database exists but persona profiles have not been generated yet.';
  } else if (lastImportAt && !lastGenerateAt) {
    stale = true;
    reason = 'Messages were imported but analytics have never been generated.';
  } else if (lastImportAt && lastGenerateAt && lastImportAt > lastGenerateAt) {
    stale = true;
    reason =
      'A newer import landed after the last generate — persona chat and charts may be stale.';
  } else if (dbExists && personaProfilesPath) {
    try {
      const dbMtime = fs.statSync(dbPath).mtimeMs;
      const profilesMtime = fs.statSync(personaProfilesPath).mtimeMs;
      // Only use mtime fallback when meta timestamps are absent/equal-ish
      if (!readMeta(db!, META_LAST_IMPORT_AT) && dbMtime > profilesMtime + 5_000) {
        stale = true;
        reason =
          'The message database looks newer than personaProfiles.json — regenerate metrics.';
      }
    } catch {
      // ignore
    }
  }

  return {
    dbPath,
    dbExists,
    messageCount,
    platforms,
    lastImportAt,
    lastImportSources,
    lastGenerateAt,
    personaProfilesPath,
    personaGeneratedAt,
    stale,
    reason,
    generateHint,
  };
}

/** Resolve dash-data directory for privacy export/wipe. */
export function resolveDashDataDirs(): string[] {
  const dirs: string[] = [];
  if (process.env.DOPPELGANGER_DASH_DIR) {
    dirs.push(path.resolve(process.env.DOPPELGANGER_DASH_DIR));
  }
  const cwd = process.cwd();
  dirs.push(path.join(cwd, 'dash-data'));
  dirs.push(path.join(cwd, '..', 'dash-data'));
  dirs.push(path.join(cwd, 'public', 'data'));
  return [...new Set(dirs.filter((d) => fs.existsSync(d)))];
}
