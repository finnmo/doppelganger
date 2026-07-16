/**
 * Pipeline freshness: import vs generate timestamps in the meta table.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import { getMeta, setMeta } from '../db/meta.js';

export const META_LAST_IMPORT_AT = 'last_import_at';
export const META_LAST_IMPORT_SOURCES = 'last_import_sources';
export const META_LAST_GENERATE_AT = 'last_generate_at';

export function recordImportComplete(
  db: DatabaseType,
  sources: string[],
  at = new Date().toISOString()
): void {
  setMeta(db, META_LAST_IMPORT_AT, at);
  setMeta(db, META_LAST_IMPORT_SOURCES, sources.join(','));
}

export function recordGenerateComplete(db: DatabaseType, at = new Date().toISOString()): void {
  setMeta(db, META_LAST_GENERATE_AT, at);
}

export interface PipelineFreshness {
  lastImportAt: string | null;
  lastImportSources: string[];
  lastGenerateAt: string | null;
  /** True when an import happened after the last successful generate (or never generated). */
  stale: boolean;
  reason: string | null;
}

export function readPipelineFreshness(db: DatabaseType): PipelineFreshness {
  const lastImportAt = getMeta(db, META_LAST_IMPORT_AT);
  const sourcesRaw = getMeta(db, META_LAST_IMPORT_SOURCES);
  const lastGenerateAt = getMeta(db, META_LAST_GENERATE_AT);
  const lastImportSources = sourcesRaw
    ? sourcesRaw
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    : [];

  if (!lastImportAt && !lastGenerateAt) {
    return {
      lastImportAt: null,
      lastImportSources,
      lastGenerateAt: null,
      stale: false,
      reason: null,
    };
  }

  if (lastImportAt && !lastGenerateAt) {
    return {
      lastImportAt,
      lastImportSources,
      lastGenerateAt: null,
      stale: true,
      reason: 'Messages were imported but analytics have never been generated.',
    };
  }

  if (lastImportAt && lastGenerateAt && lastImportAt > lastGenerateAt) {
    return {
      lastImportAt,
      lastImportSources,
      lastGenerateAt,
      stale: true,
      reason:
        'A newer import landed after the last generate — persona profiles and charts may be stale. Run `npm run generate-metrics`.',
    };
  }

  return {
    lastImportAt,
    lastImportSources,
    lastGenerateAt,
    stale: false,
    reason: null,
  };
}
