#!/usr/bin/env node
/**
 * One-click Docker entrypoint.
 *
 * 1. Looks for an Instagram export under /export (ZIP or extracted folder)
 * 2. Imports + generates metrics if the DB is empty or EXPORT_PATH is set
 * 3. Syncs dash-data → dashboard/public/data
 * 4. Starts the Next.js production server on :3000
 */
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { createRequire } from 'module';

const EXPORT_DIR = process.env.EXPORT_DIR || '/export';
const FORCE_REIMPORT =
  process.env.FORCE_REIMPORT === '1' ||
  process.env.FORCE_REIMPORT === 'true' ||
  process.env.REIMPORT === '1' ||
  process.env.REIMPORT === 'true';
const SKIP_IMPORT = process.env.SKIP_IMPORT === '1' || process.env.SKIP_IMPORT === 'true';
const PORT = process.env.PORT || '3000';
const DB_DIR = path.join(process.env.HOME || '/root', '.doppelgaenger-analytics');
const DB_PATH = path.join(DB_DIR, 'doppelgaenger-analytics.db');

function log(msg) {
  console.log(`[doppel] ${msg}`);
}

function findExport(dir) {
  if (!fs.existsSync(dir)) return null;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const zips = entries
    .filter((e) => e.isFile() && e.name.toLowerCase().endsWith('.zip'))
    .map((e) => path.join(dir, e.name))
    .sort();

  if (zips.length > 0) {
    if (zips.length > 1) {
      log(`Multiple ZIPs found; using ${path.basename(zips[0])}`);
    }
    return zips[0];
  }

  // Already-extracted Instagram tree (messages/inbox/… or message_*.json)
  const hasMessages = entries.some(
    (e) =>
      (e.isDirectory() && (e.name === 'messages' || e.name === 'inbox')) ||
      (e.isFile() && e.name.startsWith('message_') && e.name.endsWith('.json'))
  );
  if (hasMessages || entries.length > 0) {
    return dir;
  }
  return null;
}

function dbHasMessages() {
  if (!fs.existsSync(DB_PATH)) return false;
  try {
    const require = createRequire(import.meta.url);
    const Database = require('better-sqlite3');
    const db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
    const row = db.prepare('SELECT COUNT(*) AS n FROM messages').get();
    db.close();
    return (row?.n ?? 0) > 0;
  } catch {
    return false;
  }
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: 'inherit', ...opts });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} exited with code ${code}`));
    });
  });
}

function syncDashData() {
  const dashData = path.resolve('dash-data');
  const publicData = path.resolve('dashboard/public/data');
  if (!fs.existsSync(publicData)) fs.mkdirSync(publicData, { recursive: true });
  if (!fs.existsSync(dashData)) {
    log('No dash-data yet — run import/generate first.');
    return;
  }
  const files = fs.readdirSync(dashData).filter((f) => f.endsWith('.json'));
  const fileSet = new Set(files);
  for (const stale of fs.readdirSync(publicData)) {
    if (stale.endsWith('.json') && !fileSet.has(stale)) {
      fs.rmSync(path.join(publicData, stale));
    }
  }
  for (const file of files) {
    fs.copyFileSync(path.join(dashData, file), path.join(publicData, file));
  }
  log(`Synced ${files.length} metric files to the dashboard.`);
}

async function main() {
  fs.mkdirSync(DB_DIR, { recursive: true });

  const hasData = dbHasMessages();
  const exportPath = findExport(EXPORT_DIR);

  if (!SKIP_IMPORT && (FORCE_REIMPORT || !hasData)) {
    if (!exportPath) {
      log('');
      log('No Instagram export found.');
      log(`Put your ZIP (or extracted folder) in the host folder mounted at ${EXPORT_DIR}`);
      log('Then restart: docker compose up');
      log('');
      if (!hasData) {
        log('Starting empty dashboard so you can confirm the UI works.');
      }
    } else {
      log(`Importing from ${exportPath}…`);
      await run('node', ['dist/src/cli/index.js', 'import', exportPath]);
      log('Generating analytics…');
      await run('node', ['dist/src/cli/index.js', 'generate']);
    }
  } else if (hasData) {
    log('Database already has messages — skipping import.');
    log('Set FORCE_REIMPORT=1 to re-import from /export.');
    // Refresh metrics if dash-data is empty
    const dashData = path.resolve('dash-data');
    const hasMetrics =
      fs.existsSync(dashData) &&
      fs.readdirSync(dashData).some((f) => f.endsWith('.json'));
    if (!hasMetrics) {
      log('No metrics found — generating…');
      await run('node', ['dist/src/cli/index.js', 'generate']);
    }
  }

  syncDashData();

  log(`Starting dashboard at http://localhost:${PORT}`);
  await run('npm', ['run', 'start', '--prefix', 'dashboard', '--', '-p', String(PORT)], {
    env: { ...process.env, PORT: String(PORT), NODE_ENV: 'production' },
  });
}

main().catch((err) => {
  console.error('[doppel] Fatal:', err);
  process.exit(1);
});
