#!/usr/bin/env node
/**
 * One-click Docker entrypoint.
 *
 * 1. Looks for messaging export(s) under /export (ZIP(s) or extracted folder)
 * 2. Imports + generates metrics if the DB is empty or FORCE_REIMPORT is set
 * 3. Syncs dash-data → dashboard/public/data
 * 4. Starts the Next.js production server on :3000
 */
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { createRequire } from 'module';
import unzipper from 'unzipper';

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

/** All ZIPs under dir, or the dir itself if it looks like an extracted export. */
function findExports(dir) {
  if (!fs.existsSync(dir)) return [];

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const zips = entries
    .filter((e) => e.isFile() && e.name.toLowerCase().endsWith('.zip'))
    .map((e) => path.join(dir, e.name))
    .sort();

  // If there are ZIP exports, keep them but also look for other extracted export
  // folders (like iMessage TXT exports) alongside them.
  const exportPaths = [...zips];

  const iMessageTimestampLineRe = new RegExp(
    '^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{1,2},\\s+\\d{4}\\s+\\d{1,2}:\\d{2}:\\d{2}\\s+(?:AM|PM)'
  );

  function dirLooksLikeIMessageExporterTxt(dirPath) {
    let dirEntries;
    try {
      dirEntries = fs.readdirSync(dirPath, { withFileTypes: true });
    } catch {
      return false;
    }

    const txtFiles = dirEntries
      .filter((e) => e.isFile() && e.name.toLowerCase().endsWith('.txt'))
      .slice(0, 8);
    if (txtFiles.length === 0) return false;

    // Sample a few files and look for timestamp lines.
    let hits = 0;
    for (const f of txtFiles) {
      try {
        const full = path.join(dirPath, f.name);
        const sample = fs.readFileSync(full, 'utf8').slice(0, 8000);
        const lines = sample.split(/\r?\n/);
        for (const line of lines.slice(0, 60)) {
          if (iMessageTimestampLineRe.test(line.trim())) {
            hits++;
            if (hits >= 2) return true;
          }
        }
      } catch {
        // ignore unreadable files
      }
    }
    return false;
  }

  function dirLooksLikeMessages(dirPath, dirEntries) {
    if (!dirEntries) return false;
    const localHasMessages = dirEntries.some(
      (e) =>
        (e.isDirectory() &&
          (e.name === 'messages' || e.name === 'inbox' || e.name === 'your_instagram_activity')) ||
        (e.isFile() && e.name.startsWith('message_') && e.name.endsWith('.json')) ||
        (e.isFile() && (e.name === '_chat.txt' || e.name === 'chat.db'))
    );
    return localHasMessages || dirLooksLikeIMessageExporterTxt(dirPath);
  }

  const rootHasMessages = dirLooksLikeMessages(dir, entries);
  if (rootHasMessages) exportPaths.push(dir);

  // Also check extracted export folders directly under /export (common when
  // you mount a `data/` folder containing both ZIPs and extracted txt folders).
  for (const e of entries) {
    if (!e.isDirectory() || e.name.startsWith('.')) continue;
    const subPath = path.join(dir, e.name);
    const subEntries = (() => {
      try {
        return fs.readdirSync(subPath, { withFileTypes: true });
      } catch {
        return null;
      }
    })();
    if (dirLooksLikeMessages(subPath, subEntries)) exportPaths.push(subPath);
  }

  if (exportPaths.length > 0) return Array.from(new Set(exportPaths)).sort();

  // Nested single folder (common after unzip)
  const dirs = entries.filter((e) => e.isDirectory() && !e.name.startsWith('.'));
  if (dirs.length === 1) {
    return findExports(path.join(dir, dirs[0].name));
  }

  return [];
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

async function detectExportPlatform(exportPath) {
  const lowerPath = exportPath.toLowerCase();
  const iMessageTimestampLineRe = new RegExp(
    '^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+\\d{1,2},\\s+\\d{4}\\s+\\d{1,2}:\\d{2}:\\d{2}\\s+(?:AM|PM)'
  );
  const hasMetaMarkers = (entries) =>
    entries.some(
      (entry) =>
        entry.includes('/messages/inbox/') ||
        entry.includes('/your_instagram_activity/') ||
        entry.includes('/your_facebook_activity/') ||
        /\/message_\d+\.json$/.test(entry)
    );

  if (lowerPath.endsWith('.zip')) {
    try {
      const zip = await unzipper.Open.file(exportPath);
      const entries = zip.files.map((f) => f.path.toLowerCase());
      if (entries.some((entry) => entry.endsWith('/chat.db') || entry === 'chat.db')) {
        return 'iMessage (chat.db ZIP)';
      }
      if (entries.some((entry) => entry.endsWith('/_chat.txt') || entry === '_chat.txt')) {
        return 'WhatsApp (_chat.txt ZIP)';
      }
      if (hasMetaMarkers(entries)) return 'Instagram / Messenger (Meta JSON ZIP)';
      return 'Unknown ZIP format';
    } catch {
      return 'Unknown ZIP format (unreadable ZIP)';
    }
  }

  try {
    const entries = fs.readdirSync(exportPath, { withFileTypes: true });
    if (entries.some((e) => e.isFile() && e.name === 'chat.db')) return 'iMessage (chat.db folder)';
    if (entries.some((e) => e.isFile() && e.name === '_chat.txt')) return 'WhatsApp (_chat.txt folder)';

    const txtFiles = entries.filter((e) => e.isFile() && e.name.toLowerCase().endsWith('.txt')).slice(0, 6);
    let iMessageTxtHits = 0;
    for (const txt of txtFiles) {
      try {
        const sample = fs.readFileSync(path.join(exportPath, txt.name), 'utf8').slice(0, 8000);
        const lines = sample.split(/\r?\n/);
        if (lines.some((line) => iMessageTimestampLineRe.test(line.trim()))) {
          iMessageTxtHits++;
          if (iMessageTxtHits >= 2) return 'iMessage (imessage-exporter TXT)';
        }
      } catch {
        // ignore unreadable files
      }
    }

    const walk = (dirPath, depth = 0) => {
      if (depth > 5) return false;
      let subEntries;
      try {
        subEntries = fs.readdirSync(dirPath, { withFileTypes: true });
      } catch {
        return false;
      }
      for (const sub of subEntries) {
        const subPath = path.join(dirPath, sub.name);
        if (sub.isDirectory()) {
          const lowered = subPath.toLowerCase();
          if (
            lowered.includes('/messages/inbox') ||
            lowered.includes('/your_instagram_activity') ||
            lowered.includes('/your_facebook_activity')
          ) {
            return true;
          }
          if (walk(subPath, depth + 1)) return true;
        } else if (sub.isFile() && /^message_\d+\.json$/i.test(sub.name)) {
          return true;
        }
      }
      return false;
    };

    if (walk(exportPath)) return 'Instagram / Messenger (Meta JSON folder)';
  } catch {
    return 'Unknown folder format (unreadable)';
  }

  return 'Unknown folder format';
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

async function importAndGenerate(exportPaths) {
  for (const [idx, exportPath] of exportPaths.entries()) {
    const detectedPlatform = await detectExportPlatform(exportPath);
    log(`[${idx + 1}/${exportPaths.length}] Importing from ${exportPath}`);
    log(`    Detected: ${detectedPlatform}`);
    // Skip auto-generate per ZIP — one generate after all platforms land.
    await run('node', ['dist/src/cli/index.js', 'import', '--no-generate', exportPath]);
  }
  log('Generating analytics…');
  await run('node', ['dist/src/cli/index.js', 'generate']);
}

async function main() {
  fs.mkdirSync(DB_DIR, { recursive: true });

  const hasData = dbHasMessages();
  const exportPaths = findExports(EXPORT_DIR);

  if (!SKIP_IMPORT && (FORCE_REIMPORT || !hasData)) {
    if (exportPaths.length === 0) {
      log('');
      log('No messaging export found.');
      log(`Put ZIP(s) or an extracted folder in the host folder mounted at ${EXPORT_DIR}`);
      log('(Instagram / Messenger JSON, WhatsApp chat export, or iMessage chat.db)');
      log('Then restart: docker compose up');
      log('');
      if (!hasData) {
        log('Starting empty dashboard so you can confirm the UI works.');
      }
    } else {
      if (exportPaths.length > 1) {
        log(`Found ${exportPaths.length} exports — importing all (platforms merge).`);
      }
      await importAndGenerate(exportPaths);
    }
  } else if (hasData) {
    log('Database already has messages — skipping import.');
    log('Set FORCE_REIMPORT=1 to re-import from /export.');
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
