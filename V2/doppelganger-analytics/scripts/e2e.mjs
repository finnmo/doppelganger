#!/usr/bin/env node
/**
 * End-to-end smoke test:
 *  1) Build a tiny Instagram-style fixture
 *  2) Import → generate (local CLI)
 *  3) Optionally build the Docker image, run a one-shot container, hit HTTP
 *
 * Usage:
 *   node scripts/e2e.mjs           # local pipeline only
 *   node scripts/e2e.mjs --docker  # local + docker build/run smoke
 */
import fs from 'fs';
import os from 'os';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const WITH_DOCKER = process.argv.includes('--docker');

function log(step, msg) {
  console.log(`\n▶ [${step}] ${msg}`);
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {
      stdio: 'inherit',
      cwd: opts.cwd || ROOT,
      env: { ...process.env, ...opts.env },
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} exited ${code}`));
    });
  });
}

function writeFixture(exportDir) {
  const convDir = path.join(exportDir, 'messages', 'inbox', 'e2e_alice_1');
  fs.mkdirSync(convDir, { recursive: true });
  const now = Date.now();
  // Enough messages for persona profiles (min thresholds in processors)
  const messages = [];
  for (let i = 0; i < 40; i++) {
    messages.push({
      sender_name: i % 2 === 0 ? 'Alice E2E' : 'Bob E2E',
      timestamp_ms: now - (40 - i) * 120_000,
      content:
        i % 2 === 0
          ? `hey bob message ${i} — coffee later?`
          : `yeah alice reply ${i} sounds good`,
    });
  }
  fs.writeFileSync(
    path.join(convDir, 'message_1.json'),
    JSON.stringify({
      participants: [{ name: 'Alice E2E' }, { name: 'Bob E2E' }],
      messages,
    })
  );
}

async function waitForUrl(url, { timeoutMs = 180_000, intervalMs = 2500 } = {}) {
  const start = Date.now();
  let lastErr = '';
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return res.status;
      lastErr = `HTTP ${res.status}`;
    } catch (err) {
      lastErr = err instanceof Error ? err.message : String(err);
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`Timeout waiting for ${url}: ${lastErr}`);
}

async function runLocalE2E() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'doppel-e2e-'));
  const exportDir = path.join(tmp, 'export');
  const dbPath = path.join(tmp, 'analytics.db');
  const dashDir = path.join(tmp, 'dash-data');
  fs.mkdirSync(exportDir, { recursive: true });
  fs.mkdirSync(dashDir, { recursive: true });
  writeFixture(exportDir);

  log('local', `fixture → ${exportDir}`);
  log('local', 'build CLI');
  await run('npm', ['run', 'build']);

  log('local', 'import + generate');
  await run('node', ['dist/src/cli/index.js', 'import', exportDir], {
    env: {
      DOPPELGANGER_DB_PATH: dbPath,
      DOPPELGANGER_DASH_DIR: dashDir,
    },
  });

  const profilesPath = path.join(dashDir, 'personaProfiles.json');
  if (!fs.existsSync(profilesPath)) {
    throw new Error(`Missing ${profilesPath} after generate`);
  }
  const profiles = JSON.parse(fs.readFileSync(profilesPath, 'utf8'));
  if (!Array.isArray(profiles.profiles) || profiles.profiles.length === 0) {
    throw new Error('personaProfiles.json has no profiles');
  }

  const Database = (await import('better-sqlite3')).default;
  const db = new Database(dbPath, { readonly: true });
  const count = db.prepare('SELECT COUNT(*) AS n FROM messages').get().n;
  db.close();
  if (count < 40) throw new Error(`Expected ≥40 messages, got ${count}`);

  log('local', `OK — ${count} messages, ${profiles.profiles.length} personas`);
  fs.rmSync(tmp, { recursive: true, force: true });
  return { messageCount: count, personaCount: profiles.profiles.length };
}

async function runDockerE2E() {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'doppel-e2e-docker-'));
  const exportDir = path.join(tmp, 'export');
  writeFixture(exportDir);

  const image = 'doppelganger-analytics-e2e:local';
  const port = '3011';
  const name = 'doppel-e2e-run';

  log('docker', 'build image (BuildKit)');
  await run('docker', ['build', '-t', image, '.'], {
    env: { DOCKER_BUILDKIT: '1' },
  });

  // Clean any previous run
  await run('docker', ['rm', '-f', name]).catch(() => {});

  log('docker', 'run container');
  const child = spawn(
    'docker',
    [
      'run',
      '--rm',
      '--name',
      name,
      '-p',
      `${port}:3000`,
      '-e',
      'FORCE_REIMPORT=1',
      '-v',
      `${exportDir}:/export:ro`,
      image,
    ],
    { cwd: ROOT, stdio: 'inherit' }
  );

  try {
    log('docker', `wait for http://127.0.0.1:${port}`);
    await waitForUrl(`http://127.0.0.1:${port}/`);

    const home = await fetch(`http://127.0.0.1:${port}/`);
    if (!home.ok) throw new Error(`Dashboard home HTTP ${home.status}`);

    const profilesRes = await fetch(`http://127.0.0.1:${port}/data/personaProfiles.json`);
    if (!profilesRes.ok) {
      throw new Error(`personaProfiles.json HTTP ${profilesRes.status}`);
    }
    const profiles = await profilesRes.json();
    if (!Array.isArray(profiles.profiles) || profiles.profiles.length === 0) {
      throw new Error('Docker dashboard served empty personaProfiles');
    }

    log('docker', `OK — dashboard up, ${profiles.profiles.length} personas served`);
  } finally {
    child.kill('SIGTERM');
    await run('docker', ['rm', '-f', name]).catch(() => {});
    fs.rmSync(tmp, { recursive: true, force: true });
  }
}

async function main() {
  console.log('Doppelgänger e2e');
  const local = await runLocalE2E();
  if (WITH_DOCKER) {
    await runDockerE2E();
  } else {
    log('skip', 'Pass --docker to also build/run image smoke test');
  }
  console.log('\n✅ e2e passed', local);
}

main().catch((err) => {
  console.error('\n❌ e2e failed:', err);
  process.exit(1);
});
