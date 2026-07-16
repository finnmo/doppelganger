import fs from 'fs';
import os from 'os';
import path from 'path';
import Database from 'better-sqlite3';
import { migrate } from '../../src/db/schema.js';

// Isolated fixture DB + output dir, wired via env before importing the
// processor (which reads them through getDb / writeDashData).
let tmpDir: string;
let dbPath: string;
let dashDir: string;

const MIN = 60 * 1000;

function seed(db: Database.Database) {
  const insertMsg = db.prepare(`
    INSERT INTO messages (id, conversation_id, sender, timestamp_ms, content, has_photos, has_videos, has_audio, has_share, share_link)
    VALUES (@id, @c, @s, @t, @content, 0, 0, 0, 0, NULL)
  `);

  let nextId = 1;
  const add = (c: string, s: string, t: number) =>
    insertMsg.run({ id: nextId++, c, s, t, content: 'msg' });

  const tx = db.transaction(() => {
    // balanced: A and B alternate strictly (3 turns each, ratio 1).
    let t = new Date(2024, 0, 1, 12, 0, 0).getTime();
    for (let i = 0; i < 6; i++) {
      add('balanced', i % 2 === 0 ? 'A' : 'B', t);
      t += 10 * MIN;
    }

    // dominant: A sends 8 of 10 messages (80% message share, > 0.7).
    // Consecutive A messages merge into single turns, so the turn sequence
    // is A(1), B(1), A(1), C(1), A(6).
    t = new Date(2024, 0, 2, 12, 0, 0).getTime();
    const dominantSenders = ['A', 'B', 'A', 'C', 'A', 'A', 'A', 'A', 'A', 'A'];
    for (const s of dominantSenders) {
      add('dominant', s, t);
      t += 10 * MIN;
    }

    // tiny: below the 3-message minimum, must be skipped.
    t = new Date(2024, 0, 3, 12, 0, 0).getTime();
    add('tiny', 'A', t);
    add('tiny', 'B', t + MIN);

    // solo: single participant, must be skipped.
    t = new Date(2024, 0, 4, 12, 0, 0).getTime();
    add('solo', 'A', t);
    add('solo', 'A', t + MIN);
    add('solo', 'A', t + 2 * MIN);
  });
  tx();
}

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'turn-taking-test-'));
  dbPath = path.join(tmpDir, 'fixture.db');
  dashDir = path.join(tmpDir, 'dash-data');
  const db = new Database(dbPath);
  migrate(db);
  seed(db);
  db.close();

  process.env.DOPPELGANGER_DB_PATH = dbPath;
  process.env.DOPPELGANGER_DASH_DIR = dashDir;
});

afterAll(async () => {
  const { closeAllConnections } = await import('../../src/db/client.js');
  await closeAllConnections();
  delete process.env.DOPPELGANGER_DB_PATH;
  delete process.env.DOPPELGANGER_DASH_DIR;
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe('computeTurnTakingAnalysisMetrics (integration)', () => {
  test('classifies balanced and dominant conversations and skips tiny/solo ones', async () => {
    const { computeTurnTakingAnalysisMetrics } = await import('../../src/processors/turnTakingAnalysisMetrics.js');
    await computeTurnTakingAnalysisMetrics();

    const output = JSON.parse(fs.readFileSync(path.join(dashDir, 'turnTakingAnalysis.json'), 'utf-8'));

    // tiny (< 3 messages) and solo (1 participant) are excluded.
    expect(output.summary.total_conversations).toBe(2);
    const ids = output.conversation_patterns.map((p: { conversation_id: string }) => p.conversation_id).sort();
    expect(ids).toEqual(['balanced', 'dominant']);

    const balanced = output.conversation_patterns.find(
      (p: { conversation_id: string }) => p.conversation_id === 'balanced'
    );
    expect(balanced.pattern.pattern_type).toBe('balanced');
    expect(balanced.pattern.turn_ratio).toBe(1);
    expect(balanced.participants).toHaveLength(2);
    const balancedA = balanced.participants.find((p: { participant: string }) => p.participant === 'A');
    expect(balancedA.turn_count).toBe(3);
    expect(balancedA.turn_percentage).toBe(50);
    // Strict alternation with 10-minute gaps → 10-minute average response.
    expect(balancedA.avg_response_time).toBe(10 * MIN);

    const dominant = output.conversation_patterns.find(
      (p: { conversation_id: string }) => p.conversation_id === 'dominant'
    );
    // A holds 4 of 5 turns (consecutive A messages merge into one turn),
    // which is > 70% of turns.
    expect(dominant.pattern.pattern_type).toBe('dominant');
    const dominantA = dominant.participants.find((p: { participant: string }) => p.participant === 'A');
    expect(dominantA.turn_count).toBe(3);

    expect(output.summary.balanced_conversations).toBe(1);
    expect(output.summary.dominant_speaker_conversations).toBe(1);

    // Participant stats cover every participant of the analyzed conversations.
    const statNames = output.participant_stats.map((p: { participant: string }) => p.participant).sort();
    expect(statNames).toEqual(['A', 'B', 'C']);
    const statsA = output.participant_stats.find((p: { participant: string }) => p.participant === 'A');
    expect(statsA.conversations).toBe(2);
  });

  test('produces identical output on a second run (deterministic)', async () => {
    const { computeTurnTakingAnalysisMetrics } = await import('../../src/processors/turnTakingAnalysisMetrics.js');
    await computeTurnTakingAnalysisMetrics();
    const first = fs.readFileSync(path.join(dashDir, 'turnTakingAnalysis.json'), 'utf-8');
    await computeTurnTakingAnalysisMetrics();
    const second = fs.readFileSync(path.join(dashDir, 'turnTakingAnalysis.json'), 'utf-8');
    expect(first).toBe(second);
  });
});
