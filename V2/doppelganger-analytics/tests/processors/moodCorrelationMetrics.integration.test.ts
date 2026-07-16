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

// Daily average compounds per sender. B is A scaled by 0.5, so the Pearson
// correlation between them is exactly 1 (strong).
const DAILY_A = [0.8, 0.6, 0.4, 0.2, 0.0, -0.2];
const DAILY_B = DAILY_A.map(v => v * 0.5);

function seed(db: Database.Database) {
  const insertMsg = db.prepare(`
    INSERT INTO messages (id, conversation_id, sender, timestamp_ms, content, has_photos, has_videos, has_audio, has_share, share_link)
    VALUES (@id, @c, @s, @t, @content, 0, 0, 0, 0, NULL)
  `);
  const insertSentiment = db.prepare(`
    INSERT INTO sentiment (message_id, compound, positive, negative, neutral)
    VALUES (@id, @compound, 0.5, 0.2, 0.3)
  `);

  let nextId = 1;
  const tx = db.transaction(() => {
    // A and B: 3 messages/day (the processor's HAVING clause requires >= 3)
    // across 6 shared days (>= 5 shared days required for a correlation).
    for (let day = 0; day < DAILY_A.length; day++) {
      // Local noon avoids date-boundary ambiguity in localtime bucketing.
      const t = new Date(2024, 0, day + 1, 12, 0, 0).getTime();
      for (const [sender, compound] of [['A', DAILY_A[day]], ['B', DAILY_B[day]]] as const) {
        for (let k = 0; k < 3; k++) {
          const id = nextId++;
          insertMsg.run({ id, c: 'c1', s: sender, t: t + k * 1000, content: 'msg' });
          insertSentiment.run({ id, compound });
        }
      }
    }

    // C: only 2 days of data — below both the 5-shared-day correlation
    // threshold and the 5-datapoint mood-pattern threshold.
    for (let day = 0; day < 2; day++) {
      const t = new Date(2024, 0, day + 1, 12, 0, 0).getTime();
      for (let k = 0; k < 3; k++) {
        const id = nextId++;
        insertMsg.run({ id, c: 'c1', s: 'C', t: t + k * 1000, content: 'msg' });
        insertSentiment.run({ id, compound: 0.5 });
      }
    }
  });
  tx();
}

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mood-correlation-test-'));
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

describe('computeMoodCorrelationMetrics (integration)', () => {
  test('computes correlations, mood patterns, and summary from the fixture DB', async () => {
    const { computeMoodCorrelationMetrics } = await import('../../src/processors/moodCorrelationMetrics.js');
    await computeMoodCorrelationMetrics();

    const output = JSON.parse(fs.readFileSync(path.join(dashDir, 'moodCorrelationMetrics.json'), 'utf-8'));

    // A, B, and C all have qualifying daily records.
    expect(output.summary.totalParticipants).toBe(3);
    expect(output.summary.dateRange).toEqual({ start: '2024-01-01', end: '2024-01-06' });

    // Only the A↔B pair shares >= 5 days; B is a linear function of A, so
    // the correlation is exactly 1 and classified as strong.
    expect(output.correlationMatrix).toHaveLength(1);
    const pair = output.correlationMatrix[0];
    expect([pair.sender1, pair.sender2].sort()).toEqual(['A', 'B']);
    expect(pair.correlation).toBeCloseTo(1, 6);
    expect(pair.strength).toBe('strong');
    expect(pair.sharedDays).toBe(6);
    expect(output.summary.strongCorrelations).toBe(1);

    // Mood patterns exist only for senders with >= 5 daily datapoints.
    const senders = output.moodPatterns.map((p: { sender: string }) => p.sender).sort();
    expect(senders).toEqual(['A', 'B']);

    const patternA = output.moodPatterns.find((p: { sender: string }) => p.sender === 'A');
    // mean(0.8, 0.6, 0.4, 0.2, 0.0, -0.2) = 0.3
    expect(patternA.averageMood).toBeCloseTo(0.3, 3);
    // First-half avg 0.6 vs second-half avg 0.0 → declining.
    expect(patternA.moodTrend).toBe('declining');
    expect(patternA.dominantEmotion).toBe('positive');
    // Four consecutive days > 0.1, one day < -0.1.
    expect(patternA.positiveStreak).toBe(4);
    expect(patternA.negativeStreak).toBe(1);

    // One time-series entry per day.
    expect(output.timeSeriesData).toHaveLength(6);
    expect(output.timeSeriesData[0].participants.A).toBeCloseTo(0.8, 6);
    expect(output.timeSeriesData[0].participants.B).toBeCloseTo(0.4, 6);
  });

  test('produces identical output on a second run (deterministic)', async () => {
    const { computeMoodCorrelationMetrics } = await import('../../src/processors/moodCorrelationMetrics.js');
    await computeMoodCorrelationMetrics();
    const first = fs.readFileSync(path.join(dashDir, 'moodCorrelationMetrics.json'), 'utf-8');
    await computeMoodCorrelationMetrics();
    const second = fs.readFileSync(path.join(dashDir, 'moodCorrelationMetrics.json'), 'utf-8');
    expect(first).toBe(second);
  });
});
