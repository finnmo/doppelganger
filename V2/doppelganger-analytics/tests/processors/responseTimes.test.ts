import { createTestDatabase, TestDatabase } from '../utils/testDatabase.js';
import {
  rebuildResponseTimes,
  isResponseLatency,
  RESPONSE_TIME_MIN_MS,
  RESPONSE_TIME_MAX_MS
} from '../../src/processors/responseTimes.js';

const MIN = 60_000; // 1 minute in ms
const base = 1_700_000_000_000;

describe('isResponseLatency', () => {
  test('accepts values inside the window and rejects outside', () => {
    expect(isResponseLatency(RESPONSE_TIME_MIN_MS)).toBe(true);
    expect(isResponseLatency(RESPONSE_TIME_MAX_MS)).toBe(true);
    expect(isResponseLatency(RESPONSE_TIME_MIN_MS - 1)).toBe(false);
    expect(isResponseLatency(RESPONSE_TIME_MAX_MS + 1)).toBe(false);
    expect(isResponseLatency(-5)).toBe(false);
  });
});

describe('rebuildResponseTimes', () => {
  let testDb: TestDatabase;

  afterEach(() => {
    if (testDb) testDb.cleanup();
  });

  test('records only cross-sender replies within the window', () => {
    testDb = createTestDatabase('responseTimes');
    const db = testDb.getDatabase();

    testDb.insertTestMessages([
      // A speaks, B replies after 5 min -> one response
      { id: 1, conversation_id: 'c1', sender: 'A', timestamp_ms: base, content: 'hi' },
      { id: 2, conversation_id: 'c1', sender: 'A', timestamp_ms: base + 1 * MIN, content: 'you there?' }, // same sender: skipped
      { id: 3, conversation_id: 'c1', sender: 'B', timestamp_ms: base + 6 * MIN, content: 'hey' },       // reply to #2 (5 min)
      // B speaks, A replies after 48h -> outside window, skipped
      { id: 4, conversation_id: 'c1', sender: 'A', timestamp_ms: base + 6 * MIN + RESPONSE_TIME_MAX_MS + 1000, content: 'late' },
      // Different conversation, cross-sender within window -> one response
      { id: 5, conversation_id: 'c2', sender: 'X', timestamp_ms: base, content: 'yo' },
      { id: 6, conversation_id: 'c2', sender: 'Y', timestamp_ms: base + 2 * MIN, content: 'sup' }
    ]);

    const count = rebuildResponseTimes(db);
    expect(count).toBe(2);

    const rows = db.prepare('SELECT conversation_id, from_message_id, to_message_id, latency_ms FROM response_times ORDER BY id').all();
    expect(rows).toEqual([
      { conversation_id: 'c1', from_message_id: 2, to_message_id: 3, latency_ms: 5 * MIN },
      { conversation_id: 'c2', from_message_id: 5, to_message_id: 6, latency_ms: 2 * MIN }
    ]);
  });

  test('is deterministic and idempotent across runs', () => {
    testDb = createTestDatabase('responseTimesDeterminism');
    const db = testDb.getDatabase();
    testDb.insertTestMessages([
      { id: 1, conversation_id: 'c1', sender: 'A', timestamp_ms: base, content: 'a' },
      { id: 2, conversation_id: 'c1', sender: 'B', timestamp_ms: base + 3 * MIN, content: 'b' },
      { id: 3, conversation_id: 'c1', sender: 'A', timestamp_ms: base + 4 * MIN, content: 'c' }
    ]);

    // Compare the meaningful columns, not the AUTOINCREMENT surrogate id.
    const query = 'SELECT conversation_id, from_message_id, to_message_id, latency_ms FROM response_times ORDER BY from_message_id';
    const first = rebuildResponseTimes(db);
    const snapshot1 = db.prepare(query).all();
    const second = rebuildResponseTimes(db);
    const snapshot2 = db.prepare(query).all();

    expect(first).toBe(second);
    expect(snapshot1).toEqual(snapshot2);
  });
});
