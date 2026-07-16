import { Database } from 'better-sqlite3';
import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';

/**
 * Canonical definition of a "response", used by every response-time metric.
 *
 * A response is a message whose immediately-preceding message in the same
 * conversation came from a different sender, with an inter-message gap in
 * [MIN, MAX]. Gaps under 1s are effectively simultaneous; gaps over 24h mark a
 * new session rather than a reply. This is the single source of truth: the
 * response_times table is rebuilt from it and all consumers read that table.
 */
export const RESPONSE_TIME_MIN_MS = 1000;
export const RESPONSE_TIME_MAX_MS = 24 * 60 * 60 * 1000;

export function isResponseLatency(latencyMs: number): boolean {
  return latencyMs >= RESPONSE_TIME_MIN_MS && latencyMs <= RESPONSE_TIME_MAX_MS;
}

interface MessageRow {
  id: number;
  conversation_id: string;
  sender: string;
  timestamp_ms: number;
}

/**
 * Rebuilds the response_times table from scratch. Returns the number of
 * responses recorded. Pure enough to unit-test against a fixture DB.
 */
export function rebuildResponseTimes(db: Database): number {
  const messages = db.prepare(`
    SELECT id, conversation_id, sender, timestamp_ms
    FROM messages
    ORDER BY conversation_id, timestamp_ms
  `).all() as MessageRow[];

  const rows: Array<{
    conversation_id: string;
    from_message_id: number;
    to_message_id: number;
    latency_ms: number;
  }> = [];

  for (let i = 0; i < messages.length - 1; i++) {
    const current = messages[i];
    const next = messages[i + 1];

    // Ordered by conversation then time, so adjacency within a conversation is
    // chronological; skip the seam between conversations.
    if (current.conversation_id !== next.conversation_id) continue;
    if (current.sender === next.sender) continue;

    const latency = next.timestamp_ms - current.timestamp_ms;
    if (!isResponseLatency(latency)) continue;

    rows.push({
      conversation_id: current.conversation_id,
      from_message_id: current.id,
      to_message_id: next.id,
      latency_ms: latency
    });
  }

  const insert = db.prepare(`
    INSERT INTO response_times (conversation_id, from_message_id, to_message_id, latency_ms)
    VALUES (@conversation_id, @from_message_id, @to_message_id, @latency_ms)
  `);
  const insertMany = db.transaction((batch: typeof rows) => {
    for (const row of batch) insert.run(row);
  });

  // Older DBs declared a foreign key from response_times to the (unpopulated)
  // conversations table; disable enforcement for this derived-data rebuild.
  const fkWasOn = db.pragma('foreign_keys', { simple: true }) === 1;
  if (fkWasOn) db.pragma('foreign_keys = OFF');
  try {
    db.prepare('DELETE FROM response_times').run();
    insertMany(rows);
  } finally {
    if (fkWasOn) db.pragma('foreign_keys = ON');
  }

  return rows.length;
}

export async function computeResponseTimes(): Promise<void> {
  progressReporter.start('Computing response times...');
  const db = await getDb();
  try {
    const count = rebuildResponseTimes(db);
    progressReporter.success(`Response times computed (${count.toLocaleString()} responses)`);
  } catch (error) {
    progressReporter.error('Error computing response times');
    console.error(error);
    throw error;
  } finally {
    await closeDb(db);
  }
}
