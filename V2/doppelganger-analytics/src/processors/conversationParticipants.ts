/**
 * Per-conversation participant counts, computed once per connection.
 *
 * Several persona processors need "how many people are in this chat?" while
 * scanning a sender's messages. Asking SQLite per row turns into a correlated
 * subquery that rescans the whole conversation for every message, which is
 * quadratic on real exports. One grouped pass up front, cached per database
 * handle, keeps every caller to a map lookup.
 */

import type { Database as DatabaseType } from 'better-sqlite3';

const CACHE = new WeakMap<DatabaseType, Map<string, number>>();

/** conversation_id → distinct non-system sender count. */
export function conversationParticipantCounts(db: DatabaseType): Map<string, number> {
  const cached = CACHE.get(db);
  if (cached) return cached;

  const rows = db
    .prepare(
      `
    SELECT conversation_id, COUNT(DISTINCT sender) AS n
    FROM messages
    WHERE is_system = 0
    GROUP BY conversation_id
  `
    )
    .all() as Array<{ conversation_id: string; n: number }>;

  const counts = new Map(rows.map((r) => [r.conversation_id, r.n]));
  CACHE.set(db, counts);
  return counts;
}

/** Participant count for one conversation (defaults to 2 when unseen). */
export function participantCount(db: DatabaseType, conversationId: string): number {
  return conversationParticipantCounts(db).get(conversationId) ?? 2;
}
