/**
 * Conversation thread seeding + RAG over message history.
 * Prefers vector retrieval when embeddings exist; falls back to keyword search.
 */

import type { Database as DatabaseType } from 'better-sqlite3';
import { getEmbeddingApiCredentials } from './anthropicSecrets';
import { embedQuery } from './embeddingsClient';
import { bufferToFloat32, cosineSimilarity } from './embeddingsVectors';

export interface StoredMessage {
  id: number;
  conversation_id: string;
  sender: string;
  content: string;
  timestamp_ms: number;
}

export interface ThreadTurn {
  role: 'user' | 'assistant';
  content: string;
  sender: string;
  timestampMs: number;
  messageId: number;
}

export interface MemorySnippet {
  text: string;
  score: number;
  conversationId: string;
  timestampMs: number;
  source?: 'vector' | 'keyword';
}

export type RetrievalMode = 'vector' | 'keyword' | 'none';

const STOP = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
  'is', 'it', 'this', 'that', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'my',
  'your', 'was', 'are', 'be', 'have', 'has', 'had', 'do', 'did', 'so', 'if', 'not',
  'just', 'like', 'yeah', 'ok', 'okay', 'lol', 'haha', 'um', 'uh', 'what', 'when',
  'where', 'who', 'how', 'why', 'can', 'will', 'would', 'could', 'should', 'from',
  'about', 'into', 'than', 'then', 'them', 'their', 'there', 'heres', 'heres',
]);

/** Cap how many stored vectors we score per query (in-memory cosine). */
const VECTOR_CANDIDATE_CAP = 1200;
const MIN_VECTOR_SCORE = 0.28;

export function tokenizeForRetrieval(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9']{3,}/g) ?? []).filter((t) => !STOP.has(t));
}

function escapeLike(token: string): string {
  return token.replace(/[%_\\]/g, '\\$&');
}

function scoreContent(content: string, queryTokens: Set<string>): number {
  if (queryTokens.size === 0) return 0;
  const tokens = tokenizeForRetrieval(content);
  if (tokens.length === 0) return 0;
  let hits = 0;
  const seen = new Set<string>();
  for (const t of tokens) {
    if (queryTokens.has(t) && !seen.has(t)) {
      hits += 1;
      seen.add(t);
    }
  }
  return hits + hits / Math.sqrt(tokens.length);
}

function embeddingsTableExists(db: DatabaseType): boolean {
  const row = db
    .prepare(`SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'message_embeddings'`)
    .get() as { name?: string } | undefined;
  return Boolean(row?.name);
}

function countPersonaEmbeddings(db: DatabaseType, personaSender: string): number {
  if (!embeddingsTableExists(db)) return 0;
  const row = db
    .prepare(
      `
    SELECT COUNT(*) AS n
    FROM (
      SELECT e.message_id
      FROM message_embeddings e
      JOIN messages m ON m.id = e.message_id
      WHERE m.sender = ?
        AND m.is_system = 0
      LIMIT 8
    )
  `
    )
    .get(personaSender) as { n: number };
  return row?.n ?? 0;
}

/** Latest messages in a conversation, oldest → newest. */
export function loadConversationMessages(
  db: DatabaseType,
  conversationId: string,
  limit = 80
): StoredMessage[] {
  const rows = db
    .prepare(
      `
    SELECT id, conversation_id, sender, content, timestamp_ms
    FROM messages
    WHERE conversation_id = ?
      AND is_system = 0
      AND content IS NOT NULL
      AND TRIM(content) != ''
    ORDER BY timestamp_ms DESC
    LIMIT ?
  `
    )
    .all(conversationId, limit) as StoredMessage[];

  return rows.reverse();
}

/**
 * Map stored messages to chat turns relative to the persona:
 * persona → assistant, everyone else → user.
 */
export function messagesToPersonaThread(
  messages: StoredMessage[],
  personaSender: string
): ThreadTurn[] {
  return messages.map((m) => ({
    role: m.sender === personaSender ? 'assistant' : 'user',
    content: m.content.trim(),
    sender: m.sender,
    timestampMs: m.timestamp_ms,
    messageId: m.id,
  }));
}

function formatMemorySnippets(
  db: DatabaseType,
  personaSender: string,
  ranked: Array<StoredMessage & { score: number; source: 'vector' | 'keyword' }>
): MemorySnippet[] {
  const prevStmt = db.prepare(`
    SELECT sender, content
    FROM messages
    WHERE conversation_id = ?
      AND timestamp_ms < ?
      AND is_system = 0
      AND content IS NOT NULL
    ORDER BY timestamp_ms DESC
    LIMIT 1
  `);

  return ranked.map((row) => {
    const prev = prevStmt.get(row.conversation_id, row.timestamp_ms) as
      | { sender: string; content: string }
      | undefined;
    const date = new Date(row.timestamp_ms).toISOString().slice(0, 10);
    const personaLine = `${personaSender}: ${row.content.trim().slice(0, 280)}`;
    const text = prev
      ? `[${date}] ${prev.sender}: ${prev.content.trim().slice(0, 200)} → ${personaLine}`
      : `[${date}] ${personaLine}`;
    return {
      text,
      score: row.score,
      conversationId: row.conversation_id,
      timestampMs: row.timestamp_ms,
      source: row.source,
    };
  });
}

/**
 * Keyword RAG (sync). Used as fallback and to fill gaps beside vector hits.
 */
export function retrievePersonaMemoriesKeyword(
  db: DatabaseType,
  options: {
    personaSender: string;
    query: string;
    conversationId?: string;
    /** Prefer memories from these chats (e.g. threads with you). */
    preferConversationIds?: string[];
    limit?: number;
    excludeMessageIds?: Set<number>;
  }
): MemorySnippet[] {
  const limit = options.limit ?? 6;
  const queryTokens = [...new Set(tokenizeForRetrieval(options.query))];
  const exclude = options.excludeMessageIds ?? new Set<number>();
  const prefer = new Set(options.preferConversationIds ?? []);
  const candidates = new Map<number, StoredMessage & { score: number }>();

  const addCandidate = (row: StoredMessage, bonus = 0) => {
    if (exclude.has(row.id)) return;
    if (row.sender !== options.personaSender) return;
    const base = scoreContent(row.content, new Set(queryTokens));
    if (base <= 0 && bonus <= 0) return;
    let score = base + bonus;
    if (prefer.has(row.conversation_id)) score += 0.4;
    const existing = candidates.get(row.id);
    if (!existing || score > existing.score) {
      candidates.set(row.id, { ...row, score });
    }
  };

  // Prefer the open chat first — usually enough, and avoids global LIKE scans.
  if (options.conversationId) {
    const convRows = db
      .prepare(
        `
      SELECT id, conversation_id, sender, content, timestamp_ms
      FROM messages
      WHERE conversation_id = ?
        AND sender = ?
        AND is_system = 0
        AND content IS NOT NULL
        AND TRIM(content) != ''
      ORDER BY timestamp_ms DESC
      LIMIT 120
    `
      )
      .all(options.conversationId, options.personaSender) as StoredMessage[];

    for (const row of convRows) {
      addCandidate(row, queryTokens.length === 0 ? 0.1 : 0.5);
    }
  }

  // Sample from preferred with-you chats when open chat isn't enough
  const preferIds = [...prefer].slice(0, 8);
  if (preferIds.length > 0 && [...candidates.values()].filter((c) => c.score >= 1.0).length < limit) {
    const ph = preferIds.map(() => '?').join(',');
    const preferRows = db
      .prepare(
        `
      SELECT id, conversation_id, sender, content, timestamp_ms
      FROM messages
      WHERE sender = ?
        AND conversation_id IN (${ph})
        AND is_system = 0
        AND content IS NOT NULL
        AND TRIM(content) != ''
      ORDER BY timestamp_ms DESC
      LIMIT 200
    `
      )
      .all(options.personaSender, ...preferIds) as StoredMessage[];
    for (const row of preferRows) {
      addCandidate(row, queryTokens.length === 0 ? 0.15 : 0.35);
    }
  }

  const strongHits = [...candidates.values()].filter((c) => c.score >= 1.2).length;
  const needGlobal = strongHits < Math.min(3, limit) && queryTokens.length > 0;

  if (needGlobal) {
    const searchTokens = queryTokens.sort((a, b) => b.length - a.length).slice(0, 3);

    for (const token of searchTokens) {
      const like = `%${escapeLike(token)}%`;
      const rows = db
        .prepare(
          `
        SELECT id, conversation_id, sender, content, timestamp_ms
        FROM messages
        WHERE sender = ?
          AND is_system = 0
          AND content IS NOT NULL
          AND content LIKE ? ESCAPE '\\'
        ORDER BY timestamp_ms DESC
        LIMIT 20
      `
        )
        .all(options.personaSender, like) as StoredMessage[];

      for (const row of rows) {
        const sameConv = options.conversationId && row.conversation_id === options.conversationId;
        addCandidate(row, sameConv ? 0.3 : 0);
      }

      if ([...candidates.values()].filter((c) => c.score >= 1.2).length >= limit) break;
    }
  }

  if (candidates.size === 0 && options.conversationId) {
    const recent = db
      .prepare(
        `
      SELECT id, conversation_id, sender, content, timestamp_ms
      FROM messages
      WHERE conversation_id = ?
        AND sender = ?
        AND is_system = 0
        AND content IS NOT NULL
      ORDER BY timestamp_ms DESC
      LIMIT ?
    `
      )
      .all(options.conversationId, options.personaSender, limit) as StoredMessage[];
    for (const row of recent) {
      candidates.set(row.id, { ...row, score: 0.05 });
    }
  }

  const ranked = [...candidates.values()]
    .sort((a, b) => b.score - a.score || b.timestamp_ms - a.timestamp_ms)
    .slice(0, limit)
    .map((row) => ({ ...row, source: 'keyword' as const }));

  return formatMemorySnippets(db, options.personaSender, ranked);
}

type EmbeddingRow = StoredMessage & { embedding: Buffer; model: string };

function loadVectorCandidates(
  db: DatabaseType,
  personaSender: string,
  conversationId: string | undefined,
  model: string
): EmbeddingRow[] {
  const byId = new Map<number, EmbeddingRow>();

  if (conversationId) {
    const convRows = db
      .prepare(
        `
      SELECT m.id, m.conversation_id, m.sender, m.content, m.timestamp_ms,
             e.embedding, e.model
      FROM message_embeddings e
      JOIN messages m ON m.id = e.message_id
      WHERE m.conversation_id = ?
        AND m.sender = ?
        AND e.model = ?
        AND m.is_system = 0
        AND m.content IS NOT NULL
        AND TRIM(m.content) != ''
      ORDER BY m.timestamp_ms DESC
      LIMIT 800
    `
      )
      .all(conversationId, personaSender, model) as EmbeddingRow[];
    for (const row of convRows) byId.set(row.id, row);
  }

  const remaining = Math.max(0, VECTOR_CANDIDATE_CAP - byId.size);
  if (remaining > 0) {
    const globalRows = db
      .prepare(
        `
      SELECT m.id, m.conversation_id, m.sender, m.content, m.timestamp_ms,
             e.embedding, e.model
      FROM message_embeddings e
      JOIN messages m ON m.id = e.message_id
      WHERE m.sender = ?
        AND e.model = ?
        AND m.is_system = 0
        AND m.content IS NOT NULL
        AND TRIM(m.content) != ''
      ORDER BY m.timestamp_ms DESC
      LIMIT ?
    `
      )
      .all(personaSender, model, remaining + byId.size) as EmbeddingRow[];

    for (const row of globalRows) {
      if (byId.size >= VECTOR_CANDIDATE_CAP) break;
      if (!byId.has(row.id)) byId.set(row.id, row);
    }
  }

  return [...byId.values()];
}

async function retrievePersonaMemoriesVector(
  db: DatabaseType,
  options: {
    personaSender: string;
    query: string;
    conversationId?: string;
    limit?: number;
    excludeMessageIds?: Set<number>;
  }
): Promise<MemorySnippet[] | null> {
  const creds = getEmbeddingApiCredentials();
  if (!creds) return null;

  const available = countPersonaEmbeddings(db, options.personaSender);
  if (available < 5) return null;

  const limit = options.limit ?? 12;
  const exclude = options.excludeMessageIds ?? new Set<number>();

  let queryVec: number[];
  try {
    queryVec = await embedQuery(options.query, {
      provider: creds.provider,
      apiKey: creds.apiKey,
      model: creds.model,
    });
  } catch {
    return null;
  }
  if (!queryVec.length) return null;

  const candidates = loadVectorCandidates(
    db,
    options.personaSender,
    options.conversationId,
    creds.model
  );
  if (candidates.length === 0) return null;

  const scored: Array<StoredMessage & { score: number; source: 'vector' }> = [];
  for (const row of candidates) {
    if (exclude.has(row.id)) continue;
    const emb = bufferToFloat32(
      Buffer.isBuffer(row.embedding) ? row.embedding : Buffer.from(row.embedding)
    );
    let sim = cosineSimilarity(queryVec, emb);
    if (options.conversationId && row.conversation_id === options.conversationId) {
      sim += 0.04;
    }
    if (sim < MIN_VECTOR_SCORE) continue;
    scored.push({
      id: row.id,
      conversation_id: row.conversation_id,
      sender: row.sender,
      content: row.content,
      timestamp_ms: row.timestamp_ms,
      score: sim,
      source: 'vector',
    });
  }

  if (scored.length === 0) return null;

  scored.sort((a, b) => b.score - a.score || b.timestamp_ms - a.timestamp_ms);
  return formatMemorySnippets(db, options.personaSender, scored.slice(0, limit));
}

/**
 * Retrieve relevant past messages for the persona.
 * Uses vector RAG when embeddings + credentials are available; otherwise keyword.
 * Merges a few keyword hits when vector mode is active to catch rare exact tokens.
 */
export async function retrievePersonaMemories(
  db: DatabaseType,
  options: {
    personaSender: string;
    query: string;
    conversationId?: string;
    preferConversationIds?: string[];
    limit?: number;
    excludeMessageIds?: Set<number>;
  }
): Promise<{ memories: MemorySnippet[]; mode: RetrievalMode }> {
  const limit = options.limit ?? 6;

  const vector = await retrievePersonaMemoriesVector(db, { ...options, limit });
  if (vector && vector.length > 0) {
    if (vector.length >= limit) {
      return { memories: vector.slice(0, limit), mode: 'vector' };
    }
    const vectorTexts = new Set(vector.map((m) => m.text));
    const keyword = retrievePersonaMemoriesKeyword(db, {
      ...options,
      limit: limit - vector.length,
    });
    const merged = [...vector];
    for (const k of keyword) {
      if (merged.length >= limit) break;
      if (vectorTexts.has(k.text)) continue;
      merged.push(k);
    }
    return { memories: merged.slice(0, limit), mode: 'vector' };
  }

  const keyword = retrievePersonaMemoriesKeyword(db, options);
  return {
    memories: keyword,
    mode: keyword.length > 0 ? 'keyword' : 'none',
  };
}

/** Sync wrapper kept for tests that only need keyword behavior. */
export function retrievePersonaMemoriesSync(
  db: DatabaseType,
  options: {
    personaSender: string;
    query: string;
    conversationId?: string;
    limit?: number;
    excludeMessageIds?: Set<number>;
  }
): MemorySnippet[] {
  return retrievePersonaMemoriesKeyword(db, options);
}
