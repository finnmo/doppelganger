/**
 * Precompute how a persona talks specifically *with you* (the account holder):
 * address forms, openers/closers, teasing, question-back rate, length vs global,
 * shared chat IDs for few-shot/RAG bias.
 */

import type { Database as DatabaseType } from 'better-sqlite3';

const PLACE_HINTS = new Set([
  'melbourne', 'sydney', 'brisbane', 'perth', 'adelaide', 'canberra', 'hobart',
  'london', 'paris', 'tokyo', 'bali', 'thailand', 'japan', 'australia', 'nz',
  'new zealand', 'airport', 'cafe', 'coffee', 'uni', 'campus', 'office', 'gym',
  'beach', 'home', 'work', 'city', 'suburb', 'station', 'hotel',
]);

const ENDEARMENTS = [
  'babe', 'baby', 'love', 'hun', 'honey', 'bro', 'dude', 'mate', 'fam', 'hon',
  'sweetheart', 'darling', 'sis', 'bestie', 'bestiee',
];

const GREETING_RE =
  /^(?:hey|hi|hello|yo|sup|heya|hiya|morning|afternoon|evening)\s+([a-z][a-z']{1,20})\b/i;

const OPENER_RE =
  /^(hey|hi|hello|yo|sup|heya|hiya|morning|afternoon|evening|miss you|thinking of you|guess what|ok so|so+|wait|omg|hah+|lol)\b/i;

const CLOSER_RE =
  /^(ok|k|kk|okay|night|gn|goodnight|bye|cya|ttyl|talk later|love you|ily|xx+|sleep well|drive safe)\b/i;

const TEASE_RE =
  /\b(haha+|lol|lmao|jk|idiot|dumb|stupid|shut up|miss you|love you|babe|baby|cute|hot|sexy|nerd|loser|silly)\b/i;

const QUESTION_RE = /\?/;

export interface RelationshipCard {
  withPerson: string;
  addressForms: string[];
  recurringPeople: string[];
  recurringPlaces: string[];
  toneWithYou: string;
  sharedMessageCount: number;
  sharedConversationCount: number;
  /** How they open messages to you */
  openers: string[];
  /** Typical short closers / wind-downs with you */
  closers: string[];
  /** Fraction of their messages to you that contain a question */
  questionBackRate: number;
  avgWordsWithYou: number;
  avgWordsGlobal: number;
  /** Short real lines that show teasing / warmth with you */
  teasingSamples: string[];
  /** Shared conversation ids (for few-shot + RAG bias) */
  sharedConversationIds: string[];
  /** Paragraph ready for system-prompt injection */
  summary: string;
  /** Dedicated “register with you” block for the system prompt */
  registerSummary: string;
}

/** Prefer env override; else person present in the most 2-person chats. */
export function inferSelfSender(db: DatabaseType): string | null {
  const env = process.env.DOPPELGANGER_SELF_NAME?.trim();
  if (env) return env;

  const row = db
    .prepare(
      `
    WITH dyads AS (
      SELECT conversation_id
      FROM messages
      WHERE is_system = 0
      GROUP BY conversation_id
      HAVING COUNT(DISTINCT sender) = 2
    )
    SELECT m.sender AS sender, COUNT(DISTINCT m.conversation_id) AS chats
    FROM messages m
    JOIN dyads d ON d.conversation_id = m.conversation_id
    WHERE m.is_system = 0
    GROUP BY m.sender
    ORDER BY chats DESC, COUNT(*) DESC
    LIMIT 1
  `
    )
    .get() as { sender: string; chats: number } | undefined;

  return row?.sender ?? null;
}

function firstNames(full: string): string[] {
  const parts = full.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return [];
  const out = [parts[0].toLowerCase()];
  if (parts.length > 1) out.push(parts[parts.length - 1].toLowerCase());
  return out;
}

function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function topPatterns(
  contents: string[],
  matcher: (text: string) => string | null,
  limit = 6
): string[] {
  const counts = new Map<string, number>();
  for (const raw of contents) {
    const hit = matcher(raw.trim());
    if (!hit) continue;
    const key = hit.toLowerCase();
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([k]) => k);
}

function extractAddressForms(contents: string[], selfName: string): string[] {
  const selfParts = firstNames(selfName);
  const counts = new Map<string, number>();

  const bump = (form: string) => {
    const key = form.trim().toLowerCase();
    if (key.length < 2) return;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  };

  for (const raw of contents) {
    const text = raw.trim();
    if (!text) continue;
    const firstLine = text.split(/\n/)[0].slice(0, 80);

    const greet = firstLine.match(GREETING_RE);
    if (greet?.[1]) {
      const name = greet[1].toLowerCase();
      if (selfParts.some((p) => name.startsWith(p.slice(0, 3)) || p.startsWith(name.slice(0, 3)))) {
        bump(greet[1]);
      } else if (ENDEARMENTS.includes(name)) {
        bump(name);
      }
    }

    const lower = text.toLowerCase();
    for (const e of ENDEARMENTS) {
      if (new RegExp(`\\b${e}\\b`, 'i').test(lower)) bump(e);
    }

    for (const part of selfParts) {
      if (part.length >= 3 && new RegExp(`\\b${part}\\b`, 'i').test(lower)) {
        if (text.length < 40 || GREETING_RE.test(firstLine)) bump(part);
      }
    }
  }

  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([form]) => form);
}

function extractRecurringMentions(
  contents: string[],
  selfName: string,
  personaName: string,
  otherParticipants: string[]
): { people: string[]; places: string[] } {
  const peopleCounts = new Map<string, number>();
  const placeCounts = new Map<string, number>();
  const skip = new Set([
    ...firstNames(selfName),
    ...firstNames(personaName),
    selfName.toLowerCase(),
    personaName.toLowerCase(),
    'instagram',
    'user',
  ]);

  for (const name of otherParticipants) {
    const key = name.trim();
    if (key.length < 2) continue;
    const lower = key.toLowerCase();
    if (skip.has(lower)) continue;
    let hits = 0;
    const re = new RegExp(`\\b${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
    for (const c of contents) {
      if (re.test(c)) hits += 1;
    }
    if (hits >= 2) peopleCounts.set(key, hits);
  }

  for (const raw of contents) {
    const lower = raw.toLowerCase();
    for (const place of PLACE_HINTS) {
      if (lower.includes(place)) {
        placeCounts.set(place, (placeCounts.get(place) ?? 0) + 1);
      }
    }
    for (const m of raw.match(/(?<![.!?]\s|^)([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)/g) ?? []) {
      const key = m.trim();
      const lk = key.toLowerCase();
      if (skip.has(lk) || ENDEARMENTS.includes(lk)) continue;
      if (PLACE_HINTS.has(lk)) {
        placeCounts.set(lk, (placeCounts.get(lk) ?? 0) + 1);
      } else if (key.includes(' ') || key.length >= 4) {
        peopleCounts.set(key, (peopleCounts.get(key) ?? 0) + 1);
      }
    }
  }

  const people = [...peopleCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([n]) => n);
  const places = [...placeCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([n]) => n);

  return { people, places };
}

function toneLabel(avgCompound: number | null, positiveRatio: number, negativeRatio: number): string {
  if (avgCompound == null) return 'tone with you is hard to pin down from available data';
  if (avgCompound > 0.2 && positiveRatio > 0.35) {
    return 'usually warm / upbeat with you';
  }
  if (avgCompound < -0.1 || negativeRatio > 0.3) {
    return 'often blunt or cooler with you than average';
  }
  if (avgCompound > 0.05) return 'mildly positive / friendly with you';
  return 'fairly neutral with you';
}

function pickTeasingSamples(contents: string[], limit = 5): string[] {
  const scored = contents
    .map((c) => c.trim())
    .filter((c) => c.length >= 8 && c.length <= 120 && TEASE_RE.test(c))
    .map((c) => ({
      c,
      score: (TEASE_RE.test(c) ? 2 : 0) + Math.min(wordCount(c), 20) / 10,
    }))
    .sort((a, b) => b.score - a.score);

  const out: string[] = [];
  const seen = new Set<string>();
  for (const row of scored) {
    const key = row.c.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(row.c);
    if (out.length >= limit) break;
  }
  return out;
}

function buildSummary(persona: string, card: Omit<RelationshipCard, 'summary' | 'registerSummary'>): string {
  const bits: string[] = [];
  bits.push(`Relationship with ${card.withPerson}:`);
  if (card.addressForms.length) {
    bits.push(`often addresses them as ${card.addressForms.map((a) => `"${a}"`).join(', ')}.`);
  } else {
    bits.push('no strong nickname pattern detected.');
  }
  bits.push(`Tone: ${card.toneWithYou}.`);
  if (card.recurringPeople.length) {
    bits.push(`Recurring people they mention: ${card.recurringPeople.slice(0, 5).join(', ')}.`);
  }
  if (card.recurringPlaces.length) {
    bits.push(`Recurring places/topics: ${card.recurringPlaces.slice(0, 5).join(', ')}.`);
  }
  bits.push(
    `Shared history: ~${card.sharedMessageCount} of ${persona}'s messages across ${card.sharedConversationCount} chats with ${card.withPerson}.`
  );
  return bits.join(' ');
}

function buildRegisterSummary(
  persona: string,
  card: Omit<RelationshipCard, 'summary' | 'registerSummary'>
): string {
  const bits: string[] = [];
  bits.push(`How ${persona} texts ${card.withPerson} specifically (use this register):`);
  bits.push(
    `With ${card.withPerson} they average ~${card.avgWordsWithYou} words/message` +
      (card.avgWordsGlobal > 0
        ? ` (vs ~${card.avgWordsGlobal} across all chats).`
        : '.')
  );
  if (card.openers.length) {
    bits.push(`Common openers with them: ${card.openers.map((o) => `"${o}"`).join(', ')}.`);
  }
  if (card.closers.length) {
    bits.push(`Common wind-downs: ${card.closers.map((c) => `"${c}"`).join(', ')}.`);
  }
  bits.push(
    `They ask a question back in ~${Math.round(card.questionBackRate * 100)}% of messages to ${card.withPerson}.`
  );
  if (card.addressForms.length) {
    bits.push(`Address them the way they do: ${card.addressForms.map((a) => `"${a}"`).join(', ')}.`);
  }
  bits.push(`Tone with ${card.withPerson}: ${card.toneWithYou}.`);
  if (card.teasingSamples.length) {
    bits.push(
      `Examples of their teasing/warmth with ${card.withPerson}: ` +
        card.teasingSamples.map((s) => `“${s}”`).join(' · ')
    );
  }
  bits.push(
    `Prefer patterns from chats with ${card.withPerson} over how they sound in other conversations.`
  );
  return bits.join(' ');
}

/**
 * Build a relationship / with-you register card for `personaSender` relative to `selfSender`.
 * Returns null if there isn't enough shared history.
 */
export function buildRelationshipCard(
  db: DatabaseType,
  personaSender: string,
  selfSender: string
): RelationshipCard | null {
  if (!personaSender || !selfSender || personaSender === selfSender) return null;

  const sharedConvs = db
    .prepare(
      `
    SELECT conversation_id
    FROM messages
    WHERE is_system = 0
      AND conversation_id IN (
        SELECT conversation_id FROM messages WHERE sender = ? AND is_system = 0
      )
      AND conversation_id IN (
        SELECT conversation_id FROM messages WHERE sender = ? AND is_system = 0
      )
    GROUP BY conversation_id
  `
    )
    .all(personaSender, selfSender) as Array<{ conversation_id: string }>;

  if (sharedConvs.length === 0) return null;

  // Prefer 1:1 dyads with self first for register
  const dyadFirst = [...sharedConvs].sort((a, b) => {
    const countA = (
      db
        .prepare(
          `SELECT COUNT(DISTINCT sender) AS n FROM messages WHERE conversation_id = ? AND is_system = 0`
        )
        .get(a.conversation_id) as { n: number }
    ).n;
    const countB = (
      db
        .prepare(
          `SELECT COUNT(DISTINCT sender) AS n FROM messages WHERE conversation_id = ? AND is_system = 0`
        )
        .get(b.conversation_id) as { n: number }
    ).n;
    return countA - countB;
  });

  const convIds = dyadFirst.map((c) => c.conversation_id);
  const placeholders = convIds.map(() => '?').join(',');

  const personaMsgs = db
    .prepare(
      `
    SELECT content
    FROM messages
    WHERE sender = ?
      AND is_system = 0
      AND content IS NOT NULL
      AND TRIM(content) != ''
      AND conversation_id IN (${placeholders})
    ORDER BY timestamp_ms DESC
    LIMIT 3000
  `
    )
    .all(personaSender, ...convIds) as Array<{ content: string }>;

  if (personaMsgs.length < 8) return null;

  const contents = personaMsgs.map((m) => m.content);

  const others = db
    .prepare(
      `
    SELECT sender, COUNT(*) AS n
    FROM messages
    WHERE is_system = 0
      AND conversation_id IN (${placeholders})
      AND sender != ?
      AND sender != ?
    GROUP BY sender
    HAVING n >= 3
    ORDER BY n DESC
    LIMIT 20
  `
    )
    .all(...convIds, personaSender, selfSender) as Array<{ sender: string; n: number }>;

  const addressForms = extractAddressForms(contents, selfSender);
  const { people, places } = extractRecurringMentions(
    contents,
    selfSender,
    personaSender,
    others.map((o) => o.sender)
  );

  const sent = db
    .prepare(
      `
    SELECT
      AVG(s.compound) AS avg_compound,
      SUM(CASE WHEN s.compound > 0.05 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS positive_ratio,
      SUM(CASE WHEN s.compound < -0.05 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS negative_ratio
    FROM messages m
    JOIN sentiment s ON s.message_id = m.id
    WHERE m.sender = ?
      AND m.is_system = 0
      AND m.conversation_id IN (${placeholders})
  `
    )
    .get(personaSender, ...convIds) as
    | { avg_compound: number | null; positive_ratio: number; negative_ratio: number }
    | undefined;

  const toneWithYou = toneLabel(
    sent?.avg_compound ?? null,
    sent?.positive_ratio ?? 0,
    sent?.negative_ratio ?? 0
  );

  const avgWordsWithYou =
    Math.round(
      (contents.reduce((sum, c) => sum + wordCount(c), 0) / Math.max(contents.length, 1)) * 10
    ) / 10;

  let avgWordsGlobal = avgWordsWithYou;
  try {
    const globalAvgRow = db
      .prepare(
        `
      SELECT AVG(tm.word_count) AS avg_words
      FROM messages m
      LEFT JOIN text_metrics tm ON tm.message_id = m.id
      WHERE m.sender = ?
        AND m.is_system = 0
    `
      )
      .get(personaSender) as { avg_words: number | null } | undefined;
    if (globalAvgRow?.avg_words != null) {
      avgWordsGlobal = Math.round(globalAvgRow.avg_words * 10) / 10;
    }
  } catch {
    // text_metrics may be absent in unit tests
  }

  const questionBackRate =
    Math.round(
      (contents.filter((c) => QUESTION_RE.test(c)).length / Math.max(contents.length, 1)) * 1000
    ) / 1000;

  const openers = topPatterns(
    contents,
    (text) => {
      const m = text.match(OPENER_RE);
      return m ? m[1] : null;
    },
    6
  );

  const closers = topPatterns(
    contents.filter((c) => wordCount(c) <= 6),
    (text) => {
      const m = text.match(CLOSER_RE);
      return m ? m[0] : null;
    },
    6
  );

  const teasingSamples = pickTeasingSamples(contents, 5);

  const base: Omit<RelationshipCard, 'summary' | 'registerSummary'> = {
    withPerson: selfSender,
    addressForms,
    recurringPeople: people,
    recurringPlaces: places,
    toneWithYou,
    sharedMessageCount: personaMsgs.length,
    sharedConversationCount: convIds.length,
    openers,
    closers,
    questionBackRate,
    avgWordsWithYou,
    avgWordsGlobal,
    teasingSamples,
    sharedConversationIds: convIds.slice(0, 30),
  };

  return {
    ...base,
    summary: buildSummary(personaSender, base),
    registerSummary: buildRegisterSummary(personaSender, base),
  };
}

/**
 * Few-shot pairs where the persona is replying to `selfSender` (their register with you).
 */
export function extractWithYouFewShot(
  db: DatabaseType,
  personaSender: string,
  selfSender: string,
  limit = 40
): Array<{
  context: string;
  reply: string;
  conversationId: string;
  source: string;
  withYou: true;
}> {
  if (!personaSender || !selfSender || personaSender === selfSender) return [];

  const rows = db
    .prepare(
      `
    SELECT conversation_id, sender, content, timestamp_ms, source, is_system
    FROM messages
    WHERE content IS NOT NULL AND TRIM(content) != ''
      AND conversation_id IN (
        SELECT conversation_id FROM messages WHERE sender = ?
        INTERSECT
        SELECT conversation_id FROM messages WHERE sender = ?
      )
    ORDER BY conversation_id, timestamp_ms
  `
    )
    .all(personaSender, selfSender) as Array<{
    conversation_id: string;
    sender: string;
    content: string;
    timestamp_ms: number;
    source: string;
    is_system: number;
  }>;

  const pairs: Array<{
    context: string;
    reply: string;
    conversationId: string;
    source: string;
    withYou: true;
  }> = [];
  let prev: (typeof rows)[number] | null = null;

  for (const row of rows) {
    if (row.is_system) {
      prev = null;
      continue;
    }
    if (
      prev &&
      prev.conversation_id === row.conversation_id &&
      prev.sender === selfSender &&
      row.sender === personaSender &&
      row.content.length >= 2 &&
      prev.content.length >= 2 &&
      !/sent \d+ photo|sent a voice message|shared a link/i.test(row.content) &&
      !/sent \d+ photo|sent a voice message|shared a link/i.test(prev.content)
    ) {
      pairs.push({
        context: prev.content.slice(0, 400),
        reply: row.content.slice(0, 500),
        conversationId: row.conversation_id,
        source: row.source,
        withYou: true,
      });
      if (pairs.length >= limit * 2) break;
    }
    prev = row;
  }

  // Prefer longer / more engaged replies when diversifying
  const ranked = [...pairs].sort(
    (a, b) => wordCount(b.reply) - wordCount(a.reply) || a.conversationId.localeCompare(b.conversationId)
  );

  const byConv = new Map<string, typeof ranked>();
  for (const p of ranked) {
    const list = byConv.get(p.conversationId) ?? [];
    list.push(p);
    byConv.set(p.conversationId, list);
  }
  const diversified: typeof pairs = [];
  const queues = [...byConv.values()];
  let i = 0;
  while (diversified.length < limit && queues.some((q) => q.length > 0)) {
    const q = queues[i % queues.length];
    if (q.length > 0) diversified.push(q.shift()!);
    i++;
  }
  return diversified;
}
