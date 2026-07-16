/**
 * Held-out persona eval harness.
 * Samples real (context → reply) pairs not used in few-shot, optionally
 * generates model replies (live) and scores style similarity.
 */

import fs from 'fs';
import path from 'path';
import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData, getDashDir } from '../utils/output.js';
import { getAnthropicCredentialsForEval } from '../persona/evalAnthropic.js';
import chalk from 'chalk';

export interface EvalPair {
  context: string;
  actualReply: string;
  conversationId: string;
  source: string;
  generatedReply?: string;
  scores?: {
    lengthRatio: number;
    tokenJaccard: number;
    /** Placeholder for human blind rating 1–5 */
    humanStyleRating: number | null;
  };
}

export interface SenderEval {
  sender: string;
  pairCount: number;
  medianTokenJaccard: number | null;
  medianLengthRatio: number | null;
  pairs: EvalPair[];
}

const STOP = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
  'is', 'it', 'this', 'that', 'i', 'you', 'me', 'my', 'your', 'was', 'are', 'be',
]);

function tokenize(text: string): Set<string> {
  return new Set(
    (text.toLowerCase().match(/[a-z0-9']{2,}/g) ?? []).filter((t) => !STOP.has(t))
  );
}

export function tokenJaccard(a: string, b: string): number {
  const A = tokenize(a);
  const B = tokenize(b);
  if (A.size === 0 && B.size === 0) return 1;
  if (A.size === 0 || B.size === 0) return 0;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter += 1;
  return inter / (A.size + B.size - inter);
}

export function lengthRatio(actual: string, generated: string): number {
  const aw = Math.max(1, actual.trim().split(/\s+/).length);
  const gw = Math.max(1, generated.trim().split(/\s+/).length);
  return Math.min(aw, gw) / Math.max(aw, gw);
}

function median(nums: number[]): number | null {
  if (nums.length === 0) return null;
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function loadFewShotKeys(): Set<string> {
  const candidates = [
    path.join(getDashDir(), 'personaProfiles.json'),
    path.join(process.cwd(), 'dash-data', 'personaProfiles.json'),
  ];
  const keys = new Set<string>();
  for (const p of candidates) {
    if (!fs.existsSync(p)) continue;
    try {
      const data = JSON.parse(fs.readFileSync(p, 'utf8')) as {
        profiles?: Array<{
          fewShotExamples?: Array<{ context: string; reply: string }>;
          withYouFewShotExamples?: Array<{ context: string; reply: string }>;
        }>;
      };
      for (const profile of data.profiles ?? []) {
        for (const ex of [
          ...(profile.fewShotExamples ?? []),
          ...(profile.withYouFewShotExamples ?? []),
        ]) {
          keys.add(`${ex.context.trim()}:::${ex.reply.trim()}`);
        }
      }
      break;
    } catch {
      // ignore
    }
  }
  return keys;
}

function sampleHeldOutPairs(
  db: import('better-sqlite3').Database,
  sender: string,
  fewShotKeys: Set<string>,
  limit: number
): EvalPair[] {
  const rows = db
    .prepare(
      `
    SELECT conversation_id, sender, content, timestamp_ms, source, is_system
    FROM messages
    WHERE content IS NOT NULL AND TRIM(content) != ''
      AND conversation_id IN (
        SELECT DISTINCT conversation_id FROM messages WHERE sender = ?
      )
    ORDER BY conversation_id, timestamp_ms
  `
    )
    .all(sender) as Array<{
    conversation_id: string;
    sender: string;
    content: string;
    timestamp_ms: number;
    source: string;
    is_system: number;
  }>;

  const candidates: EvalPair[] = [];
  let prev: (typeof rows)[number] | null = null;
  for (const row of rows) {
    if (row.is_system) {
      prev = null;
      continue;
    }
    if (
      prev &&
      prev.conversation_id === row.conversation_id &&
      prev.sender !== row.sender &&
      row.sender === sender &&
      row.content.length >= 8 &&
      prev.content.length >= 4 &&
      !/sent \d+ photo|sent a voice message|shared a link/i.test(row.content)
    ) {
      const context = prev.content.slice(0, 400);
      const actualReply = row.content.slice(0, 500);
      const key = `${context.trim()}:::${actualReply.trim()}`;
      if (!fewShotKeys.has(key)) {
        candidates.push({
          context,
          actualReply,
          conversationId: row.conversation_id,
          source: row.source,
        });
      }
    }
    prev = row;
  }

  // Prefer substantive replies; diversify by conversation
  candidates.sort((a, b) => b.actualReply.length - a.actualReply.length);
  const byConv = new Map<string, EvalPair[]>();
  for (const c of candidates) {
    const list = byConv.get(c.conversationId) ?? [];
    list.push(c);
    byConv.set(c.conversationId, list);
  }
  const out: EvalPair[] = [];
  const queues = [...byConv.values()];
  let i = 0;
  while (out.length < limit && queues.some((q) => q.length > 0)) {
    const q = queues[i % queues.length];
    if (q.length > 0) out.push(q.shift()!);
    i++;
  }
  return out;
}

async function generateReply(
  apiKey: string,
  model: string,
  sender: string,
  styleSummary: string,
  context: string
): Promise<string> {
  const system = [
    `You are ${sender} texting. Match their style.`,
    styleSummary,
    'Reply with only the message text. Keep the conversation going naturally.',
  ].join('\n');

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model,
      max_tokens: 300,
      system,
      messages: [{ role: 'user', content: context }],
    }),
    signal: AbortSignal.timeout(60_000),
  });

  const data = (await res.json()) as {
    content?: Array<{ type: string; text?: string }>;
    error?: { message?: string };
  };
  if (!res.ok) {
    throw new Error(data.error?.message || `Anthropic error ${res.status}`);
  }
  return (data.content ?? [])
    .filter((b) => b.type === 'text' && b.text)
    .map((b) => b.text!.trim())
    .join('\n')
    .trim();
}

function loadStyleSummaries(): Map<string, string> {
  const map = new Map<string, string>();
  const p = path.join(getDashDir(), 'personaProfiles.json');
  if (!fs.existsSync(p)) return map;
  try {
    const data = JSON.parse(fs.readFileSync(p, 'utf8')) as {
      profiles?: Array<{ sender: string; styleSummary?: string }>;
    };
    for (const profile of data.profiles ?? []) {
      map.set(profile.sender, profile.styleSummary ?? '');
    }
  } catch {
    // ignore
  }
  return map;
}

/**
 * Build held-out eval set. Pass live:true to generate + score with Claude.
 */
export async function computePersonaEval(options?: {
  live?: boolean;
  maxSenders?: number;
  pairsPerSender?: number;
  senderFilter?: string;
}): Promise<void> {
  const live = options?.live ?? false;
  const maxSenders = options?.maxSenders ?? 12;
  const pairsPerSender = options?.pairsPerSender ?? 25;
  progressReporter.start(
    live ? 'Running live persona eval (Claude)…' : 'Building held-out persona eval set…'
  );

  const db = await getDb();
  try {
    const fewShotKeys = loadFewShotKeys();
    const styleMap = loadStyleSummaries();

    const senders = db
      .prepare(
        `
      SELECT sender, COUNT(*) AS n
      FROM messages
      WHERE is_system = 0 AND content IS NOT NULL AND TRIM(content) != ''
      GROUP BY sender
      HAVING n >= 80
      ORDER BY n DESC
      LIMIT ?
    `
      )
      .all(maxSenders * 2) as Array<{ sender: string; n: number }>;

    const filtered = options?.senderFilter
      ? senders.filter((s) => s.sender === options.senderFilter)
      : senders.slice(0, maxSenders);

    let creds: { apiKey: string; model: string } | null = null;
    if (live) {
      creds = getAnthropicCredentialsForEval();
      if (!creds) {
        console.log(
          chalk.yellow(
            '⚠️  No Anthropic key for live eval — writing held-out pairs only. ' +
              'Set ANTHROPIC_API_KEY or save a key in dashboard settings.'
          )
        );
      }
    }

    const senderEvals: SenderEval[] = [];

    for (const row of filtered) {
      const pairs = sampleHeldOutPairs(db, row.sender, fewShotKeys, pairsPerSender);
      if (pairs.length === 0) continue;

      if (creds) {
        const style = styleMap.get(row.sender) ?? `${row.sender} texts casually.`;
        for (const pair of pairs) {
          try {
            const generated = await generateReply(
              creds.apiKey,
              creds.model,
              row.sender,
              style,
              pair.context
            );
            pair.generatedReply = generated;
            pair.scores = {
              lengthRatio: lengthRatio(pair.actualReply, generated),
              tokenJaccard: tokenJaccard(pair.actualReply, generated),
              humanStyleRating: null,
            };
          } catch (err) {
            pair.generatedReply = undefined;
            pair.scores = {
              lengthRatio: 0,
              tokenJaccard: 0,
              humanStyleRating: null,
            };
            console.log(
              chalk.yellow(
                `  eval fail ${row.sender}: ${err instanceof Error ? err.message : String(err)}`
              )
            );
          }
        }
      }

      const jaccards = pairs.map((p) => p.scores?.tokenJaccard).filter((n): n is number => n != null);
      const lengths = pairs.map((p) => p.scores?.lengthRatio).filter((n): n is number => n != null);

      senderEvals.push({
        sender: row.sender,
        pairCount: pairs.length,
        medianTokenJaccard: median(jaccards),
        medianLengthRatio: median(lengths),
        pairs,
      });
    }

    const scored = senderEvals.flatMap((s) => s.pairs.filter((p) => p.scores));
    const allJ = scored.map((p) => p.scores!.tokenJaccard);
    const allL = scored.map((p) => p.scores!.lengthRatio);

    writeDashData('personaEval.json', {
      generatedAt: new Date().toISOString(),
      mode: creds ? 'scored' : 'held_out_only',
      summary: {
        senders: senderEvals.length,
        pairs: senderEvals.reduce((n, s) => n + s.pairCount, 0),
        scoredPairs: scored.length,
        medianTokenJaccard: median(allJ),
        medianLengthRatio: median(allL),
        targetNote:
          'Aim for median token Jaccard ≥ ~0.25 and human style ≥ 4/5 on a blind sample. ' +
          'Fill humanStyleRating manually or re-run with --live after prompt changes.',
      },
      senders: senderEvals,
    });

    progressReporter.success(
      `Persona eval: ${senderEvals.length} senders, ` +
        `${senderEvals.reduce((n, s) => n + s.pairCount, 0)} held-out pairs` +
        (creds ? ' (live-scored)' : ' (held-out only)')
    );
  } finally {
    await closeDb(db);
  }
}
