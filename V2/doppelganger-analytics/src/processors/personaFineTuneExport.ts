/**
 * Tier-4 fine-tune export scaffold.
 * Writes chat-style JSONL suitable for OpenAI-style fine-tuning (or conversion).
 * Does not train a model — export only.
 */

import fs from 'fs';
import path from 'path';
import { getDb, closeDb } from '../db/client.js';
import { getDashDir, writeDashData } from '../utils/output.js';
import { progressReporter } from '../utils/progressReporter.js';
import { inferSelfSender } from './relationshipCard.js';

export interface FineTuneExportOptions {
  maxPairsPerSender?: number;
  maxSenders?: number;
  /** Prefer replies to the account holder when available */
  preferWithYou?: boolean;
}

interface JsonlRow {
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  meta: {
    sender: string;
    conversationId: string;
    source: string;
    withYou: boolean;
  };
}

function loadStyleMap(): Map<string, string> {
  const map = new Map<string, string>();
  const p = path.join(getDashDir(), 'personaProfiles.json');
  if (!fs.existsSync(p)) return map;
  try {
    const data = JSON.parse(fs.readFileSync(p, 'utf8')) as {
      profiles?: Array<{
        sender: string;
        styleSummary?: string;
        relationshipCard?: { registerSummary?: string } | null;
        turnTakingHabits?: { styleSummary?: string } | null;
      }>;
    };
    for (const profile of data.profiles ?? []) {
      const bits = [profile.styleSummary ?? ''];
      if (profile.relationshipCard?.registerSummary) bits.push(profile.relationshipCard.registerSummary);
      if (profile.turnTakingHabits?.styleSummary) bits.push(profile.turnTakingHabits.styleSummary);
      map.set(profile.sender, bits.filter(Boolean).join('\n'));
    }
  } catch {
    // ignore
  }
  return map;
}

/**
 * Export fine-tuning JSONL + a small README next to dash-data.
 */
export async function exportPersonaFineTune(options?: FineTuneExportOptions): Promise<void> {
  const maxPairsPerSender = options?.maxPairsPerSender ?? 200;
  const maxSenders = options?.maxSenders ?? 20;
  const preferWithYou = options?.preferWithYou ?? true;

  progressReporter.start('Exporting persona fine-tune JSONL…');
  const db = await getDb();

  try {
    const selfSender = inferSelfSender(db);
    const styleMap = loadStyleMap();

    const senders = db
      .prepare(
        `
      SELECT sender, COUNT(*) AS n
      FROM messages
      WHERE is_system = 0 AND content IS NOT NULL AND TRIM(content) != ''
      GROUP BY sender
      HAVING n >= 100
      ORDER BY n DESC
      LIMIT ?
    `
      )
      .all(maxSenders) as Array<{ sender: string; n: number }>;

    const rowsOut: JsonlRow[] = [];

    for (const { sender } of senders) {
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

      const pairs: Array<{
        context: string;
        reply: string;
        conversationId: string;
        source: string;
        withYou: boolean;
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
          prev.sender !== row.sender &&
          row.sender === sender &&
          row.content.length >= 4 &&
          prev.content.length >= 2
        ) {
          const withYou = Boolean(selfSender && prev.sender === selfSender);
          if (preferWithYou && selfSender && !withYou && pairs.length > maxPairsPerSender / 2) {
            prev = row;
            continue;
          }
          pairs.push({
            context: prev.content.slice(0, 500),
            reply: row.content.slice(0, 600),
            conversationId: row.conversation_id,
            source: row.source,
            withYou,
          });
          if (pairs.length >= maxPairsPerSender * 2) break;
        }
        prev = row;
      }

      // Prefer with-you, then longer replies
      pairs.sort((a, b) => Number(b.withYou) - Number(a.withYou) || b.reply.length - a.reply.length);
      const selected = pairs.slice(0, maxPairsPerSender);
      const system =
        styleMap.get(sender) ||
        `You are ${sender} texting. Match their style exactly. Reply with only the message text.`;

      for (const pair of selected) {
        rowsOut.push({
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: pair.context },
            { role: 'assistant', content: pair.reply },
          ],
          meta: {
            sender,
            conversationId: pair.conversationId,
            source: pair.source,
            withYou: pair.withYou,
          },
        });
      }
    }

    const dir = getDashDir();
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const jsonlPath = path.join(dir, 'personaFineTune.jsonl');
    const body = rowsOut.map((r) => JSON.stringify(r)).join('\n') + (rowsOut.length ? '\n' : '');
    fs.writeFileSync(jsonlPath, body);

    writeDashData(
      'personaFineTune.meta.json',
      {
        generatedAt: new Date().toISOString(),
        format: 'chat-jsonl',
        path: 'personaFineTune.jsonl',
        examples: rowsOut.length,
        senders: senders.length,
        preferWithYou,
        inferredSelf: selfSender,
        notes: [
          'This is an export scaffold for Tier-4 fine-tuning — no model is trained here.',
          'Strip the meta field before uploading to OpenAI if required.',
          'Prefer with-you pairs when inferredSelf is set.',
          'Validate privacy/consent before training on personal messages.',
          'After fine-tuning, point Persona Chat at the fine-tuned model id.',
        ],
      },
      { pretty: true }
    );

    const readme = path.join(dir, 'PERSONA_FINETUNE.md');
    fs.writeFileSync(
      readme,
      `# Persona fine-tune export

Generated \`${rowsOut.length}\` chat examples in \`personaFineTune.jsonl\`.

## Next steps

1. Review consent/privacy — only train on people who agreed.
2. Convert/strip \`meta\` if your trainer rejects extra fields.
3. Upload to your fine-tune provider (OpenAI, etc.) or local LoRA pipeline.
4. Evaluate with \`npm run persona-eval -- --live\` against held-out pairs.
5. Wire the resulting model id into dashboard Claude settings (or a custom provider).

This repo does **not** run the training job for you.
`
    );

    progressReporter.success(
      `Fine-tune export: ${rowsOut.length} examples → dash-data/personaFineTune.jsonl`
    );
  } finally {
    await closeDb(db);
  }
}
