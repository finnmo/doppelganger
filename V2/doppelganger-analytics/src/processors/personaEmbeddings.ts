/**
 * Generate-time message embeddings for vector RAG in Persona Chat.
 * Skips cleanly when no OpenAI/Voyage key is configured.
 */

import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';
import { getEmbeddingCredentials } from '../embeddings/credentials.js';
import { embedTexts } from '../embeddings/client.js';
import { float32ToBuffer, EMBEDDING_MODELS } from '../embeddings/vectors.js';
import chalk from 'chalk';

const MAX_PER_SENDER = 2500;
const MIN_CONTENT_LEN = 8;
const MIN_SENDER_MESSAGES = 20;
const BATCH_SIZE = 64;

export async function computePersonaEmbeddings(): Promise<void> {
  progressReporter.start('Computing persona embeddings...');
  const creds = getEmbeddingCredentials();

  if (!creds) {
    console.log(
      chalk.yellow(
        '⚠️  No embeddings API key — skipping vector index. ' +
          'Add an OpenAI or Voyage key in dashboard API settings (or OPENAI_API_KEY / VOYAGE_API_KEY). ' +
          'Persona chat will keep using keyword RAG until then.'
      )
    );
    writeDashData('personaEmbeddings.json', {
      status: 'skipped',
      reason: 'no_credentials',
      generatedAt: new Date().toISOString(),
    });
    progressReporter.success('Persona embeddings skipped (no credentials)');
    return;
  }

  const db = await getDb();
  const modelMeta = EMBEDDING_MODELS[creds.provider];
  const model = creds.model || modelMeta.id;

  try {
    // Ensure table exists even if migrate was run on an older binary
    db.exec(`
      CREATE TABLE IF NOT EXISTS message_embeddings (
        message_id  INTEGER PRIMARY KEY,
        provider    TEXT    NOT NULL,
        model       TEXT    NOT NULL,
        dims        INTEGER NOT NULL,
        embedding   BLOB    NOT NULL,
        FOREIGN KEY(message_id) REFERENCES messages(id)
      );
    `);

    const senders = db.prepare(`
      SELECT sender, COUNT(*) AS n
      FROM messages
      WHERE is_system = 0
        AND content IS NOT NULL
        AND LENGTH(TRIM(content)) >= ?
      GROUP BY sender
      HAVING n >= ?
      ORDER BY n DESC
    `).all(MIN_CONTENT_LEN, MIN_SENDER_MESSAGES) as Array<{ sender: string; n: number }>;

    if (senders.length === 0) {
      writeDashData('personaEmbeddings.json', {
        status: 'skipped',
        reason: 'no_eligible_senders',
        generatedAt: new Date().toISOString(),
      });
      progressReporter.success('Persona embeddings skipped (no eligible senders)');
      return;
    }

    // Drop stale rows from a different model so we don't mix spaces
    db.prepare(`DELETE FROM message_embeddings WHERE model != ?`).run(model);

    const already = new Set(
      (
        db.prepare(`SELECT message_id FROM message_embeddings WHERE model = ?`).all(model) as Array<{
          message_id: number;
        }>
      ).map((r) => r.message_id)
    );

    const insert = db.prepare(`
      INSERT OR REPLACE INTO message_embeddings (message_id, provider, model, dims, embedding)
      VALUES (@message_id, @provider, @model, @dims, @embedding)
    `);

    let embedded = 0;
    let skippedExisting = 0;
    let errors = 0;

    progressReporter.update(
      `Embedding with ${creds.provider}/${model} for ${senders.length} senders (max ${MAX_PER_SENDER}/sender)…`
    );

    for (const { sender, n } of senders) {
      const rows = db.prepare(`
        SELECT id, content
        FROM messages
        WHERE sender = ?
          AND is_system = 0
          AND content IS NOT NULL
          AND LENGTH(TRIM(content)) >= ?
        ORDER BY timestamp_ms DESC
        LIMIT ?
      `).all(sender, MIN_CONTENT_LEN, MAX_PER_SENDER) as Array<{ id: number; content: string }>;

      const todo = rows.filter((r) => !already.has(r.id));
      skippedExisting += rows.length - todo.length;

      for (let i = 0; i < todo.length; i += BATCH_SIZE) {
        const batch = todo.slice(i, i + BATCH_SIZE);
        try {
          const vectors = await embedTexts(
            batch.map((b) => b.content.trim()),
            { provider: creds.provider, apiKey: creds.apiKey, model }
          );

          const tx = db.transaction(() => {
            for (let j = 0; j < batch.length; j++) {
              const vec = vectors[j];
              if (!vec || vec.length === 0) continue;
              insert.run({
                message_id: batch[j].id,
                provider: creds.provider,
                model,
                dims: vec.length,
                embedding: float32ToBuffer(vec),
              });
              embedded++;
            }
          });
          tx();
        } catch (err) {
          errors++;
          console.warn(
            chalk.yellow(
              `Embedding batch failed for ${sender}: ${err instanceof Error ? err.message : String(err)}`
            )
          );
          // Don't abort the whole pipeline — keep what we have
          if (errors >= 5) {
            console.warn(chalk.yellow('Too many embedding errors — stopping early.'));
            break;
          }
        }
      }

      progressReporter.update(
        `Embeddings: ${embedded.toLocaleString()} new · ${sender} (${n.toLocaleString()} msgs)`
      );
      if (errors >= 5) break;
    }

    const total = (
      db.prepare(`SELECT COUNT(*) AS n FROM message_embeddings WHERE model = ?`).get(model) as {
        n: number;
      }
    ).n;

    writeDashData('personaEmbeddings.json', {
      status: 'ok',
      provider: creds.provider,
      model,
      source: creds.source,
      embeddedNew: embedded,
      skippedExisting,
      totalIndexed: total,
      senders: senders.length,
      maxPerSender: MAX_PER_SENDER,
      generatedAt: new Date().toISOString(),
      errors,
    });

    progressReporter.success(
      `Persona embeddings ready: ${total.toLocaleString()} vectors (${creds.provider}/${model})`
    );
  } finally {
    await closeDb(db);
  }
}
