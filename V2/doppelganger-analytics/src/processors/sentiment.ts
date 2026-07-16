// src/processors/sentiment.ts
import { Database } from 'better-sqlite3';
import { getDb, closeDb } from '../db/client.js';
import { getMeta, setMeta } from '../db/meta.js';
import { decodeInstagramUnicode } from '../utils/unicodeDecoder.js';
import { isSystemMessage } from '../utils/messageFilters.js';
import { analyzeSentimentBatch } from '../services/sentimentService.js';
import { progressReporter } from '../utils/progressReporter.js';
import ProgressBar from 'progress';

// Bump this whenever the scoring engine, its skip filter, or output semantics
// change; on the next run analyzeSentiment() detects the mismatch, wipes stale
// scores, and rescores every message. (Suffix -f2: unified system-message filter.)
const SENTIMENT_ENGINE_VERSION = 'vader-sentiment-1.1.3-f3';
const SENTIMENT_ENGINE_META_KEY = 'sentiment_engine_version';

interface MessageRow {
  id: number;
  content: string | null;
  is_system: number;
}

interface SentimentResult {
  message_id: number;
  compound: number;
  positive: number;
  negative: number;
  neutral: number;
}

async function processBatch(db: Database, messages: MessageRow[], progressBar: ProgressBar): Promise<void> {
  const batchSize = 250; // Much larger batch size for local processing
  const insertSentiment = db.prepare(`
    INSERT INTO sentiment (message_id, compound, positive, negative, neutral)
    VALUES (@message_id, @compound, @positive, @negative, @neutral)
  `);

  const insertMany = db.transaction((rows: SentimentResult[]) => {
    let count = 0;
    for (const row of rows) {
      insertSentiment.run(row);
      count++;
    }
    return count;
  });

  for (let i = 0; i < messages.length; i += batchSize) {
    const batch = messages.slice(i, i + batchSize);
    
    // Filter out messages that shouldn't be analyzed
    const validMessages = batch.filter(msg => !isSystemMessage(msg.content, msg.is_system));
    
    if (validMessages.length === 0) {
      progressBar.tick(batch.length);
      continue;
    }

    // Extract texts for batch analysis with Unicode decoding
    const texts = validMessages.map(msg => decodeInstagramUnicode(msg.content!));
    
    // Analyze sentiment for entire batch locally
    const sentimentResults = await analyzeSentimentBatch(texts);
    
    // Convert results to database format
    const results: SentimentResult[] = validMessages.map((msg, index) => ({
      message_id: msg.id,
      compound: sentimentResults[index].compound,
      positive: sentimentResults[index].positive,
      negative: sentimentResults[index].negative,
      neutral: sentimentResults[index].neutral
    }));

    // Add neutral sentiment for skipped messages
    const skippedMessages = batch.filter(msg => isSystemMessage(msg.content, msg.is_system));
    
    for (const msg of skippedMessages) {
      results.push({
        message_id: msg.id,
        compound: 0,
        positive: 0,
        negative: 0,
        neutral: 1
      });
    }

    insertMany(results);
    progressBar.tick(batch.length);
  }
}

export async function analyzeSentiment(): Promise<void> {
  progressReporter.start('Computing sentiment metrics...');
  const db = await getDb();

  try {
    // If the scoring engine changed since these rows were written, they are
    // stale: wipe them so every message is rescored below.
    const storedVersion = getMeta(db, SENTIMENT_ENGINE_META_KEY);
    if (storedVersion !== SENTIMENT_ENGINE_VERSION) {
      progressReporter.update(
        `Sentiment engine changed (${storedVersion ?? 'none'} → ${SENTIMENT_ENGINE_VERSION}); rescoring all messages...`
      );
      db.prepare('DELETE FROM sentiment').run();
      setMeta(db, SENTIMENT_ENGINE_META_KEY, SENTIMENT_ENGINE_VERSION);
    }

    // Get messages that need sentiment analysis
    const query = `
      SELECT m.id, m.content, m.is_system
      FROM messages m
      LEFT JOIN sentiment s ON m.id = s.message_id
      WHERE s.message_id IS NULL
      AND m.content IS NOT NULL
      AND length(m.content) > 2
      ORDER BY m.id DESC
    `;

    const messages = db.prepare(query).all() as MessageRow[];

    if (messages.length === 0) {
      progressReporter.success('All messages already have sentiment data');
      return;
    }

    progressReporter.update(`Processing sentiment for ${messages.length} messages...`);

    const bar = progressReporter.createProgressBar(messages.length, 'Analyzing sentiment');

    await processBatch(db, messages, bar);
    progressReporter.success('Sentiment metrics computed');
  } catch (error) {
    progressReporter.error('Error computing sentiment metrics');
    console.error(error);
    throw error;
  } finally {
    await closeDb(db);
  }
}
