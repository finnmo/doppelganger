import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

// Import the core text processing logic
import { processText } from './textProcessor.core.js';
export { processText } from './textProcessor.core.js';
export type { TextMetrics } from './textProcessor.core.js';

const DEFAULT_TEXT_METRICS = {
  summary: {
    totalMessages: 0,
    averageWordCount: 0,
    averageEmojiCount: 0,
    averageUrlCount: 0,
    totalEmojis: 0,
    totalUrls: 0
  }
};

export async function computeTextMetrics(): Promise<void> {
  progressReporter.start('Computing text metrics...');
  const db = await getDb();

  try {
    const rows = db.prepare(`
      SELECT id, content
      FROM messages
      WHERE content IS NOT NULL
    `).all() as { id: number; content: string | null }[];

    if (rows.length === 0) {
      progressReporter.update('No messages found. Generating default metrics...');
      writeDashData('textMetrics.json', DEFAULT_TEXT_METRICS);
      progressReporter.success('Generated default text metrics');
      return;
    }

    // Process messages with progress bar
    progressReporter.update(`Processing ${rows.length.toLocaleString()} messages with Unicode decoding...`);
    const progressBar = progressReporter.createProgressBar(rows.length, 'Processing messages');
    
    const processedRows = rows.map((row) => {
      const metrics = processText(row.content);
      progressBar.tick(1);
      
      return {
        id: row.id,
        word_count: metrics.wordCount,
        emoji_count: metrics.emojiCount,
        url_count: metrics.urlCount
      };
    });

    // Insert metrics
    progressReporter.update('Inserting metrics into database...');
    const insertMetric = db.prepare(`
      INSERT OR REPLACE INTO text_metrics (message_id, word_count, emoji_count, url_count)
      VALUES (@id, @word_count, @emoji_count, @url_count)
    `);

    const insertMany = db.transaction((rows) => {
      let count = 0;
      for (const row of rows) {
        insertMetric.run(row);
        count++;
      }
      return count;
    });

    const inserted = insertMany(processedRows);
    progressReporter.update(`Text metrics computed for ${inserted} messages with Unicode decoding`);

    // Export metrics
    progressReporter.update('Exporting metrics...');
    const metrics = {
      summary: {
        totalMessages: rows.length,
        averageWordCount: processedRows.reduce((sum, row) => sum + row.word_count, 0) / rows.length,
        averageEmojiCount: processedRows.reduce((sum, row) => sum + row.emoji_count, 0) / rows.length,
        averageUrlCount: processedRows.reduce((sum, row) => sum + row.url_count, 0) / rows.length,
        totalEmojis: processedRows.reduce((sum, row) => sum + row.emoji_count, 0),
        totalUrls: processedRows.reduce((sum, row) => sum + row.url_count, 0)
      }
    };

    writeDashData('textMetrics.json', metrics);

    progressReporter.success('Text metrics exported with Unicode decoding improvements');
  } catch (error) {
    progressReporter.error('Error computing text metrics');
    console.error(error);
    // Emit default metrics so the dashboard has a valid file, then rethrow so
    // the failure is visible instead of silently reported as success.
    writeDashData('textMetrics.json', DEFAULT_TEXT_METRICS);
    throw error;
  } finally {
    await closeDb(db);
  }
} 