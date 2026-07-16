import { getDb } from '../db/client.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface MessageLengthDistribution {
  conversation_id: string;
  bucket: string;
  count: number;
  percentage: number;
  label: string;
  minWords: number;
  maxWords: number;
}

// Define the length buckets
const LENGTH_BUCKETS = [
  { range: '1-2', label: 'Very Short (1-2 words)', minWords: 1, maxWords: 2 },
  { range: '3-5', label: 'Short (3-5 words)', minWords: 3, maxWords: 5 },
  { range: '6-10', label: 'Medium (6-10 words)', minWords: 6, maxWords: 10 },
  { range: '11-20', label: 'Long (11-20 words)', minWords: 11, maxWords: 20 },
  { range: '21-50', label: 'Very Long (21-50 words)', minWords: 21, maxWords: 50 },
  { range: '50+', label: 'Extremely Long (50+ words)', minWords: 50, maxWords: 999 }
];

function categorizeMsgLength(wordCount: number): typeof LENGTH_BUCKETS[0] {
  for (const bucket of LENGTH_BUCKETS) {
    if (wordCount >= bucket.minWords && (bucket.maxWords === 999 || wordCount <= bucket.maxWords)) {
      return bucket;
    }
  }
  return LENGTH_BUCKETS[0]; // fallback to first bucket
}

export async function computeMessageLengthMetrics(): Promise<void> {
  progressReporter.start('Computing message length distribution metrics...');
  
  try {
    const db = await getDb();
    
    // Query message word counts by conversation
    const messageLengths = db.prepare(`
      SELECT 
        m.conversation_id,
        tm.word_count
      FROM messages m
      JOIN text_metrics tm ON m.id = tm.message_id
      WHERE tm.word_count > 0
      ORDER BY m.conversation_id
    `).all() as { conversation_id: string; word_count: number }[];

    progressReporter.update(`  Processing ${messageLengths.length} messages with length data...`);

    // Group by conversation and bucket
    const conversationDistributions = new Map<string, Map<string, number>>();
    
    messageLengths.forEach(msg => {
      if (!conversationDistributions.has(msg.conversation_id)) {
        conversationDistributions.set(msg.conversation_id, new Map());
      }
      
      const bucketMap = conversationDistributions.get(msg.conversation_id)!;
      const bucket = categorizeMsgLength(msg.word_count);
      
      if (!bucketMap.has(bucket.range)) {
        bucketMap.set(bucket.range, 0);
      }
      
      bucketMap.set(bucket.range, bucketMap.get(bucket.range)! + 1);
    });

    // Convert to output format
    const results: MessageLengthDistribution[] = [];
    
    for (const [conversationId, bucketMap] of conversationDistributions.entries()) {
      const totalMessages = Array.from(bucketMap.values()).reduce((sum, count) => sum + count, 0);
      
      // Ensure all buckets are present for each conversation
      for (const bucket of LENGTH_BUCKETS) {
        const count = bucketMap.get(bucket.range) || 0;
        results.push({
          conversation_id: conversationId,
          bucket: bucket.range,
          count: count,
          percentage: totalMessages > 0 ? (count / totalMessages) * 100 : 0,
          label: bucket.label,
          minWords: bucket.minWords,
          maxWords: bucket.maxWords
        });
      }
    }

    // Export results
    progressReporter.update('Exporting message length distribution metrics...');
    writeDashData('messageLengthDistribution.json', results);

    progressReporter.success(`Message length distribution computed: ${results.length} records across ${conversationDistributions.size} conversations`);
    
  } catch (error) {
    console.error(chalk.red('❌ Error computing message length metrics:'), error);
    throw error;
  }
} 