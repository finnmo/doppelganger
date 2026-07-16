import { getDb, closeDb } from '../db/client.js';
import type { Database } from 'better-sqlite3';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface ReplyLatencyDistribution {
  conversation_id: string;
  bucket: string;
  count: number;
}

interface SentimentBySenderRecord {
  sender: string;
  conversation_id: string;
  avg_sentiment: number;
  message_count: number;
  avg_positive: number;
  avg_negative: number;
  avg_neutral: number;
}

function round3(value: number): number {
  return Math.round(value * 1000) / 1000;
}

export async function computeAdditionalMetrics(): Promise<void> {
  progressReporter.start('Computing additional metrics...');
  const db = await getDb();

  try {
    // Generate monthly messages with conversation_id
    await generateMonthlyMessages(db);
    
    // Generate reply latency distribution
    await generateReplyLatencyDistribution(db);
    
    // Generate active hours
    await generateActiveHours(db);
    
    // Generate sentiment by sender
    await generateSentimentBySender(db);
    
    // Generate attachment time series
    await generateAttachmentTimeSeries(db);
    
    progressReporter.success('Additional metrics computed successfully');
  } catch (error) {
    console.error(chalk.red('❌ Error computing additional metrics:'), error);
    throw error;
  } finally {
    await closeDb(db);
  }
}

async function generateMonthlyMessages(db: Database): Promise<void> {
  progressReporter.update('Generating monthly messages...');
  
  const monthlyData = db.prepare(`
    SELECT 
      conversation_id,
      strftime('%Y-%m', datetime(timestamp_ms / 1000, 'unixepoch', 'localtime')) as month,
      COUNT(*) as messageCount
    FROM messages 
    WHERE conversation_id IS NOT NULL
    GROUP BY conversation_id, month
    ORDER BY conversation_id, month
  `).all();

  writeDashData('monthly-messages.json', monthlyData);
  progressReporter.success(`Monthly messages saved (${monthlyData.length} records)`);
}

async function generateReplyLatencyDistribution(db: Database): Promise<void> {
  progressReporter.update('Generating reply latency distribution...');

  // Read from the canonical response_times table (populated once by
  // computeResponseTimes) rather than recomputing response gaps here.
  const replies = db.prepare(`
    SELECT conversation_id, latency_ms
    FROM response_times
  `).all() as Array<{ conversation_id: string; latency_ms: number }>;

  progressReporter.update(`Found ${replies.length} response instances`);

  const latencyBuckets: ReplyLatencyDistribution[] = [];
  const buckets = [
    { min: 0, max: 10000, label: '0-10s' },
    { min: 10000, max: 30000, label: '10-30s' },
    { min: 30000, max: 60000, label: '30-60s' },
    { min: 60000, max: 300000, label: '1-5m' },
    { min: 300000, max: 900000, label: '5-15m' },
    { min: 900000, max: 3600000, label: '15-60m' },
    { min: 3600000, max: Infinity, label: '>1h' }
  ];

  // Group by conversation and bucket
  const conversationBuckets = new Map<string, Map<string, number>>();
  
  replies.forEach((reply: { conversation_id: string; latency_ms: number }) => {
    if (!conversationBuckets.has(reply.conversation_id)) {
      conversationBuckets.set(reply.conversation_id, new Map());
    }
    
    const convBuckets = conversationBuckets.get(reply.conversation_id)!;
    const bucket = buckets.find(b => reply.latency_ms >= b.min && reply.latency_ms < b.max);
    
    if (bucket) {
      convBuckets.set(bucket.label, (convBuckets.get(bucket.label) || 0) + 1);
    }
  });

  // Convert to output format
  conversationBuckets.forEach((bucketCounts, conversationId) => {
    bucketCounts.forEach((count, bucketLabel) => {
      latencyBuckets.push({
        conversation_id: conversationId,
        bucket: bucketLabel,
        count
      });
    });
  });

  if (latencyBuckets.length === 0) {
    progressReporter.update('No response data found; this may indicate single-person conversations');
  }

  writeDashData('replyLatencyDistribution.json', latencyBuckets);
  progressReporter.success(`Reply latency distribution saved (${latencyBuckets.length} records)`);
}

async function generateActiveHours(db: Database): Promise<void> {
  progressReporter.update('Generating active hours...');
  
  const activeHoursData = db.prepare(`
    SELECT 
      conversation_id,
      CAST(strftime('%H', datetime(timestamp_ms / 1000, 'unixepoch', 'localtime')) AS INTEGER) as hour,
      CAST(strftime('%w', datetime(timestamp_ms / 1000, 'unixepoch', 'localtime')) AS INTEGER) as day_of_week,
      CASE CAST(strftime('%w', datetime(timestamp_ms / 1000, 'unixepoch', 'localtime')) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
      END as day,
      sender,
      COUNT(*) as count
    FROM messages 
    WHERE conversation_id IS NOT NULL
    GROUP BY conversation_id, hour, day_of_week, day, sender
    ORDER BY conversation_id, day_of_week, hour, sender
  `).all();

  writeDashData('activeHours.json', activeHoursData);
  progressReporter.success(`Active hours saved (${activeHoursData.length} records)`);
}

async function generateSentimentBySender(db: Database): Promise<void> {
  progressReporter.update('Generating sentiment by sender...');

  // Messages without a sentiment row (e.g. very short content) count as neutral (0)
  const sentimentData = db.prepare(`
    SELECT
      m.sender,
      m.conversation_id,
      AVG(COALESCE(s.compound, 0)) as avg_sentiment,
      COUNT(m.id) as message_count,
      AVG(COALESCE(s.positive, 0)) as avg_positive,
      AVG(COALESCE(s.negative, 0)) as avg_negative,
      AVG(COALESCE(s.neutral, 0)) as avg_neutral
    FROM messages m
    LEFT JOIN sentiment s ON m.id = s.message_id
    WHERE m.content IS NOT NULL
    AND m.content != ''
    AND m.sender IS NOT NULL
    AND m.conversation_id IS NOT NULL
    GROUP BY m.sender, m.conversation_id
    HAVING message_count >= 10
    ORDER BY message_count DESC
  `).all() as SentimentBySenderRecord[];

  progressReporter.update(`Processing ${sentimentData.length} sender-conversation pairs...`);

  const records = sentimentData.map(record => ({
    sender: record.sender,
    conversation_id: record.conversation_id,
    avg_sentiment: round3(record.avg_sentiment),
    message_count: record.message_count,
    avg_positive: round3(record.avg_positive),
    avg_negative: round3(record.avg_negative),
    avg_neutral: round3(record.avg_neutral)
  }));

  writeDashData('sentimentBySender.json', records);

  progressReporter.success(`Sentiment by sender saved (${records.length} records)`);

  if (records.length > 0) {
    const avgSentiment = records.reduce((sum, record) => sum + record.avg_sentiment, 0) / records.length;
    const min = Math.min(...records.map(r => r.avg_sentiment));
    const max = Math.max(...records.map(r => r.avg_sentiment));
    progressReporter.update(`Average sentiment: ${avgSentiment.toFixed(3)}`);
    progressReporter.update(`Sentiment range: ${min.toFixed(3)} to ${max.toFixed(3)}`);
  }
}

async function generateAttachmentTimeSeries(db: Database): Promise<void> {
  progressReporter.update('Generating attachment time series...');
  
  const attachmentData = db.prepare(`
    SELECT
      m.conversation_id,
      strftime('%Y-%m', datetime(m.timestamp_ms / 1000, 'unixepoch', 'localtime')) as month,
      SUM((SELECT COUNT(*) FROM message_photos p WHERE p.message_id = m.id)) as photo_count,
      SUM((SELECT COUNT(*) FROM message_videos v WHERE v.message_id = m.id)) as video_count
    FROM messages m
    WHERE m.conversation_id IS NOT NULL
      AND (m.has_photos = 1 OR m.has_videos = 1)
    GROUP BY m.conversation_id, month
    ORDER BY m.conversation_id, month
  `).all();

  writeDashData('attachmentTimeSeries.json', attachmentData);
  progressReporter.success(`Attachment time series saved (${attachmentData.length} records)`);
}
 