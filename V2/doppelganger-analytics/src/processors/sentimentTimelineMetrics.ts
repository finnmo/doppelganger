import { getDb, closeDb } from '../db/client.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface SentimentTimePoint {
  date: string;
  timestamp: number;
  avgCompound: number;
  avgPositive: number;
  avgNegative: number;
  avgNeutral: number;
  messageCount: number;
  sentiment: 'positive' | 'negative' | 'neutral';
}

interface SentimentBySender {
  sender: string;
  timeSeries: SentimentTimePoint[];
  overallSentiment: {
    avgCompound: number;
    avgPositive: number;
    avgNegative: number;
    avgNeutral: number;
    totalMessages: number;
  };
}

export async function computeSentimentTimelineMetrics(): Promise<void> {
  progressReporter.start('Computing sentiment timeline metrics...');
  const db = await getDb();
  
  try {
    // Get sentiment data with timestamps
    const sentimentData = db.prepare(`
      SELECT 
        s.compound,
        s.positive,
        s.negative,
        s.neutral,
        m.sender,
        m.timestamp_ms,
        m.conversation_id,
        date(m.timestamp_ms / 1000, 'unixepoch', 'localtime') as date
      FROM sentiment s
      JOIN messages m ON s.message_id = m.id
      WHERE s.compound IS NOT NULL
      ORDER BY m.timestamp_ms
    `).all() as Array<{
      compound: number;
      positive: number;
      negative: number;
      neutral: number;
      sender: string;
      timestamp_ms: number;
      conversation_id: string;
      date: string;
    }>;

    progressReporter.update(`Processing ${sentimentData.length.toLocaleString()} sentiment records for timeline...`);

    // Group by date for overall timeline
    const dateGroups = new Map<string, {
      sentiments: number[];
      positives: number[];
      negatives: number[];
      neutrals: number[];
      count: number;
    }>();

    // Group by sender and date for sender-specific timelines
    const senderDateGroups = new Map<string, Map<string, {
      sentiments: number[];
      positives: number[];
      negatives: number[];
      neutrals: number[];
      count: number;
    }>>();

    // Group by conversation and date so the dashboard can rebuild an honest
    // daily timeline for any conversation selection. Sums (not averages) are
    // stored so multiple conversations can be merged exactly.
    const conversationDateGroups = new Map<string, Map<string, {
      compoundSum: number;
      positiveSum: number;
      negativeSum: number;
      neutralSum: number;
      count: number;
    }>>();

    for (const record of sentimentData) {
      const date = record.date;

      let convGroups = conversationDateGroups.get(record.conversation_id);
      if (!convGroups) {
        convGroups = new Map();
        conversationDateGroups.set(record.conversation_id, convGroups);
      }
      let convDay = convGroups.get(date);
      if (!convDay) {
        convDay = { compoundSum: 0, positiveSum: 0, negativeSum: 0, neutralSum: 0, count: 0 };
        convGroups.set(date, convDay);
      }
      convDay.compoundSum += record.compound;
      convDay.positiveSum += record.positive;
      convDay.negativeSum += record.negative;
      convDay.neutralSum += record.neutral;
      convDay.count++;
      
      // Overall timeline
      if (!dateGroups.has(date)) {
        dateGroups.set(date, {
          sentiments: [],
          positives: [],
          negatives: [],
          neutrals: [],
          count: 0
        });
      }
      
      const dateGroup = dateGroups.get(date)!;
      dateGroup.sentiments.push(record.compound);
      dateGroup.positives.push(record.positive);
      dateGroup.negatives.push(record.negative);
      dateGroup.neutrals.push(record.neutral);
      dateGroup.count++;

      // Sender-specific timeline
      if (!senderDateGroups.has(record.sender)) {
        senderDateGroups.set(record.sender, new Map());
      }
      
      const senderGroups = senderDateGroups.get(record.sender)!;
      if (!senderGroups.has(date)) {
        senderGroups.set(date, {
          sentiments: [],
          positives: [],
          negatives: [],
          neutrals: [],
          count: 0
        });
      }
      
      const senderDateGroup = senderGroups.get(date)!;
      senderDateGroup.sentiments.push(record.compound);
      senderDateGroup.positives.push(record.positive);
      senderDateGroup.negatives.push(record.negative);
      senderDateGroup.neutrals.push(record.neutral);
      senderDateGroup.count++;
    }

    // Process overall timeline
    const overallTimeline: SentimentTimePoint[] = [];
    for (const [date, group] of dateGroups.entries()) {
      const avgCompound = group.sentiments.reduce((a, b) => a + b, 0) / group.sentiments.length;
      const avgPositive = group.positives.reduce((a, b) => a + b, 0) / group.positives.length;
      const avgNegative = group.negatives.reduce((a, b) => a + b, 0) / group.negatives.length;
      const avgNeutral = group.neutrals.reduce((a, b) => a + b, 0) / group.neutrals.length;
      
      let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
      if (avgCompound > 0.1) sentiment = 'positive';
      else if (avgCompound < -0.1) sentiment = 'negative';
      
      overallTimeline.push({
        date,
        timestamp: new Date(date).getTime(),
        avgCompound: Math.round(avgCompound * 1000) / 1000,
        avgPositive: Math.round(avgPositive * 1000) / 1000,
        avgNegative: Math.round(avgNegative * 1000) / 1000,
        avgNeutral: Math.round(avgNeutral * 1000) / 1000,
        messageCount: group.count,
        sentiment
      });
    }

    // Sort by timestamp
    overallTimeline.sort((a, b) => a.timestamp - b.timestamp);

    // Process sender-specific timelines
    const senderTimelines: SentimentBySender[] = [];
    for (const [sender, senderGroups] of senderDateGroups.entries()) {
      const timeSeries: SentimentTimePoint[] = [];
      let totalCompound = 0;
      let totalPositive = 0;
      let totalNegative = 0;
      let totalNeutral = 0;
      let totalMessages = 0;

      for (const [date, group] of senderGroups.entries()) {
        const avgCompound = group.sentiments.reduce((a, b) => a + b, 0) / group.sentiments.length;
        const avgPositive = group.positives.reduce((a, b) => a + b, 0) / group.positives.length;
        const avgNegative = group.negatives.reduce((a, b) => a + b, 0) / group.negatives.length;
        const avgNeutral = group.neutrals.reduce((a, b) => a + b, 0) / group.neutrals.length;
        
        let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
        if (avgCompound > 0.1) sentiment = 'positive';
        else if (avgCompound < -0.1) sentiment = 'negative';
        
        timeSeries.push({
          date,
          timestamp: new Date(date).getTime(),
          avgCompound: Math.round(avgCompound * 1000) / 1000,
          avgPositive: Math.round(avgPositive * 1000) / 1000,
          avgNegative: Math.round(avgNegative * 1000) / 1000,
          avgNeutral: Math.round(avgNeutral * 1000) / 1000,
          messageCount: group.count,
          sentiment
        });

        totalCompound += avgCompound * group.count;
        totalPositive += avgPositive * group.count;
        totalNegative += avgNegative * group.count;
        totalNeutral += avgNeutral * group.count;
        totalMessages += group.count;
      }

      // Sort timeseries by timestamp
      timeSeries.sort((a, b) => a.timestamp - b.timestamp);

      senderTimelines.push({
        sender,
        timeSeries,
        overallSentiment: {
          avgCompound: Math.round((totalCompound / totalMessages) * 1000) / 1000,
          avgPositive: Math.round((totalPositive / totalMessages) * 1000) / 1000,
          avgNegative: Math.round((totalNegative / totalMessages) * 1000) / 1000,
          avgNeutral: Math.round((totalNeutral / totalMessages) * 1000) / 1000,
          totalMessages
        }
      });
    }

    // Sort senders by message count
    senderTimelines.sort((a, b) => b.overallSentiment.totalMessages - a.overallSentiment.totalMessages);

    // Calculate summary statistics
    const summary = {
      totalDays: overallTimeline.length,
      totalMessages: sentimentData.length,
      uniqueSenders: senderTimelines.length,
      dateRange: {
        start: overallTimeline[0]?.date || 'N/A',
        end: overallTimeline[overallTimeline.length - 1]?.date || 'N/A'
      },
      avgDailySentiment: overallTimeline.reduce((sum, day) => sum + day.avgCompound, 0) / overallTimeline.length,
      mostPositiveDay: overallTimeline.reduce((max, day) => day.avgCompound > max.avgCompound ? day : max, overallTimeline[0]),
      mostNegativeDay: overallTimeline.reduce((min, day) => day.avgCompound < min.avgCompound ? day : min, overallTimeline[0])
    };

    // Export data

    const timelineData = {
      summary,
      overallTimeline,
      senderTimelines: senderTimelines.slice(0, 10) // Top 10 most active senders
    };

    writeDashData('sentimentTimelineMetrics.json', timelineData);

    // Per-conversation daily rows (separate file: it is only fetched when a
    // conversation filter is active). Values are rounded sums; the dashboard
    // divides by count after merging selected conversations per day.
    const dailyByConversation: Array<{
      conversation_id: string;
      date: string;
      compoundSum: number;
      positiveSum: number;
      negativeSum: number;
      neutralSum: number;
      messageCount: number;
    }> = [];
    for (const [conversationId, days] of conversationDateGroups.entries()) {
      for (const [date, day] of days.entries()) {
        dailyByConversation.push({
          conversation_id: conversationId,
          date,
          compoundSum: Math.round(day.compoundSum * 1000) / 1000,
          positiveSum: Math.round(day.positiveSum * 1000) / 1000,
          negativeSum: Math.round(day.negativeSum * 1000) / 1000,
          neutralSum: Math.round(day.neutralSum * 1000) / 1000,
          messageCount: day.count
        });
      }
    }
    writeDashData('sentimentDailyByConversation.json', dailyByConversation);

    // Per-sender daily rows for filtered "by sender" timeline rebuild.
    const dailyBySender: Array<{
      sender: string;
      conversation_id: string;
      date: string;
      compoundSum: number;
      positiveSum: number;
      negativeSum: number;
      neutralSum: number;
      messageCount: number;
    }> = [];
    for (const [sender, days] of senderDateGroups.entries()) {
      for (const [date, group] of days.entries()) {
        // Re-aggregate from raw records for conversation_id linkage
        const recordsForDay = sentimentData.filter(
          r => r.sender === sender && r.date === date
        );
        const byConv = new Map<string, typeof group>();
        for (const r of recordsForDay) {
          let g = byConv.get(r.conversation_id);
          if (!g) {
            g = { sentiments: [], positives: [], negatives: [], neutrals: [], count: 0 };
            byConv.set(r.conversation_id, g);
          }
          g.sentiments.push(r.compound);
          g.positives.push(r.positive);
          g.negatives.push(r.negative);
          g.neutrals.push(r.neutral);
          g.count++;
        }
        for (const [conversationId, g] of byConv.entries()) {
          dailyBySender.push({
            sender,
            conversation_id: conversationId,
            date,
            compoundSum: Math.round(g.sentiments.reduce((a, b) => a + b, 0) * 1000) / 1000,
            positiveSum: Math.round(g.positives.reduce((a, b) => a + b, 0) * 1000) / 1000,
            negativeSum: Math.round(g.negatives.reduce((a, b) => a + b, 0) * 1000) / 1000,
            neutralSum: Math.round(g.neutrals.reduce((a, b) => a + b, 0) * 1000) / 1000,
            messageCount: g.count
          });
        }
      }
    }
    writeDashData('sentimentDailyBySender.json', dailyBySender);

    progressReporter.success('Sentiment timeline metrics computed and exported.');
    progressReporter.update(`Timeline spans ${summary.totalDays} days`);
    progressReporter.update(`Average daily sentiment: ${summary.avgDailySentiment.toFixed(3)}`);
    progressReporter.update(`Most positive day: ${summary.mostPositiveDay?.date} (${summary.mostPositiveDay?.avgCompound.toFixed(3)})`);
    progressReporter.update(`Most negative day: ${summary.mostNegativeDay?.date} (${summary.mostNegativeDay?.avgCompound.toFixed(3)})`);
    
  } catch (error) {
    console.error(chalk.red('❌ Error computing sentiment timeline metrics:'), error);
    throw error;
  } finally {
    closeDb(db);
  }
} 