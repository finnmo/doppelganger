import { getDb, closeDb } from '../db/client.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';

interface SentimentRecord {
  sender: string;
  date: string;
  avgCompound: number;
  avgPositive: number;
  avgNegative: number;
  avgNeutral: number;
  messageCount: number;
  conversationId: string;
}

interface CorrelationPair {
  sender1: string;
  sender2: string;
  correlation: number;
  sharedDays: number;
  totalDays: number;
  strength: 'strong' | 'moderate' | 'weak' | 'none';
}

interface MoodPattern {
  sender: string;
  averageMood: number;
  moodVariability: number;
  positiveStreak: number;
  negativeStreak: number;
  moodTrend: 'improving' | 'declining' | 'stable';
  dominantEmotion: 'positive' | 'negative' | 'neutral';
}

interface MoodCorrelationData {
  summary: {
    totalParticipants: number;
    totalCorrelations: number;
    strongCorrelations: number;
    averageCorrelation: number;
    dateRange: {
      start: string;
      end: string;
    };
  };
  correlationMatrix: CorrelationPair[];
  moodPatterns: MoodPattern[];
  timeSeriesData: Array<{
    date: string;
    participants: Record<string, number>;
  }>;
}

function calculateCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;
  
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
  
  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  
  return denominator === 0 ? 0 : numerator / denominator;
}

function getCorrelationStrength(correlation: number): 'strong' | 'moderate' | 'weak' | 'none' {
  const abs = Math.abs(correlation);
  if (abs >= 0.7) return 'strong';
  if (abs >= 0.4) return 'moderate';
  if (abs >= 0.2) return 'weak';
  return 'none';
}

function calculateMoodTrend(sentiments: number[]): 'improving' | 'declining' | 'stable' {
  if (sentiments.length < 3) return 'stable';
  
  const firstHalf = sentiments.slice(0, Math.floor(sentiments.length / 2));
  const secondHalf = sentiments.slice(Math.floor(sentiments.length / 2));
  
  const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
  const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
  
  const difference = secondAvg - firstAvg;
  
  if (difference > 0.05) return 'improving';
  if (difference < -0.05) return 'declining';
  return 'stable';
}

export async function computeMoodCorrelationMetrics(): Promise<void> {
  progressReporter.start('Computing mood correlation metrics...');
  const db = await getDb();
  
  try {
    // Get daily sentiment data for each sender
    const sentimentData = db.prepare(`
      SELECT 
        m.sender,
        date(m.timestamp_ms / 1000, 'unixepoch', 'localtime') as date,
        m.conversation_id,
        AVG(s.compound) as avgCompound,
        AVG(s.positive) as avgPositive,
        AVG(s.negative) as avgNegative,
        AVG(s.neutral) as avgNeutral,
        COUNT(*) as messageCount
      FROM messages m
      JOIN sentiment s ON m.id = s.message_id
      WHERE s.compound IS NOT NULL
      GROUP BY m.sender, date, m.conversation_id
      HAVING messageCount >= 3
      ORDER BY date, m.sender
    `).all() as SentimentRecord[];

    progressReporter.update(`Processing ${sentimentData.length.toLocaleString()} daily sentiment records...`);

    // Group data by date for correlation analysis
    const dateGroups = new Map<string, Map<string, number>>();
    const senderData = new Map<string, number[]>();
    const senderDates = new Map<string, string[]>();

    for (const record of sentimentData) {
      // Group by date
      if (!dateGroups.has(record.date)) {
        dateGroups.set(record.date, new Map());
      }
      dateGroups.get(record.date)!.set(record.sender, record.avgCompound);

      // Group by sender for pattern analysis
      if (!senderData.has(record.sender)) {
        senderData.set(record.sender, []);
        senderDates.set(record.sender, []);
      }
      senderData.get(record.sender)!.push(record.avgCompound);
      senderDates.get(record.sender)!.push(record.date);
    }

    // Calculate correlations between all sender pairs
    const correlations: CorrelationPair[] = [];
    const senders = Array.from(senderData.keys());

    for (let i = 0; i < senders.length; i++) {
      for (let j = i + 1; j < senders.length; j++) {
        const sender1 = senders[i];
        const sender2 = senders[j];
        
        // Find shared dates
        const dates1 = new Set(senderDates.get(sender1) || []);
        const dates2 = new Set(senderDates.get(sender2) || []);
        const sharedDates = Array.from(dates1).filter(date => dates2.has(date));
        
        if (sharedDates.length >= 5) { // Need at least 5 shared days for meaningful correlation
          // Get sentiment values for shared dates
          const values1: number[] = [];
          const values2: number[] = [];
          
          for (const date of sharedDates) {
            const dayData = dateGroups.get(date);
            if (dayData && dayData.has(sender1) && dayData.has(sender2)) {
              values1.push(dayData.get(sender1)!);
              values2.push(dayData.get(sender2)!);
            }
          }
          
          if (values1.length >= 5) {
            const correlation = calculateCorrelation(values1, values2);
            
            correlations.push({
              sender1,
              sender2,
              correlation,
              sharedDays: values1.length,
              totalDays: Math.max(dates1.size, dates2.size),
              strength: getCorrelationStrength(correlation)
            });
          }
        }
      }
    }

    // Sort correlations by absolute value
    correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

    // Calculate mood patterns for each sender
    const moodPatterns: MoodPattern[] = [];
    
    for (const [sender, sentiments] of senderData.entries()) {
      if (sentiments.length >= 5) {
        const averageMood = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
        const variance = sentiments.reduce((sum, val) => sum + Math.pow(val - averageMood, 2), 0) / sentiments.length;
        const moodVariability = Math.sqrt(variance);
        
        // Calculate streaks
        let positiveStreak = 0;
        let negativeStreak = 0;
        let currentPositiveStreak = 0;
        let currentNegativeStreak = 0;
        
        for (const sentiment of sentiments) {
          if (sentiment > 0.1) {
            currentPositiveStreak++;
            currentNegativeStreak = 0;
            positiveStreak = Math.max(positiveStreak, currentPositiveStreak);
          } else if (sentiment < -0.1) {
            currentNegativeStreak++;
            currentPositiveStreak = 0;
            negativeStreak = Math.max(negativeStreak, currentNegativeStreak);
          } else {
            currentPositiveStreak = 0;
            currentNegativeStreak = 0;
          }
        }
        
        const moodTrend = calculateMoodTrend(sentiments);
        const dominantEmotion = averageMood > 0.05 ? 'positive' : averageMood < -0.05 ? 'negative' : 'neutral';
        
        moodPatterns.push({
          sender,
          averageMood: Math.round(averageMood * 1000) / 1000,
          moodVariability: Math.round(moodVariability * 1000) / 1000,
          positiveStreak,
          negativeStreak,
          moodTrend,
          dominantEmotion
        });
      }
    }

    // Sort mood patterns by average mood
    moodPatterns.sort((a, b) => b.averageMood - a.averageMood);

    // Create time series data for visualization
    const timeSeriesData = Array.from(dateGroups.entries())
      .map(([date, senderMoods]) => ({
        date,
        participants: Object.fromEntries(senderMoods.entries())
      }))
      .sort((a, b) => a.date.localeCompare(b.date));

    // Calculate summary statistics
    const strongCorrelations = correlations.filter(c => c.strength === 'strong').length;
    const averageCorrelation = correlations.length > 0 
      ? correlations.reduce((sum, c) => sum + Math.abs(c.correlation), 0) / correlations.length 
      : 0;

    const dates = Array.from(dateGroups.keys()).sort();
    const dateRange = {
      start: dates[0] || 'N/A',
      end: dates[dates.length - 1] || 'N/A'
    };

    // Export data

    const moodCorrelationData: MoodCorrelationData = {
      summary: {
        totalParticipants: senders.length,
        totalCorrelations: correlations.length,
        strongCorrelations,
        averageCorrelation: Math.round(averageCorrelation * 1000) / 1000,
        dateRange
      },
      correlationMatrix: correlations.slice(0, 50), // Top 50 correlations
      moodPatterns: moodPatterns.slice(0, 20), // Top 20 most active participants
      timeSeriesData: timeSeriesData.slice(0, 365) // Last year of data for performance
    };

    writeDashData('moodCorrelationMetrics.json', moodCorrelationData);

    progressReporter.success('Mood correlation metrics computed and exported.');
    progressReporter.update(`Analyzed ${senders.length} participants`);
    progressReporter.update(`Found ${correlations.length} correlations`);
    progressReporter.update(`${strongCorrelations} strong correlations detected`);
    progressReporter.update(`Average correlation strength: ${averageCorrelation.toFixed(3)}`);
    
  } catch (error) {
    console.error(chalk.red('❌ Error computing mood correlation metrics:'), error);
    throw error;
  } finally {
    closeDb(db);
  }
} 