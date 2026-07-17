import {
  filterRowsBySelection,
  type ConversationParticipantRecord,
  buildParticipantIndex,
  isKnownParticipant,
} from './participantFilter';

export interface RawSentimentData {
  sender: string;
  conversation_id: string;
  avg_sentiment: number;
  message_count: number;
  avg_positive: number;
  avg_negative: number;
  avg_neutral: number;
}

export interface SentimentDailyBySenderRow {
  conversation_id: string;
  sender: string;
  date: string;
  compoundSum: number;
  positiveSum: number;
  negativeSum: number;
  neutralSum: number;
  messageCount: number;
}

export interface CorrelationPair {
  sender1: string;
  sender2: string;
  correlation: number;
  sharedDays: number;
  totalDays: number;
  strength: 'strong' | 'moderate' | 'weak' | 'none';
}

export interface MoodPattern {
  sender: string;
  averageMood: number;
  moodVariability: number;
  positiveStreak: number;
  negativeStreak: number;
  moodTrend: 'improving' | 'declining' | 'stable';
  dominantEmotion: 'positive' | 'negative' | 'neutral';
}

export interface MoodCorrelationData {
  summary: {
    totalParticipants: number;
    totalCorrelations: number;
    strongCorrelations: number;
    averageCorrelation: number;
    dateRange: { start: string; end: string };
  };
  correlationMatrix: CorrelationPair[];
  moodPatterns: MoodPattern[];
  timeSeriesData: Array<{
    date: string;
    participants: Record<string, number>;
  }>;
}

const calculateVariability = (values: number[]): number => {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
};

const getCorrelationStrength = (correlation: number): 'strong' | 'moderate' | 'weak' | 'none' => {
  const abs = Math.abs(correlation);
  if (abs >= 0.7) return 'strong';
  if (abs >= 0.4) return 'moderate';
  if (abs >= 0.2) return 'weak';
  return 'none';
};

function pearsonCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length < 2) return 0;
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
  const num = n * sumXY - sumX * sumY;
  const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  return den === 0 ? 0 : num / den;
}

/** Pairs of participants who share at least one selected conversation. */
export function eligibleMoodPairs(
  conversations: ConversationParticipantRecord[],
  selectedConversationIds: string[],
  participantIndex: Map<string, Set<string>>
): Set<string> {
  const selected = new Set(selectedConversationIds);
  const pairs = new Set<string>();
  for (const conv of conversations) {
    if (!selected.has(conv.conversation_id)) continue;
    const parts = [...(participantIndex.get(conv.conversation_id) ?? conv.participants ?? [])];
    for (let i = 0; i < parts.length; i++) {
      for (let j = i + 1; j < parts.length; j++) {
        const key = [parts[i], parts[j]].sort().join('\0');
        pairs.add(key);
      }
    }
  }
  return pairs;
}

export function buildFilteredMoodData(
  sentimentData: RawSentimentData[],
  dailySenderRows: SentimentDailyBySenderRow[],
  selectedConversations: string[],
  conversations: ConversationParticipantRecord[] = []
): MoodCorrelationData {
  const participantIndex = buildParticipantIndex(conversations);
  const scopedSentiment = filterRowsBySelection(
    sentimentData,
    selectedConversations,
    participantIndex,
    { senderKey: 'sender' }
  );
  const scopedDaily = filterRowsBySelection(
    dailySenderRows,
    selectedConversations,
    participantIndex,
    { senderKey: 'sender' }
  );

  const uniqueSenders = [...new Set(scopedSentiment.map((item) => item.sender))];
  const pairKeys = eligibleMoodPairs(conversations, selectedConversations, participantIndex);

  const dateGroups = new Map<string, Map<string, { sum: number; count: number }>>();
  for (const row of scopedDaily) {
    let day = dateGroups.get(row.date);
    if (!day) {
      day = new Map();
      dateGroups.set(row.date, day);
    }
    const existing = day.get(row.sender) ?? { sum: 0, count: 0 };
    existing.sum += row.compoundSum;
    existing.count += row.messageCount;
    day.set(row.sender, existing);
  }

  const senderDaily = new Map<string, Map<string, number>>();
  for (const [date, senders] of dateGroups.entries()) {
    for (const [sender, vals] of senders.entries()) {
      if (!senderDaily.has(sender)) senderDaily.set(sender, new Map());
      senderDaily.get(sender)!.set(date, vals.sum / vals.count);
    }
  }

  const correlationMatrix: CorrelationPair[] = [];
  for (let i = 0; i < uniqueSenders.length; i++) {
    for (let j = i + 1; j < uniqueSenders.length; j++) {
      const s1 = uniqueSenders[i];
      const s2 = uniqueSenders[j];
      const pairKey = [s1, s2].sort().join('\0');
      if (conversations.length > 0 && !pairKeys.has(pairKey)) continue;

      const dates1 = senderDaily.get(s1);
      const dates2 = senderDaily.get(s2);
      if (!dates1 || !dates2) continue;
      const sharedDates = [...dates1.keys()].filter((d) => dates2.has(d)).sort();
      if (sharedDates.length < 3) continue;
      const x = sharedDates.map((d) => dates1.get(d)!);
      const y = sharedDates.map((d) => dates2.get(d)!);
      const correlation = pearsonCorrelation(x, y);
      correlationMatrix.push({
        sender1: s1,
        sender2: s2,
        correlation: Math.round(correlation * 1000) / 1000,
        sharedDays: sharedDates.length,
        totalDays: Math.max(dates1.size, dates2.size),
        strength: getCorrelationStrength(correlation),
      });
    }
  }

  const moodPatterns: MoodPattern[] = uniqueSenders.map((sender) => {
    const senderRows = scopedSentiment.filter((item) => item.sender === sender);
    const totalMessages = senderRows.reduce((sum, item) => sum + item.message_count, 0) || 1;
    const avgMood =
      senderRows.reduce((sum, item) => sum + item.avg_sentiment * item.message_count, 0) /
      totalMessages;
    const dailyVals = [...(senderDaily.get(sender)?.values() ?? [])];
    const moodVariability = calculateVariability(dailyVals.length > 1 ? dailyVals : [avgMood]);
    return {
      sender,
      averageMood: Math.round(avgMood * 1000) / 1000,
      moodVariability: Math.round(moodVariability * 1000) / 1000,
      positiveStreak: 0,
      negativeStreak: 0,
      moodTrend: avgMood > 0.05 ? 'improving' : avgMood < -0.05 ? 'declining' : 'stable',
      dominantEmotion: avgMood > 0.1 ? 'positive' : avgMood < -0.1 ? 'negative' : 'neutral',
    };
  });

  const timeSeriesData = [...dateGroups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, senders]) => ({
      date,
      participants: Object.fromEntries(
        [...senders.entries()].map(([s, v]) => [s, Math.round((v.sum / v.count) * 1000) / 1000])
      ),
    }));

  const dates = timeSeriesData.map((d) => d.date);
  const strongCorrelations = correlationMatrix.filter((c) => c.strength === 'strong').length;
  const averageCorrelation =
    correlationMatrix.length > 0
      ? correlationMatrix.reduce((sum, c) => sum + Math.abs(c.correlation), 0) /
        correlationMatrix.length
      : 0;

  return {
    summary: {
      totalParticipants: uniqueSenders.length,
      totalCorrelations: correlationMatrix.length,
      strongCorrelations,
      averageCorrelation: Math.round(averageCorrelation * 1000) / 1000,
      dateRange: { start: dates[0] || 'N/A', end: dates[dates.length - 1] || 'N/A' },
    },
    correlationMatrix,
    moodPatterns,
    timeSeriesData,
  };
}

export { isKnownParticipant };
