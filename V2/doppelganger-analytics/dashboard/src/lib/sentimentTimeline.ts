// Builds a daily sentiment timeline for a conversation selection from
// per-conversation daily rows (sentimentDailyByConversation.json). Rows store
// sums so merging conversations stays exact: merged average = Σsums / Σcounts.

export interface SentimentDailyRow {
  conversation_id: string;
  date: string;
  compoundSum: number;
  positiveSum: number;
  negativeSum: number;
  neutralSum: number;
  messageCount: number;
}

export interface SentimentDailyBySenderRow extends SentimentDailyRow {
  sender: string;
}

export interface SentimentTimePoint {
  date: string;
  timestamp: number;
  avgCompound: number;
  avgPositive: number;
  avgNegative: number;
  avgNeutral: number;
  messageCount: number;
  sentiment: 'positive' | 'negative' | 'neutral';
}

const round3 = (n: number) => Math.round(n * 1000) / 1000;

export function buildTimelineFromDailyRows(
  rows: SentimentDailyRow[],
  selectedConversations: string[]
): SentimentTimePoint[] {
  const selected = new Set(selectedConversations);

  const byDate = new Map<string, {
    compoundSum: number;
    positiveSum: number;
    negativeSum: number;
    neutralSum: number;
    count: number;
  }>();

  for (const row of rows) {
    if (!selected.has(row.conversation_id)) continue;
    let day = byDate.get(row.date);
    if (!day) {
      day = { compoundSum: 0, positiveSum: 0, negativeSum: 0, neutralSum: 0, count: 0 };
      byDate.set(row.date, day);
    }
    day.compoundSum += row.compoundSum;
    day.positiveSum += row.positiveSum;
    day.negativeSum += row.negativeSum;
    day.neutralSum += row.neutralSum;
    day.count += row.messageCount;
  }

  const timeline: SentimentTimePoint[] = [];
  for (const [date, day] of byDate.entries()) {
    if (day.count === 0) continue;
    const avgCompound = day.compoundSum / day.count;
    timeline.push({
      date,
      timestamp: new Date(date).getTime(),
      avgCompound: round3(avgCompound),
      avgPositive: round3(day.positiveSum / day.count),
      avgNegative: round3(day.negativeSum / day.count),
      avgNeutral: round3(day.neutralSum / day.count),
      messageCount: day.count,
      sentiment: avgCompound > 0.1 ? 'positive' : avgCompound < -0.1 ? 'negative' : 'neutral'
    });
  }

  timeline.sort((a, b) => a.timestamp - b.timestamp);
  return timeline;
}

export function buildSenderTimelineFromDailyRows(
  rows: SentimentDailyBySenderRow[],
  selectedConversations: string[],
  sender: string
): SentimentTimePoint[] {
  const selected = new Set(selectedConversations);

  const byDate = new Map<string, {
    compoundSum: number;
    positiveSum: number;
    negativeSum: number;
    neutralSum: number;
    count: number;
  }>();

  for (const row of rows) {
    if (row.sender !== sender) continue;
    if (selected.size > 0 && !selected.has(row.conversation_id)) continue;
    let day = byDate.get(row.date);
    if (!day) {
      day = { compoundSum: 0, positiveSum: 0, negativeSum: 0, neutralSum: 0, count: 0 };
      byDate.set(row.date, day);
    }
    day.compoundSum += row.compoundSum;
    day.positiveSum += row.positiveSum;
    day.negativeSum += row.negativeSum;
    day.neutralSum += row.neutralSum;
    day.count += row.messageCount;
  }

  const timeline: SentimentTimePoint[] = [];
  for (const [date, day] of byDate.entries()) {
    if (day.count === 0) continue;
    const avgCompound = day.compoundSum / day.count;
    timeline.push({
      date,
      timestamp: new Date(date).getTime(),
      avgCompound: round3(avgCompound),
      avgPositive: round3(day.positiveSum / day.count),
      avgNegative: round3(day.negativeSum / day.count),
      avgNeutral: round3(day.neutralSum / day.count),
      messageCount: day.count,
      sentiment: avgCompound > 0.1 ? 'positive' : avgCompound < -0.1 ? 'negative' : 'neutral'
    });
  }

  timeline.sort((a, b) => a.timestamp - b.timestamp);
  return timeline;
}

export function formatFullDate(dateStr: string): string {
  if (!dateStr || dateStr === 'N/A') return dateStr;
  const d = new Date(dateStr + 'T12:00:00');
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export interface TimelineSummary {
  totalDays: number;
  totalMessages: number;
  dateRange: { start: string; end: string };
  avgDailySentiment: number;
  mostPositiveDay: SentimentTimePoint | null;
  mostNegativeDay: SentimentTimePoint | null;
}

export function summarizeTimeline(timeline: SentimentTimePoint[]): TimelineSummary {
  if (timeline.length === 0) {
    return {
      totalDays: 0,
      totalMessages: 0,
      dateRange: { start: 'N/A', end: 'N/A' },
      avgDailySentiment: 0,
      mostPositiveDay: null,
      mostNegativeDay: null
    };
  }

  return {
    totalDays: timeline.length,
    totalMessages: timeline.reduce((sum, day) => sum + day.messageCount, 0),
    dateRange: { start: timeline[0].date, end: timeline[timeline.length - 1].date },
    avgDailySentiment: round3(
      timeline.reduce((sum, day) => sum + day.avgCompound, 0) / timeline.length
    ),
    mostPositiveDay: timeline.reduce((max, day) => (day.avgCompound > max.avgCompound ? day : max), timeline[0]),
    mostNegativeDay: timeline.reduce((min, day) => (day.avgCompound < min.avgCompound ? day : min), timeline[0])
  };
}
