import {
  buildTimelineFromDailyRows,
  summarizeTimeline,
  type SentimentDailyRow
} from '../dashboard/src/lib/sentimentTimeline';

const rows: SentimentDailyRow[] = [
  // c1: two days
  { conversation_id: 'c1', date: '2024-01-01', compoundSum: 1.2, positiveSum: 0.6, negativeSum: 0.1, neutralSum: 1.3, messageCount: 2 },
  { conversation_id: 'c1', date: '2024-01-02', compoundSum: -0.5, positiveSum: 0.1, negativeSum: 0.4, neutralSum: 0.5, messageCount: 1 },
  // c2: overlaps day one
  { conversation_id: 'c2', date: '2024-01-01', compoundSum: 0.8, positiveSum: 0.4, negativeSum: 0.0, neutralSum: 0.6, messageCount: 2 }
];

describe('buildTimelineFromDailyRows', () => {
  test('single conversation: averages are sums divided by counts', () => {
    const timeline = buildTimelineFromDailyRows(rows, ['c1']);
    expect(timeline).toHaveLength(2);
    expect(timeline[0].date).toBe('2024-01-01');
    expect(timeline[0].avgCompound).toBe(0.6); // 1.2 / 2
    expect(timeline[0].messageCount).toBe(2);
    expect(timeline[1].avgCompound).toBe(-0.5);
    expect(timeline[1].sentiment).toBe('negative');
  });

  test('multiple conversations merge exactly on shared days', () => {
    const timeline = buildTimelineFromDailyRows(rows, ['c1', 'c2']);
    const day1 = timeline.find(d => d.date === '2024-01-01')!;
    expect(day1.messageCount).toBe(4);
    expect(day1.avgCompound).toBe(0.5); // (1.2 + 0.8) / 4
  });

  test('unselected conversations are excluded', () => {
    const timeline = buildTimelineFromDailyRows(rows, ['c2']);
    expect(timeline).toHaveLength(1);
    expect(timeline[0].avgCompound).toBe(0.4); // 0.8 / 2
  });

  test('empty selection yields empty timeline', () => {
    expect(buildTimelineFromDailyRows(rows, [])).toHaveLength(0);
  });
});

describe('summarizeTimeline', () => {
  test('summary reflects merged timeline', () => {
    const timeline = buildTimelineFromDailyRows(rows, ['c1', 'c2']);
    const summary = summarizeTimeline(timeline);
    expect(summary.totalDays).toBe(2);
    expect(summary.totalMessages).toBe(5);
    expect(summary.dateRange).toEqual({ start: '2024-01-01', end: '2024-01-02' });
    expect(summary.mostPositiveDay?.date).toBe('2024-01-01');
    expect(summary.mostNegativeDay?.date).toBe('2024-01-02');
  });

  test('empty timeline gives honest empty summary', () => {
    const summary = summarizeTimeline([]);
    expect(summary.totalDays).toBe(0);
    expect(summary.mostPositiveDay).toBeNull();
    expect(summary.dateRange).toEqual({ start: 'N/A', end: 'N/A' });
  });
});
