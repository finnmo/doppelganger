import { buildFilteredMoodData } from '../dashboard/src/lib/moodCorrelation.js';

const conversations = [
  { conversation_id: 'c1', participants: ['Alice', 'Bob'] },
  { conversation_id: 'c2', participants: ['Alice', 'Carol'] },
];

describe('buildFilteredMoodData participant scoping', () => {
  test('excludes senders who are not conversation participants', () => {
    const sentiment = [
      { sender: 'Alice', conversation_id: 'c1', avg_sentiment: 0.5, message_count: 10, avg_positive: 0.6, avg_negative: 0.1, avg_neutral: 0.3 },
      { sender: 'Eve', conversation_id: 'c1', avg_sentiment: -0.9, message_count: 5, avg_positive: 0.1, avg_negative: 0.8, avg_neutral: 0.1 },
      { sender: 'Bob', conversation_id: 'c1', avg_sentiment: 0.2, message_count: 8, avg_positive: 0.4, avg_negative: 0.2, avg_neutral: 0.4 },
    ];
    const daily = [
      { sender: 'Alice', conversation_id: 'c1', date: '2024-01-01', compoundSum: 0.5, positiveSum: 0.3, negativeSum: 0.1, neutralSum: 0.2, messageCount: 1 },
      { sender: 'Alice', conversation_id: 'c1', date: '2024-01-02', compoundSum: 0.6, positiveSum: 0.3, negativeSum: 0.1, neutralSum: 0.2, messageCount: 1 },
      { sender: 'Alice', conversation_id: 'c1', date: '2024-01-03', compoundSum: 0.4, positiveSum: 0.3, negativeSum: 0.1, neutralSum: 0.2, messageCount: 1 },
      { sender: 'Bob', conversation_id: 'c1', date: '2024-01-01', compoundSum: 0.2, positiveSum: 0.2, negativeSum: 0.1, neutralSum: 0.2, messageCount: 1 },
      { sender: 'Bob', conversation_id: 'c1', date: '2024-01-02', compoundSum: 0.3, positiveSum: 0.2, negativeSum: 0.1, neutralSum: 0.2, messageCount: 1 },
      { sender: 'Bob', conversation_id: 'c1', date: '2024-01-03', compoundSum: 0.1, positiveSum: 0.2, negativeSum: 0.1, neutralSum: 0.2, messageCount: 1 },
      { sender: 'Eve', conversation_id: 'c1', date: '2024-01-01', compoundSum: -0.9, positiveSum: 0.1, negativeSum: 0.8, neutralSum: 0.1, messageCount: 1 },
    ];

    const result = buildFilteredMoodData(sentiment, daily, ['c1'], conversations);
    expect(result.moodPatterns.map((p) => p.sender).sort()).toEqual(['Alice', 'Bob']);
    expect(result.summary.totalParticipants).toBe(2);
  });
});
