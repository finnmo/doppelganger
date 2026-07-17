import {
  computeEmojiChampions,
  computePeaksFromActiveHours,
  computeFilteredOverviewMetrics,
  type ActiveHourRecord,
  type SenderEmojiRecord,
  type OverviewInputs
} from '../dashboard/src/lib/overviewAggregate';

const activeHours: ActiveHourRecord[] = [
  { conversation_id: 'c1', hour: 21, day: 'Friday', day_of_week: 5, sender: 'A', count: 50 },
  { conversation_id: 'c1', hour: 22, day: 'Friday', day_of_week: 5, sender: 'A', count: 30 },
  { conversation_id: 'c1', hour: 9, day: 'Monday', day_of_week: 1, sender: 'B', count: 10 },
  { conversation_id: 'c2', hour: 9, day: 'Monday', day_of_week: 1, sender: 'C', count: 500 }
];

const emoji: SenderEmojiRecord[] = [
  { conversation_id: 'c1', sender: 'A', emoji_count: 120 },
  { conversation_id: 'c1', sender: 'B', emoji_count: 40 },
  { conversation_id: 'c2', sender: 'C', emoji_count: 999 }
];

describe('computePeaksFromActiveHours', () => {
  test('peaks come only from the given rows', () => {
    const peaks = computePeaksFromActiveHours(activeHours.filter(r => r.conversation_id === 'c1'));
    expect(peaks.peak_hours[0]).toBe(21);
    expect(peaks.peak_days[0]).toBe('Friday');
    // c2's dominant Monday 9am volume must not leak in
    expect(peaks.peak_hours[0]).not.toBe(9);
  });

  test('empty input yields empty peaks (honest empty state)', () => {
    expect(computePeaksFromActiveHours([])).toEqual({ peak_hours: [], peak_days: [] });
  });
});

describe('computeEmojiChampions', () => {
  const convs = [
    { conversation_id: 'c1', participants: ['A', 'B'] },
    { conversation_id: 'c2', participants: ['C'] },
  ];

  test('aggregates only selected conversations', () => {
    const champions = computeEmojiChampions(emoji, ['c1'], convs);
    expect(champions).toEqual([
      { sender: 'A', count: 120 },
      { sender: 'B', count: 40 }
    ]);
  });

  test('sums a sender across selected conversations when they are participants', () => {
    const convsWithAInC2 = [
      { conversation_id: 'c1', participants: ['A', 'B'] },
      { conversation_id: 'c2', participants: ['A', 'C'] },
    ];
    const champions = computeEmojiChampions(
      [...emoji, { conversation_id: 'c2', sender: 'A', emoji_count: 5 }],
      ['c1', 'c2'],
      convsWithAInC2
    );
    expect(champions[0]).toEqual({ sender: 'C', count: 999 });
    expect(champions.find(c => c.sender === 'A')).toEqual({ sender: 'A', count: 125 });
  });

  test('drops emoji rows from non-participant senders', () => {
    const rows = [
      { conversation_id: 'c1', sender: 'A', emoji_count: 10 },
      { conversation_id: 'c1', sender: 'Intruder', emoji_count: 999 },
    ];
    expect(computeEmojiChampions(rows, ['c1'], convs)).toEqual([{ sender: 'A', count: 10 }]);
  });
});

describe('computeFilteredOverviewMetrics with emoji + activeHours', () => {
  const inputs: OverviewInputs = {
    text: {
      summary: {
        totalMessages: 1000, totalEmojis: 1159, totalUrls: 10,
        averageWordCount: 5, averageEmojiCount: 1, averageUrlCount: 0
      }
    },
    conversation: {
      summary: {
        totalConversations: 2, totalUniqueParticipants: 3,
        averageTurns: 1, averageDuration: 1, averageParticipants: 2, messagesProcessed: 1000
      },
      conversations: [
        { conversation_id: 'c1', participants: ['A', 'B'], total_messages: 90, turns: 10, duration_ms: 1000, messages_by_sender: { A: 50, B: 40 } },
        { conversation_id: 'c2', participants: ['C'], total_messages: 910, turns: 20, duration_ms: 2000, messages_by_sender: { C: 910 } }
      ]
    },
    media: {
      summary: {
        total_media_messages: 0, total_photos: 0, total_videos: 0, total_attachments: 0,
        media_percentage: 0, top_media_sender: 'A', most_active_month: '2024-01'
      },
      conversation_metrics: []
    },
    time: { peak_hours: [9], peak_days: ['Monday'], activity_patterns: null },
    latency: [],
    emoji,
    activeHours
  };

  test('filtered emoji total comes from selected conversations only', () => {
    const result = computeFilteredOverviewMetrics(inputs, ['c1'], true);
    expect(result.textMetrics.totalEmojis).toBe(160); // 120 + 40, not 1159
  });

  test('filtered peaks come from selected activeHours, not global timeMetrics', () => {
    const result = computeFilteredOverviewMetrics(inputs, ['c1'], true);
    expect(result.timeMetrics.peak_hours[0]).toBe(21);
    expect(result.timeMetrics.peak_days[0]).toBe('Friday');
  });

  test('missing emoji/activeHours data gives 0/empty, never global values', () => {
    const withoutExtras = { ...inputs, emoji: undefined, activeHours: undefined };
    const result = computeFilteredOverviewMetrics(withoutExtras, ['c1'], true);
    expect(result.textMetrics.totalEmojis).toBe(0);
    expect(result.timeMetrics.peak_hours).toEqual([]);
    expect(result.timeMetrics.peak_days).toEqual([]);
  });

  test('unfiltered view returns global summaries unchanged', () => {
    const result = computeFilteredOverviewMetrics(inputs, [], false);
    expect(result.textMetrics.totalEmojis).toBe(1159);
    expect(result.timeMetrics.peak_hours).toEqual([9]);
  });
});
