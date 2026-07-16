import {
  aggregateReactionsForConversations,
  type ConversationReactionRow
} from '../dashboard/src/lib/filterReactions';

const rows: ConversationReactionRow[] = [
  {
    conversation_id: 'c1',
    count: 3,
    top_emoji: '❤️',
    emoji_counts: [
      { emoji: '❤️', count: 2 },
      { emoji: '😂', count: 1 }
    ],
    senders: [
      {
        sender: 'B',
        reactionsGiven: 2,
        reactionsReceived: 1,
        topEmojisGiven: [{ emoji: '❤️', count: 2 }],
        topEmojisReceived: [{ emoji: '😂', count: 1 }],
        reactionRatio: 2
      },
      {
        sender: 'A',
        reactionsGiven: 1,
        reactionsReceived: 2,
        topEmojisGiven: [{ emoji: '😂', count: 1 }],
        topEmojisReceived: [{ emoji: '❤️', count: 2 }],
        reactionRatio: 0.5
      }
    ]
  },
  {
    conversation_id: 'c2',
    count: 10,
    top_emoji: '👍',
    emoji_counts: [{ emoji: '👍', count: 10 }],
    senders: [
      {
        sender: 'X',
        reactionsGiven: 10,
        reactionsReceived: 0,
        topEmojisGiven: [{ emoji: '👍', count: 10 }],
        topEmojisReceived: [],
        reactionRatio: 10
      }
    ]
  }
];

describe('aggregateReactionsForConversations', () => {
  test('scopes emoji counts and sender stats to the selected conversation', () => {
    const result = aggregateReactionsForConversations(rows, ['c1'], 16);

    expect(result.summary.totalReactions).toBe(3);
    expect(result.summary.totalMessages).toBe(16);
    expect(result.summary.reactionRate).toBeCloseTo((3 / 16) * 100, 5);
    expect(result.summary.topEmoji).toBe('❤️');

    expect(result.reactionSummaries).toHaveLength(2);
    expect(result.reactionSummaries.find(r => r.emoji === '❤️')?.count).toBe(2);
    expect(result.reactionSummaries.find(r => r.emoji === '👍')).toBeUndefined();

    expect(result.senderStats.map(s => s.sender).sort()).toEqual(['A', 'B']);
    expect(result.senderStats.find(s => s.sender === 'X')).toBeUndefined();
  });

  test('sums across multiple selected conversations', () => {
    const result = aggregateReactionsForConversations(rows, ['c1', 'c2'], 100);
    expect(result.summary.totalReactions).toBe(13);
    expect(result.reactionSummaries.find((r: { emoji: string }) => r.emoji === '👍')?.count).toBe(10);
  });
});
