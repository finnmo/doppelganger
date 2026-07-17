import { buildParticipantIndex } from '../dashboard/src/lib/participantFilter.js';
import { aggregateReactionsForConversations } from '../dashboard/src/lib/filterReactions.js';

describe('aggregateReactionsForConversations participant scoping', () => {
  const rows = [
    {
      conversation_id: 'c1',
      count: 10,
      top_emoji: '❤️',
      emoji_counts: [{ emoji: '❤️', count: 10 }],
      senders: [
        {
          sender: 'Alice',
          reactionsGiven: 5,
          reactionsReceived: 3,
          topEmojisGiven: [{ emoji: '❤️', count: 5 }],
          topEmojisReceived: [{ emoji: '❤️', count: 3 }],
          reactionRatio: 1.5,
        },
        {
          sender: 'Eve',
          reactionsGiven: 99,
          reactionsReceived: 99,
          topEmojisGiven: [{ emoji: '❤️', count: 99 }],
          topEmojisReceived: [{ emoji: '❤️', count: 99 }],
          reactionRatio: 1,
        },
      ],
    },
  ];

  const participantIndex = buildParticipantIndex([
    { conversation_id: 'c1', participants: ['Alice', 'Bob'] },
  ]);

  test('excludes non-participant senders from senderStats', () => {
    const result = aggregateReactionsForConversations(rows, ['c1'], 100, participantIndex);
    expect(result.senderStats.map((s) => s.sender)).toEqual(['Alice']);
    expect(result.senderStats[0].reactionsGiven).toBe(5);
  });
});
