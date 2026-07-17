import {
  buildParticipantIndex,
  filterMessagesBySender,
  filterRowsBySelection,
  getParticipantUnion,
  isKnownParticipant,
} from '../dashboard/src/lib/participantFilter.js';

const conversations = [
  { conversation_id: 'c1', participants: ['Alice', 'Bob'] },
  { conversation_id: 'c2', participants: ['Alice', 'Carol'] },
];

describe('participantFilter', () => {
  const index = buildParticipantIndex(conversations);

  test('isKnownParticipant is per-conversation', () => {
    expect(isKnownParticipant(index, 'c1', 'Alice')).toBe(true);
    expect(isKnownParticipant(index, 'c1', 'Carol')).toBe(false);
    expect(isKnownParticipant(index, 'c2', 'Carol')).toBe(true);
  });

  test('getParticipantUnion deduplicates across selected conversations', () => {
    const union = getParticipantUnion(conversations, ['c1', 'c2']);
    expect([...union].sort()).toEqual(['Alice', 'Bob', 'Carol']);
  });

  test('filterRowsBySelection requires conversation and participant', () => {
    const rows = [
      { conversation_id: 'c1', sender: 'Alice', count: 1 },
      { conversation_id: 'c1', sender: 'Eve', count: 9 },
      { conversation_id: 'c2', sender: 'Bob', count: 2 },
    ];
    const filtered = filterRowsBySelection(rows, ['c1'], index, { senderKey: 'sender' });
    expect(filtered).toEqual([{ conversation_id: 'c1', sender: 'Alice', count: 1 }]);
  });

  test('filterRowsBySelection without senderKey keeps conversation filter only', () => {
    const rows = [
      { conversation_id: 'c1', bucket: 'short', count: 3 },
      { conversation_id: 'c3', bucket: 'short', count: 1 },
    ];
    expect(filterRowsBySelection(rows, ['c1'], index)).toHaveLength(1);
  });

  test('filterMessagesBySender drops non-participants', () => {
    expect(
      filterMessagesBySender({ Alice: 10, Eve: 99, Bob: 5 }, ['Alice', 'Bob'])
    ).toEqual({ Alice: 10, Bob: 5 });
  });
});
