import {
  tokenizeParticipantName,
  buildConversationNameBlocklist,
} from '../../src/utils/messageFilters.js';
import { reservoirSampleWordExamples } from '../../src/utils/wordExamples.js';

describe('participant name blocklist', () => {
  test('tokenizeParticipantName splits display names', () => {
    expect(tokenizeParticipantName('Tia Shannon')).toEqual(['tia', 'shannon']);
    expect(tokenizeParticipantName('Finn')).toEqual(['finn']);
    expect(tokenizeParticipantName('A')).toEqual([]);
  });

  test('buildConversationNameBlocklist is per-conversation', () => {
    const map = buildConversationNameBlocklist([
      { sender: 'Tia Shannon', conversation_id: 'c1' },
      { sender: 'Finn', conversation_id: 'c2' },
    ]);
    expect(map.get('c1')?.has('tia')).toBe(true);
    expect(map.get('c1')?.has('finn')).toBe(false);
    expect(map.get('c2')?.has('finn')).toBe(true);
  });
});

describe('reservoirSampleWordExamples', () => {
  test('samples up to 3 examples per top word', () => {
    const messages = [
      { content: 'hello world again', sender: 'A', conversation_id: 'c1', words: ['hello', 'world'] },
      { content: 'hello there friend', sender: 'B', conversation_id: 'c1', words: ['hello', 'friend'] },
      { content: 'world of wonder', sender: 'A', conversation_id: 'c1', words: ['world', 'wonder'] },
      { content: 'hello once more', sender: 'C', conversation_id: 'c2', words: ['hello'] },
    ];
    const result = reservoirSampleWordExamples(messages, ['hello', 'world', 'missing']);
    const hello = result.find((e) => e.word === 'hello');
    expect(hello?.examples.length).toBeGreaterThan(0);
    expect(hello?.examples.length).toBeLessThanOrEqual(3);
    expect(result.find((e) => e.word === 'missing')).toBeUndefined();
  });
});
