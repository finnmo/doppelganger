import { findSessionStarters } from '../../src/processors/conversationStarterAnalysisMetrics.js';

const GAP = 4 * 60 * 60 * 1000;

describe('findSessionStarters', () => {
  test('treats the first message as a starter', () => {
    const messages = [
      { id: 1, conversation_id: 'c1', sender: 'A', timestamp_ms: 1000, content: 'hi' },
      { id: 2, conversation_id: 'c1', sender: 'B', timestamp_ms: 2000, content: 'hey' }
    ];
    const starters = findSessionStarters(messages);
    expect(starters).toHaveLength(1);
    expect(starters[0].sender).toBe('A');
  });

  test('starts a new session after a 4h quiet gap', () => {
    const t0 = Date.UTC(2024, 0, 1, 12, 0, 0);
    const messages = [
      { id: 1, conversation_id: 'c1', sender: 'A', timestamp_ms: t0, content: 'hi' },
      { id: 2, conversation_id: 'c1', sender: 'B', timestamp_ms: t0 + 60_000, content: 'hey' },
      { id: 3, conversation_id: 'c1', sender: 'B', timestamp_ms: t0 + 60_000 + GAP + 1, content: 'you there?' },
      { id: 4, conversation_id: 'c1', sender: 'A', timestamp_ms: t0 + 60_000 + GAP + 60_000, content: 'yep' }
    ];
    const starters = findSessionStarters(messages);
    expect(starters).toHaveLength(2);
    expect(starters[1].content).toBe('you there?');
  });

  test('does not split sessions for gaps under 4h', () => {
    const t0 = Date.UTC(2024, 0, 1, 12, 0, 0);
    const messages = [
      { id: 1, conversation_id: 'c1', sender: 'A', timestamp_ms: t0, content: 'hi' },
      { id: 2, conversation_id: 'c1', sender: 'B', timestamp_ms: t0 + GAP - 1, content: 'still here' }
    ];
    expect(findSessionStarters(messages)).toHaveLength(1);
  });
});
