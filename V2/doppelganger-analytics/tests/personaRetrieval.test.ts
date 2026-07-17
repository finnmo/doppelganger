import { describe, test, expect } from '@jest/globals';
import {
  tokenizeForRetrieval,
  messagesToPersonaThread,
} from '../dashboard/src/lib/server/personaRetrieval.js';
import { buildAnthropicPersonaRequest } from '../dashboard/src/lib/server/buildPersonaPrompt.js';

describe('personaRetrieval helpers', () => {
  test('tokenizeForRetrieval drops stopwords and short tokens', () => {
    const tokens = tokenizeForRetrieval('Yeah I went to Melbourne with Sam for coffee');
    expect(tokens).toContain('melbourne');
    expect(tokens).toContain('sam');
    expect(tokens).toContain('coffee');
    expect(tokens).not.toContain('yeah');
    expect(tokens).not.toContain('the');
  });

  test('messagesToPersonaThread maps persona to assistant', () => {
    const turns = messagesToPersonaThread(
      [
        {
          id: 1,
          conversation_id: 'c1',
          sender: 'Finn',
          content: 'you free friday?',
          timestamp_ms: 1,
        },
        {
          id: 2,
          conversation_id: 'c1',
          sender: 'Tia',
          content: 'yeah after 6',
          timestamp_ms: 2,
        },
      ],
      'Tia'
    );
    expect(turns[0].role).toBe('user');
    expect(turns[1].role).toBe('assistant');
    expect(turns[1].content).toBe('yeah after 6');
  });
});

describe('buildAnthropicPersonaRequest with memories', () => {
  test('injects memory block into system prompt', () => {
    const { system, memoryCount } = buildAnthropicPersonaRequest(
      {
        sender: 'Tia',
        messageCount: 10,
        vocabulary: { topWords: [], avgWordsPerMessage: 4, avgEmojiPerMessage: 0 },
        sentiment: { avgCompound: 0, positiveRatio: 0, negativeRatio: 0 },
        responsiveness: { medianReplyMs: null, p90ReplyMs: null, label: 'unknown' },
        starters: { conversationStarts: 0, startRatio: 0 },
        styleSummary: 'Tia is brief.',
        fewShotExamples: [],
        sources: ['instagram'],
      },
      [{ role: 'user', content: 'remember melbourne?' }],
      {
        memories: [
          {
            text: '[2023-01-01] Finn: trip? → Tia: melbourne was so fun',
            score: 2,
            conversationId: 'c1',
            timestampMs: 1,
          },
        ],
      }
    );

    expect(memoryCount).toBe(1);
    expect(system).toContain('Relevant memories from their real past messages');
    expect(system).toContain('melbourne was so fun');
    // Base style guidance still surrounds the injected memory block.
    expect(system).toContain('natural engaged reply');
  });
});
