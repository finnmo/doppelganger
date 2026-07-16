import { buildPersonaPrompt } from '../src/persona/buildPrompt.js';
import type { PersonaStyleProfile } from '../src/processors/personaMetrics.js';
import { namespacedConversationId, parseConversationId, sourceLabel } from '../src/utils/platformSource.js';

describe('platformSource', () => {
  test('namespaces and parses conversation ids', () => {
    expect(namespacedConversationId('instagram', 'alice_123')).toBe('instagram:alice_123');
    expect(namespacedConversationId('whatsapp', 'instagram:already')).toBe('instagram:already');
    expect(parseConversationId('imessage:Sam')).toEqual({ source: 'imessage', rawId: 'Sam' });
    expect(sourceLabel('whatsapp')).toBe('WhatsApp');
  });
});

describe('buildPersonaPrompt', () => {
  test('assembles system + few-shot messages for LLM', () => {
    const profile: PersonaStyleProfile = {
      sender: 'Alex',
      messageCount: 100,
      vocabulary: { topWords: [{ word: 'yeah', count: 10 }], avgWordsPerMessage: 4, avgEmojiPerMessage: 0.2 },
      sentiment: { avgCompound: 0.2, positiveRatio: 0.5, negativeRatio: 0.1 },
      responsiveness: { medianReplyMs: 60000, p90ReplyMs: 300000, label: 'quick (under 5m)' },
      starters: { conversationStarts: 5, startRatio: 0.05 },
      styleSummary: 'Alex sends short messages.',
      fewShotExamples: [],
      sources: ['instagram']
    };

    const bundle = buildPersonaPrompt(profile, [
      { context: 'want coffee?', reply: 'yeah down', conversationId: 'instagram:x', source: 'instagram' }
    ]);

    expect(bundle.messages[0].role).toBe('system');
    expect(bundle.messages[0].content).toContain('Alex');
    expect(bundle.messages[1]).toEqual({ role: 'user', content: 'want coffee?' });
    expect(bundle.messages[2]).toEqual({ role: 'assistant', content: 'yeah down' });
  });
});
