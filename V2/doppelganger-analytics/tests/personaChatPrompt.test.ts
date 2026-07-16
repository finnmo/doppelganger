import { describe, test, expect } from '@jest/globals';
import {
  buildAnthropicPersonaRequest,
  scoreExampleSimilarity,
  selectExamples,
} from '../dashboard/src/lib/server/buildPersonaPrompt.js';
import type { PersonaProfile } from '../dashboard/src/lib/server/personaProfiles.js';

function mockProfile(overrides?: Partial<PersonaProfile>): PersonaProfile {
  return {
    sender: 'Alex',
    messageCount: 100,
    vocabulary: { topWords: [], avgWordsPerMessage: 5, avgEmojiPerMessage: 0.2 },
    sentiment: { avgCompound: 0.1, positiveRatio: 0.4, negativeRatio: 0.1 },
    responsiveness: { medianReplyMs: 60000, p90ReplyMs: 120000, label: 'quick' },
    starters: { conversationStarts: 3, startRatio: 0.03 },
    styleSummary: 'Alex is brief.',
    fewShotExamples: [
      {
        context: 'coffee later?',
        reply: 'yeah sure',
        conversationId: 'instagram:chat_a',
        source: 'instagram',
      },
      {
        context: 'you free?',
        reply: 'in 10',
        conversationId: 'whatsapp:chat_b',
        source: 'whatsapp',
      },
    ],
    sources: ['instagram', 'whatsapp'],
    ...overrides,
  };
}

describe('buildAnthropicPersonaRequest', () => {
  test('builds system + few-shot + live user message for Anthropic', () => {
    const { system, messages } = buildAnthropicPersonaRequest(mockProfile(), [
      { role: 'user', content: 'want lunch?' },
    ]);

    expect(system).toContain('Alex');
    expect(system).toContain('Alex is brief.');
    expect(messages[0]).toEqual({ role: 'user', content: 'coffee later?' });
    expect(messages[1]).toEqual({ role: 'assistant', content: 'yeah sure' });
    expect(messages[messages.length - 1]).toEqual({ role: 'user', content: 'want lunch?' });
  });

  test('prefers few-shot from the active conversation when query has no tokens', () => {
    const profile = mockProfile({
      fewShotExamples: Array.from({ length: 10 }, (_, i) => ({
        context: `ctx ${i}`,
        reply: `rep ${i}`,
        conversationId: 'instagram:focus',
        source: 'instagram',
      })),
    });

    const { messages } = buildAnthropicPersonaRequest(
      profile,
      [{ role: 'user', content: 'ok' }],
      { conversationId: 'instagram:focus', maxExamples: 5 }
    );

    // 5 pairs = 10 messages + 1 live user
    expect(messages).toHaveLength(11);
    expect(messages.slice(0, 10).every((m, i) => {
      if (i % 2 === 0) return m.role === 'user' && m.content.startsWith('ctx');
      return m.role === 'assistant' && m.content.startsWith('rep');
    })).toBe(true);
  });

  test('selectExamples ranks by similarity to the current message', () => {
    const profile = mockProfile({
      fewShotExamples: [
        {
          context: 'want coffee later?',
          reply: 'yeah espresso',
          conversationId: 'c1',
          source: 'instagram',
        },
        {
          context: 'did you finish the report?',
          reply: 'almost done',
          conversationId: 'c2',
          source: 'whatsapp',
        },
        {
          context: 'melbourne trip photos?',
          reply: 'sending now',
          conversationId: 'c3',
          source: 'instagram',
        },
      ],
    });

    const picked = selectExamples(profile, {
      maxExamples: 2,
      query: 'coffee this afternoon?',
    });

    expect(picked[0].context).toContain('coffee');
    expect(scoreExampleSimilarity('coffee this afternoon?', picked[0])).toBeGreaterThan(
      scoreExampleSimilarity('coffee this afternoon?', picked[1])
    );
  });

  test('injects relationship card into system prompt', () => {
    const { system } = buildAnthropicPersonaRequest(
      mockProfile({
        relationshipCard: {
          withPerson: 'Finn',
          addressForms: ['babe', 'finn'],
          recurringPeople: ['Sam'],
          recurringPlaces: ['melbourne'],
          toneWithYou: 'usually warm / upbeat with you',
          sharedMessageCount: 100,
          sharedConversationCount: 1,
          openers: ['hey', 'hi'],
          closers: ['night'],
          questionBackRate: 0.3,
          avgWordsWithYou: 8,
          avgWordsGlobal: 6,
          teasingSamples: ['miss you idiot'],
          sharedConversationIds: ['c1'],
          summary:
            'Relationship with Finn: often addresses them as "babe", "finn". Tone: usually warm / upbeat with you.',
          registerSummary:
            'How Alex texts Finn specifically: Common openers with them: "hey". Ask questions back ~30% of the time.',
        },
      }),
      [{ role: 'user', content: 'hey' }]
    );

    expect(system).toContain('Register with this person');
    expect(system).toContain('How Alex texts Finn');
    expect(system).toContain('babe');
  });

  test('prefers with-you few-shot examples over unrelated chats', () => {
    const profile = mockProfile({
      fewShotExamples: [
        {
          context: 'did you finish the report?',
          reply: 'almost done',
          conversationId: 'work',
          source: 'whatsapp',
        },
      ],
      withYouFewShotExamples: [
        {
          context: 'coffee this afternoon?',
          reply: 'yeah babe espresso sounds perfect miss you',
          conversationId: 'dm_you',
          source: 'instagram',
          withYou: true,
        },
      ],
      relationshipCard: {
        withPerson: 'Finn',
        addressForms: ['babe'],
        recurringPeople: [],
        recurringPlaces: [],
        toneWithYou: 'warm',
        sharedMessageCount: 10,
        sharedConversationCount: 1,
        sharedConversationIds: ['dm_you'],
        summary: 'Relationship with Finn.',
        registerSummary: 'How Alex texts Finn specifically.',
      },
    });

    const picked = selectExamples(profile, {
      maxExamples: 1,
      query: 'want coffee later?',
    });
    expect(picked[0].withYou).toBe(true);
    expect(picked[0].reply).toContain('espresso');
  });

  test('injects per-conversation voice when conversationId matches', () => {
    const { system } = buildAnthropicPersonaRequest(
      mockProfile({
        conversationVoices: [
          {
            conversationId: 'instagram:group_chat',
            source: 'instagram',
            participantCount: 6,
            messageCount: 200,
            avgWordsPerMessage: 14,
            avgEmojiPerMessage: 0.6,
            lengthLabel: 'medium',
            chatType: 'group',
            styleSummary:
              'In this larger group chat (6 people), Alex tends to send medium messages (~14 words) and uses emoji often here.',
          },
        ],
      }),
      [{ role: 'user', content: 'lol' }],
      { conversationId: 'instagram:group_chat' }
    );

    expect(system).toContain('Voice for this specific chat');
    expect(system).toContain('larger group chat');
  });

  test('rejects history that does not end with user', () => {
    expect(() =>
      buildAnthropicPersonaRequest(mockProfile({ fewShotExamples: [] }), [
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'yo' },
      ])
    ).toThrow(/Last message must be from the user/);
  });

  test('describes multi-bubble as conditional, not a forced default', () => {
    const profile = mockProfile({
      bubbleHabits: {
        avgBubblesPerTurn: 2.8,
        medianBubblesWhenMulti: 2,
        multiBubbleRate: 0.65,
        sampleTurns: [['yeah that sounds good', 'what time were you thinking?']],
        contextualSamples: [
          {
            context: 'friday brunch?',
            bubbles: ['yeah that sounds good', 'what time were you thinking?'],
          },
        ],
        styleSummary:
          'Alex splits into multiple texts on ~65% of turns. Use <<<BUBBLE>>> only when this reply naturally has separate chunks. Never pad with extra bubbles just to hit a rate.',
      },
      fewShotExamples: [
        {
          context: 'want coffee later?',
          reply: 'sure',
          conversationId: 'c1',
          source: 'instagram',
        },
      ],
    });

    const { system } = buildAnthropicPersonaRequest(profile, [
      { role: 'user', content: 'ok' },
    ]);

    expect(system).toContain('conditional');
    expect(system).not.toContain('DEFAULT FORMAT');
    expect(system).not.toContain('CRITICAL');
    expect(system).toContain('Never pad');
    expect(system).toContain('friday brunch');
  });
});
