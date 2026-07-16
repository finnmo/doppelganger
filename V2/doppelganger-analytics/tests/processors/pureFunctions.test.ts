import { classifyMessage } from '../../src/processors/contentTypeMetrics.js';
import { calculateImportanceScore } from '../../src/processors/insightMetrics.js';
import { analyzeSentiment } from '../../src/services/sentimentService.js';

function makeMessage(overrides: Partial<{
  content: string;
  compound: number;
  has_media: number;
  timestamp_ms: number;
}>) {
  return {
    id: 1,
    sender: 'Test',
    conversation_id: 'c1',
    timestamp_ms: new Date(2024, 0, 1, 12, 0).getTime(),
    content: '',
    compound: 0,
    positive: 0,
    negative: 0,
    neutral: 1,
    has_media: 0,
    ...overrides
  };
}

describe('classifyMessage', () => {
  test('classifies media notifications via the media flag', () => {
    expect(classifyMessage('Finn sent 2 photos', true)).toBe('media_notification');
  });
  test('classifies links, calls, and text lengths', () => {
    expect(classifyMessage('check this https://example.com', false)).toBe('link_share');
    expect(classifyMessage('Finn started a call', false)).toBe('call_event');
    expect(classifyMessage('hello', false)).toBe('single_word');
    expect(classifyMessage('how are you doing', false)).toBe('short_text');
    expect(classifyMessage('this is a somewhat longer message with plenty of words in it', false)).toBe('medium_text');
  });
});

describe('calculateImportanceScore', () => {
  test('flags a question', () => {
    const { factors } = calculateImportanceScore(makeMessage({ content: 'what time are we meeting?' }));
    expect(factors).toContain('Question');
  });

  test('flags emphasis on ALL-CAPS (original casing, not lowercased)', () => {
    const { factors } = calculateImportanceScore(makeMessage({ content: 'this is SO BIG news' }));
    expect(factors).toContain('Emphasis');
  });

  test('lowercase text does not trigger emphasis', () => {
    const { factors } = calculateImportanceScore(makeMessage({ content: 'this is so big news' }));
    expect(factors).not.toContain('Emphasis');
  });

  test('is deterministic (no randomness) and floors at 0.1', () => {
    const msg = makeMessage({ content: 'ok' });
    const a = calculateImportanceScore(msg);
    const b = calculateImportanceScore(msg);
    expect(a).toEqual(b);
    expect(a.score).toBeGreaterThanOrEqual(0.1);
  });
});

describe('analyzeSentiment (VADER)', () => {
  test('scores clearly positive text positive', () => {
    expect(analyzeSentiment('I love this, it is amazing!').compound).toBeGreaterThan(0.5);
  });
  test('scores clearly negative text negative', () => {
    expect(analyzeSentiment('this is terrible and I hate it').compound).toBeLessThan(-0.5);
  });
  test('handles negation', () => {
    expect(analyzeSentiment('this is not good at all').compound).toBeLessThan(0);
  });
  test('empty text is neutral', () => {
    expect(analyzeSentiment('')).toEqual({ compound: 0, positive: 0, negative: 0, neutral: 1 });
  });
});
