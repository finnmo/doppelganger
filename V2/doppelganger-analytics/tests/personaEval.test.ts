import { describe, test, expect } from '@jest/globals';
import { lengthRatio, tokenJaccard } from '../src/processors/personaEval.js';

describe('personaEval scoring helpers', () => {
  test('tokenJaccard is 1 for identical text', () => {
    expect(tokenJaccard('melbourne was so fun', 'melbourne was so fun')).toBe(1);
  });

  test('tokenJaccard is low for unrelated text', () => {
    expect(tokenJaccard('coffee later?', 'report is done')).toBeLessThan(0.2);
  });

  test('lengthRatio is 1 for same word counts', () => {
    expect(lengthRatio('one two three', 'aaa bbb ccc')).toBe(1);
  });
});
