import { decodeInstagramUnicode, decodeObjectUnicode, hasInstagramEncoding } from '../src/utils/unicodeDecoder.js';

describe('Unicode Decoding', () => {
  describe('decodeInstagramUnicode', () => {
    test('should decode right single quotation mark correctly', () => {
      const input = 'don\\u00e2\\u0080\\u0099t';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('don\u2019t'); // Right single quotation mark (U+2019)
    });

    test('should decode left single quotation mark correctly', () => {
      const input = '\\u00e2\\u0080\\u0098hello\\u00e2\\u0080\\u0099';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('\u2018hello\u2019'); // Left and right single quotation marks
    });

    test('should decode double quotation marks correctly', () => {
      const input = '\\u00e2\\u0080\\u009chello\\u00e2\\u0080\\u009d';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('\u201chello\u201d'); // Left and right double quotation marks
    });

    test('should decode em dash correctly', () => {
      const input = 'Hello\\u00e2\\u0080\\u0094world';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('Hello\u2014world'); // Em dash (U+2014)
    });

    test('should decode en dash correctly', () => {
      const input = '2020\\u00e2\\u0080\\u00932021';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('2020\u20132021'); // En dash (U+2013)
    });

    test('should decode ellipsis correctly', () => {
      const input = 'Wait\\u00e2\\u0080\\u00a6';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('Wait\u2026'); // Ellipsis (U+2026)
    });

    test('should decode emojis correctly', () => {
      const input = 'Hello \\u00f0\\u009f\\u0098\\u0080 World!';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('Hello \ud83d\ude00 World!'); // Grinning face emoji
    });

    test('should decode complex emoji correctly', () => {
      const input = '\\u00f0\\u009f\\u0091\\u008d\\u00f0\\u009f\\u008f\\u00bb';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('\ud83d\udc4d\ud83c\udffb'); // Thumbs up with light skin tone
    });

    test('should handle multiple malformed sequences', () => {
      const input = 'I can\\u00e2\\u0080\\u0099t believe it\\u00e2\\u0080\\u0094amazing!';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('I can\u2019t believe it\u2014amazing!');
    });

    test('should handle mixed normal and malformed text', () => {
      const input = 'Normal text and don\\u00e2\\u0080\\u0099t forget this!';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('Normal text and don\u2019t forget this!');
    });

    test('should handle empty and null inputs', () => {
      expect(decodeInstagramUnicode('')).toBe('');
      expect(decodeInstagramUnicode(null as any)).toBe(null);
      expect(decodeInstagramUnicode(undefined as any)).toBe(undefined);
    });

    test('should handle non-string inputs gracefully', () => {
      expect(decodeInstagramUnicode(123 as any)).toBe(123);
      expect(decodeInstagramUnicode(true as any)).toBe(true);
    });

    test('should handle invalid Unicode sequences gracefully', () => {
      const input = 'Invalid \\u00XX sequence';
      const result = decodeInstagramUnicode(input);
      expect(result).toBe('Invalid \\u00XX sequence'); // Should return original
    });
  });

  describe('decodeObjectUnicode', () => {
    test('should decode strings in objects', () => {
      const input = {
        name: 'John\\u00e2\\u0080\\u0099s Place',
        message: 'Don\\u00e2\\u0080\\u0099t worry!'
      };
      const result = decodeObjectUnicode(input);
      expect(result).toEqual({
        name: 'John\u2019s Place',
        message: 'Don\u2019t worry!'
      });
    });

    test('should decode strings in nested objects', () => {
      const input = {
        user: {
          name: 'Alice\\u00e2\\u0080\\u0099s Account',
          settings: {
            theme: 'dark\\u00e2\\u0080\\u0094mode'
          }
        }
      };
      const result = decodeObjectUnicode(input);
      expect(result).toEqual({
        user: {
          name: 'Alice\u2019s Account',
          settings: {
            theme: 'dark\u2014mode'
          }
        }
      });
    });

    test('should decode strings in arrays', () => {
      const input = [
        'don\\u00e2\\u0080\\u0099t',
        'can\\u00e2\\u0080\\u0099t',
        'won\\u00e2\\u0080\\u0099t'
      ];
      const result = decodeObjectUnicode(input);
      expect(result).toEqual(['don\u2019t', 'can\u2019t', 'won\u2019t']);
    });

    test('should handle mixed data types', () => {
      const input = {
        text: 'don\\u00e2\\u0080\\u0099t',
        number: 42,
        boolean: true,
        nullValue: null,
        array: ['test\\u00e2\\u0080\\u0099s']
      };
      const result = decodeObjectUnicode(input);
      expect(result).toEqual({
        text: 'don\u2019t',
        number: 42,
        boolean: true,
        nullValue: null,
        array: ['test\u2019s']
      });
    });
  });

  describe('hasInstagramEncoding', () => {
    test('should detect Instagram encoding patterns', () => {
      expect(hasInstagramEncoding('don\\u00e2\\u0080\\u0099t')).toBe(true);
      expect(hasInstagramEncoding('\\u00f0\\u009f\\u0098\\u0080')).toBe(true);
      expect(hasInstagramEncoding('Hello\\u00e2\\u0080\\u0094world')).toBe(true);
    });

    test('should not detect normal Unicode', () => {
      expect(hasInstagramEncoding('don\'t')).toBe(false);
      expect(hasInstagramEncoding('Hello world')).toBe(false);
      expect(hasInstagramEncoding('\\u0041')).toBe(false); // Normal \u0041 for 'A'
    });

    test('should handle edge cases', () => {
      expect(hasInstagramEncoding('')).toBe(false);
      expect(hasInstagramEncoding(null as any)).toBe(false);
      expect(hasInstagramEncoding(undefined as any)).toBe(false);
    });
  });

  describe('Real-world Instagram data patterns', () => {
    test('should handle common name patterns', () => {
      const testCases = [
        { input: '\\u00e2\\u009a\\u00a1 Aiden Anderson \\u00e2\\u009a\\u00a1', expected: '\u26a1 Aiden Anderson \u26a1' },
        { input: 'No Jawline \\u00f0\\u009f\\u0092\\u00a9\\u00f0\\u009f\\u0094\\u00a5', expected: 'No Jawline \ud83d\udca9\ud83d\udd25' }
      ];

      testCases.forEach(({ input, expected }) => {
        const result = decodeInstagramUnicode(input);
        expect(result).toBe(expected);
      });
    });

    test('should handle message content patterns', () => {
      const testCases = [
        'I don\\u00e2\\u0080\\u0099t think so',
        'That\\u00e2\\u0080\\u0099s amazing!',
        'Wait\\u00e2\\u0080\\u00a6 what?',
        'Hello\\u00e2\\u0080\\u0094how are you?'
      ];

      testCases.forEach(input => {
        const result = decodeInstagramUnicode(input);
        expect(typeof result).toBe('string');
        expect(result).not.toContain('\\u00'); // Should not contain escape sequences
      });
    });
  });
}); 