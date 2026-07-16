import { isSystemMessage, STOP_WORDS } from '../../src/utils/messageFilters.js';

describe('messageFilters', () => {
  describe('isSystemMessage', () => {
    test('treats empty/short content as system', () => {
      expect(isSystemMessage(null)).toBe(true);
      expect(isSystemMessage(undefined)).toBe(true);
      expect(isSystemMessage('')).toBe(true);
      expect(isSystemMessage('hi')).toBe(true); // < 3 chars
    });

    test('detects Instagram system notifications', () => {
      expect(isSystemMessage('Finn sent an attachment')).toBe(true);
      expect(isSystemMessage('Finn sent a photo')).toBe(true);
      expect(isSystemMessage('Finn sent a video')).toBe(true);
      expect(isSystemMessage('Tia reacted to your message')).toBe(true);
      expect(isSystemMessage('Tia liked a message')).toBe(true);
      expect(isSystemMessage('Finn started a call')).toBe(true);
      expect(isSystemMessage('__system__')).toBe(true);
    });

    test('does NOT flag real messages that merely contain "sent"', () => {
      expect(isSystemMessage('I sent you the file yesterday')).toBe(false);
      expect(isSystemMessage('did you get what I sent')).toBe(false);
    });

    test('keeps normal conversational text', () => {
      expect(isSystemMessage('hey how are you doing today')).toBe(false);
      expect(isSystemMessage('lol that is hilarious')).toBe(false);
    });

    test('honors importer is_system flag over content heuristics', () => {
      expect(isSystemMessage('normal looking text', true)).toBe(true);
      expect(isSystemMessage('normal looking text', 1)).toBe(true);
      expect(isSystemMessage('normal looking text', 0)).toBe(false);
    });
  });

  describe('STOP_WORDS', () => {
    test('contains common stop words and is case-sensitive lowercase', () => {
      expect(STOP_WORDS.has('the')).toBe(true);
      expect(STOP_WORDS.has('and')).toBe(true);
      expect(STOP_WORDS.has('would')).toBe(true);
      expect(STOP_WORDS.has('pizza')).toBe(false);
    });
  });
});
