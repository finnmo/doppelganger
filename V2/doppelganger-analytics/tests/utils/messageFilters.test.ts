import {
  isSystemMessage,
  STOP_WORDS,
  NOTIFICATION_STOP_WORDS,
  tokenizeParticipantName,
  tokenizeConversationLabel,
  buildConversationNameBlocklist,
} from '../../src/utils/messageFilters.js';

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

    test('detects Messenger wording and importer synthetic text', () => {
      expect(isSystemMessage('Ella Williams sent 1 photo')).toBe(true);
      expect(isSystemMessage('Ella Williams sent 3 photos')).toBe(true);
      expect(isSystemMessage('ellu reacted ❤️ to your message')).toBe(true);
      expect(isSystemMessage('Finn sent a voice message')).toBe(true);
      expect(isSystemMessage('Finn shared a link')).toBe(true);
      expect(isSystemMessage('GamePigeon message: Your move.')).toBe(true);
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

  describe('NOTIFICATION_STOP_WORDS', () => {
    test('blocks Messenger notification tokens', () => {
      expect(NOTIFICATION_STOP_WORDS.has('sent')).toBe(true);
      expect(NOTIFICATION_STOP_WORDS.has('photo')).toBe(true);
      expect(NOTIFICATION_STOP_WORDS.has('reacted')).toBe(true);
      expect(NOTIFICATION_STOP_WORDS.has('message')).toBe(true);
      expect(NOTIFICATION_STOP_WORDS.has('dinner')).toBe(false);
    });
  });

  describe('buildConversationNameBlocklist', () => {
    test('includes sender names and conversation folder tokens', () => {
      const map = buildConversationNameBlocklist([
        { sender: 'Ella Williams', conversation_id: 'messenger:ellu_radhu_guys' },
      ]);
      const blocked = map.get('messenger:ellu_radhu_guys');
      expect(blocked?.has('ella')).toBe(true);
      expect(blocked?.has('williams')).toBe(true);
      expect(blocked?.has('ellu')).toBe(true);
      expect(blocked?.has('radhu')).toBe(true);
      expect(blocked?.has('guys')).toBe(true);
    });
  });

  describe('tokenizeConversationLabel', () => {
    test('splits underscore-separated folder names', () => {
      expect(tokenizeConversationLabel('ellu_radhu_guys')).toEqual(['ellu', 'radhu', 'guys']);
    });
  });
});
