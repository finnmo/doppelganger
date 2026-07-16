import { parseIMessageExporterText, IMESSAGE_TIMESTAMP_RE } from '../../src/importers/imessageExporter.js';

const SAMPLE = `May 12, 2020  8:48:51 AM (Read by them after 2 seconds)
Me
Hey there

Dec 03, 2020 11:10:31 AM
Caelan Morris
Yes please 

May 21, 2025  8:42:02 PM (Read by you after 4 seconds)
Dad
Maybe remove all WAPs
Tapbacks:
    Liked by Me

Jun 11, 2020  5:49:10 PM Mum added Caelan Morris to the conversation.

Mar 28, 2024 10:18:20 PM (Read by you after 13 seconds)
Dad
Hi ya Finn are you home at the moment
`;

describe('imessageExporter', () => {
  test('IMESSAGE_TIMESTAMP_RE matches exporter timestamps', () => {
    expect(IMESSAGE_TIMESTAMP_RE.test('May 12, 2020  8:48:51 AM')).toBe(true);
    expect(IMESSAGE_TIMESTAMP_RE.test('Dec 03, 2020 11:10:31 AM')).toBe(true);
    expect(IMESSAGE_TIMESTAMP_RE.test('May 21, 2025  8:42:02 PM (Read by you after 4 seconds)')).toBe(true);
    expect(IMESSAGE_TIMESTAMP_RE.test('[12/31/23, 10:30:45 PM] Alice: hi')).toBe(false);
  });

  test('parseIMessageExporterText extracts messages and skips announcements/tapbacks', () => {
    const messages = parseIMessageExporterText(SAMPLE);
    expect(messages).toHaveLength(4);

    expect(messages[0].sender).toBe('Me');
    expect(messages[0].text).toBe('Hey there');
    expect(messages[0].timestampMs).toBe(new Date(2020, 4, 12, 8, 48, 51).getTime());

    expect(messages[1].sender).toBe('Caelan Morris');
    expect(messages[1].text).toBe('Yes please');

    const dad = messages.find(m => m.sender === 'Dad' && m.text?.includes('WAPs'));
    expect(dad).toBeDefined();
    expect(dad!.text).not.toContain('Liked by');

    expect(messages.some(m => m.text?.includes('added'))).toBe(false);
  });

  test('parses attachment paths as media messages', () => {
    const raw = `Sep 18, 2020  6:14:22 PM
Me
attachments/176/287.HEIC

`;
    const messages = parseIMessageExporterText(raw);
    expect(messages).toHaveLength(1);
    expect(messages[0].attachments).toHaveLength(1);
    expect(messages[0].attachments[0].uri).toContain('attachments/');
  });
});
