import Database from 'better-sqlite3';
import { migrate } from '../../src/db/schema.js';
import { conversationParticipantCounts } from '../../src/processors/conversationParticipants.js';
import { buildConversationVoices, buildPlatformVoices } from '../../src/processors/conversationVoice.js';

const MIN = 60 * 1000;

/**
 * `dm` is a 1:1 chat, `group` has 4 people. Both carry enough messages from A
 * to clear MIN_MESSAGES (25) / MIN_PLATFORM_MESSAGES (40).
 */
function seed(db: Database.Database) {
  const insertMsg = db.prepare(`
    INSERT INTO messages (id, conversation_id, sender, timestamp_ms, content, source, is_system,
                          has_photos, has_videos, has_audio, has_share, share_link)
    VALUES (@id, @c, @s, @t, @content, @source, 0, 0, 0, 0, 0, NULL)
  `);

  let nextId = 1;
  const add = (c: string, s: string, source: string) =>
    insertMsg.run({ id: nextId++, c, s, t: nextId * MIN, content: 'hello there', source });

  db.transaction(() => {
    // 1:1 DM on instagram — 30 from A, 30 from B.
    for (let i = 0; i < 30; i++) {
      add('dm', 'A', 'instagram');
      add('dm', 'B', 'instagram');
    }
    // 4-person group on instagram — 30 from A.
    for (let i = 0; i < 30; i++) {
      add('group', 'A', 'instagram');
      add('group', ['B', 'C', 'D'][i % 3], 'instagram');
    }
  })();
}

describe('conversation voice participant counts', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = new Database(':memory:');
    migrate(db);
    seed(db);
  });

  afterAll(() => db.close());

  test('counts distinct non-system senders per conversation', () => {
    const counts = conversationParticipantCounts(db);
    expect(counts.get('dm')).toBe(2);
    expect(counts.get('group')).toBe(4);
  });

  test('caches per connection (same map instance)', () => {
    expect(conversationParticipantCounts(db)).toBe(conversationParticipantCounts(db));
  });

  test('classifies chat type from real participant counts', () => {
    const voices = buildConversationVoices(db, 'A');
    const dm = voices.find((v) => v.conversationId === 'dm');
    const group = voices.find((v) => v.conversationId === 'group');

    expect(dm?.participantCount).toBe(2);
    expect(dm?.chatType).toBe('dm');
    expect(group?.participantCount).toBe(4);
    expect(group?.chatType).toBe('small_group');
  });

  test('dmShare counts only messages in 1:1 chats', () => {
    const [instagram] = buildPlatformVoices(db, 'A');
    // A sent 30 in the DM and 30 in the group → half their messages are 1:1.
    expect(instagram.messageCount).toBe(60);
    expect(instagram.dmShare).toBeCloseTo(0.5, 3);
  });
});
