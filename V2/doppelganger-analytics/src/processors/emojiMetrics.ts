import { getDb, closeDb } from '../db/client.js';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';
import { decodeInstagramUnicode } from '../utils/unicodeDecoder.js';

// Same ranges as extractEmojis in unicodeDecoder.ts, used here to attribute
// counts to individual emojis (extractEmojis only returns totals).
const EMOJI_REGEX = /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu;

interface SenderEmojiRow {
  conversation_id: string;
  sender: string;
  emoji_count: number;
  top_emojis: Array<{ emoji: string; count: number }>;
}

/**
 * Per-conversation, per-sender emoji usage. This is the honest data source for
 * the Overview "Emoji Champions" card and for filtered emoji totals — counts
 * come straight from text_metrics (the same source as textMetrics.json
 * totals), so filtered and global views always add up.
 *
 * Must run after computeTextMetrics (reads the text_metrics table).
 */
export async function computeEmojiMetrics(): Promise<void> {
  progressReporter.start('Computing emoji metrics...');
  const db = await getDb();

  try {
    // Authoritative per-sender counts from text_metrics.
    const countRows = db.prepare(`
      SELECT m.conversation_id, m.sender, SUM(t.emoji_count) AS emoji_count
      FROM text_metrics t
      JOIN messages m ON m.id = t.message_id
      WHERE t.emoji_count > 0
      GROUP BY m.conversation_id, m.sender
    `).all() as Array<{ conversation_id: string; sender: string; emoji_count: number }>;

    // Which emojis each sender uses, from message content.
    const contentRows = db.prepare(`
      SELECT m.conversation_id, m.sender, m.content
      FROM text_metrics t
      JOIN messages m ON m.id = t.message_id
      WHERE t.emoji_count > 0 AND m.content IS NOT NULL
    `).all() as Array<{ conversation_id: string; sender: string; content: string }>;

    const emojiBreakdown = new Map<string, Map<string, number>>();
    const globalEmojiTotals = new Map<string, number>();

    for (const row of contentRows) {
      const decoded = decodeInstagramUnicode(row.content);
      const matches = decoded.match(EMOJI_REGEX);
      if (!matches) continue;

      const key = `${row.conversation_id}\u0000${row.sender}`;
      let perSender = emojiBreakdown.get(key);
      if (!perSender) {
        perSender = new Map();
        emojiBreakdown.set(key, perSender);
      }
      for (const emoji of matches) {
        perSender.set(emoji, (perSender.get(emoji) || 0) + 1);
        globalEmojiTotals.set(emoji, (globalEmojiTotals.get(emoji) || 0) + 1);
      }
    }

    const senderEmojis: SenderEmojiRow[] = countRows.map(row => {
      const breakdown = emojiBreakdown.get(`${row.conversation_id}\u0000${row.sender}`);
      const top_emojis = breakdown
        ? Array.from(breakdown.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5)
            .map(([emoji, count]) => ({ emoji, count }))
        : [];
      return {
        conversation_id: row.conversation_id,
        sender: row.sender,
        emoji_count: row.emoji_count,
        top_emojis
      };
    });

    const totalEmojis = senderEmojis.reduce((sum, r) => sum + r.emoji_count, 0);

    const senderTotals = new Map<string, number>();
    for (const row of senderEmojis) {
      senderTotals.set(row.sender, (senderTotals.get(row.sender) || 0) + row.emoji_count);
    }
    const topSender = Array.from(senderTotals.entries()).sort((a, b) => b[1] - a[1])[0];
    const topEmoji = Array.from(globalEmojiTotals.entries()).sort((a, b) => b[1] - a[1])[0];

    writeDashData('emojiMetrics.json', {
      summary: {
        totalEmojis,
        uniqueEmojis: globalEmojiTotals.size,
        totalSenders: senderTotals.size,
        topSender: topSender ? topSender[0] : null,
        topEmoji: topEmoji ? topEmoji[0] : null
      },
      senderEmojis
    });

    progressReporter.success(
      `Emoji metrics computed: ${totalEmojis.toLocaleString()} emojis across ${senderTotals.size} senders`
    );
  } catch (error) {
    progressReporter.error('Error computing emoji metrics');
    console.error(error);
    writeDashData('emojiMetrics.json', {
      summary: { totalEmojis: 0, uniqueEmojis: 0, totalSenders: 0, topSender: null, topEmoji: null },
      senderEmojis: []
    });
    throw error;
  } finally {
    await closeDb(db);
  }
}
