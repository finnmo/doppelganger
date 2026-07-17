import { getDb, closeDb } from '../db/client.js';
import chalk from 'chalk';
import { progressReporter } from '../utils/progressReporter.js';
import { writeDashData } from '../utils/output.js';
import { localDateString } from '../utils/dates.js';

interface ReactionRow {
  reaction: string;
  actor: string;
  timestamp: number; // seconds since epoch as exported by Instagram, 0 when missing
  message_id: number;
  target_sender: string;
  conversation_id: string;
  message_timestamp_ms: number;
}

interface ReactionData {
  messageId: number;
  reactorSender: string;
  targetSender: string;
  reactionType: string;
  emoji: string;
  timestamp: number;
  conversationId: string;
}

interface ReactionSummary {
  emoji: string;
  count: number;
  percentage: number;
  conversation_ids: string[];
  topReactors: Array<{
    sender: string;
    count: number;
  }>;
  topTargets: Array<{
    sender: string;
    count: number;
  }>;
}

interface SenderReactionStats {
  sender: string;
  conversation_ids: string[];
  reactionsGiven: number;
  reactionsReceived: number;
  topEmojisGiven: Array<{
    emoji: string;
    count: number;
  }>;
  topEmojisReceived: Array<{
    emoji: string;
    count: number;
  }>;
  reactionRatio: number; // given/received
}

/**
 * Instagram exports the same emoji both with and without the U+FE0F
 * variation selector (e.g. '❤️' and '❤'). Group them under one key so
 * counts aren't split across visually identical reactions.
 */
function emojiGroupKey(reaction: string): string {
  return reaction.replace(/\u{FE0F}/gu, '');
}

function topEntries(map: Map<string, number>, limit: number): Array<{ sender: string; count: number }> {
  return Array.from(map.entries())
    .map(([sender, count]) => ({ sender, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}

function topEmojiEntries(map: Map<string, number>, limit: number): Array<{ emoji: string; count: number }> {
  return Array.from(map.entries())
    .map(([emoji, count]) => ({ emoji, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}

export async function computeReactionMetrics(): Promise<void> {
  progressReporter.start('Computing reaction metrics...');
  const db = await getDb();

  try {
    const reactionRows = db.prepare(`
      SELECT
        r.reaction,
        r.actor,
        r.timestamp,
        r.message_id,
        m.sender AS target_sender,
        m.conversation_id,
        m.timestamp_ms AS message_timestamp_ms
      FROM message_reactions r
      JOIN messages m ON m.id = r.message_id
      ORDER BY m.timestamp_ms
    `).all() as ReactionRow[];

    progressReporter.update(`Processing ${reactionRows.length.toLocaleString()} reactions...`);

    // Per-emoji aggregation
    const emojiStats = new Map<string, {
      displayVariants: Map<string, number>;
      count: number;
      conversationIds: Set<string>;
      reactors: Map<string, number>;
      targets: Map<string, number>;
    }>();

    // Per-sender aggregation
    const senderStats = new Map<string, {
      given: Map<string, number>;
      received: Map<string, number>;
      totalGiven: number;
      totalReceived: number;
      conversationIds: Set<string>;
    }>();

    const getSenderStats = (sender: string) => {
      let stats = senderStats.get(sender);
      if (!stats) {
        stats = {
          given: new Map(),
          received: new Map(),
          totalGiven: 0,
          totalReceived: 0,
          conversationIds: new Set()
        };
        senderStats.set(sender, stats);
      }
      return stats;
    };

    // Per-conversation totals + per-sender breakdowns for honest filtered views.
    const conversationReactions = new Map<string, {
      count: number;
      emojis: Map<string, number>;
      senders: Map<string, {
        given: number;
        received: number;
        emojisGiven: Map<string, number>;
        emojisReceived: Map<string, number>;
      }>;
    }>();

    const getConvSender = (
      conv: { senders: Map<string, { given: number; received: number; emojisGiven: Map<string, number>; emojisReceived: Map<string, number> }> },
      sender: string
    ) => {
      let stats = conv.senders.get(sender);
      if (!stats) {
        stats = { given: 0, received: 0, emojisGiven: new Map(), emojisReceived: new Map() };
        conv.senders.set(sender, stats);
      }
      return stats;
    };

    const rawReactions: ReactionData[] = [];
    let earliestMs = Infinity;
    let latestMs = -Infinity;

    for (const row of reactionRows) {
      const key = emojiGroupKey(row.reaction);

      let convStats = conversationReactions.get(row.conversation_id);
      if (!convStats) {
        convStats = { count: 0, emojis: new Map(), senders: new Map() };
        conversationReactions.set(row.conversation_id, convStats);
      }
      convStats.count++;
      convStats.emojis.set(key, (convStats.emojis.get(key) || 0) + 1);

      const giverInConv = getConvSender(convStats, row.actor);
      giverInConv.given++;
      giverInConv.emojisGiven.set(key, (giverInConv.emojisGiven.get(key) || 0) + 1);

      const receiverInConv = getConvSender(convStats, row.target_sender);
      receiverInConv.received++;
      receiverInConv.emojisReceived.set(key, (receiverInConv.emojisReceived.get(key) || 0) + 1);

      let emoji = emojiStats.get(key);
      if (!emoji) {
        emoji = {
          displayVariants: new Map(),
          count: 0,
          conversationIds: new Set(),
          reactors: new Map(),
          targets: new Map()
        };
        emojiStats.set(key, emoji);
      }
      emoji.displayVariants.set(row.reaction, (emoji.displayVariants.get(row.reaction) || 0) + 1);
      emoji.count++;
      emoji.conversationIds.add(row.conversation_id);
      emoji.reactors.set(row.actor, (emoji.reactors.get(row.actor) || 0) + 1);
      emoji.targets.set(row.target_sender, (emoji.targets.get(row.target_sender) || 0) + 1);

      const giver = getSenderStats(row.actor);
      giver.given.set(key, (giver.given.get(key) || 0) + 1);
      giver.totalGiven++;
      giver.conversationIds.add(row.conversation_id);

      const receiver = getSenderStats(row.target_sender);
      receiver.received.set(key, (receiver.received.get(key) || 0) + 1);
      receiver.totalReceived++;
      receiver.conversationIds.add(row.conversation_id);

      // Reaction timestamps may be seconds or milliseconds depending on platform.
      // Values above ~1e12 are already ms (post-2001). Fall back to message time when missing.
      const timestampMs =
        row.timestamp > 1e12
          ? row.timestamp
          : row.timestamp > 0
            ? row.timestamp * 1000
            : row.message_timestamp_ms;
      earliestMs = Math.min(earliestMs, timestampMs);
      latestMs = Math.max(latestMs, timestampMs);

      if (rawReactions.length < 1000) {
        rawReactions.push({
          messageId: row.message_id,
          reactorSender: row.actor,
          targetSender: row.target_sender,
          reactionType: 'emoji',
          emoji: row.reaction,
          timestamp: timestampMs,
          conversationId: row.conversation_id
        });
      }
    }

    const totalReactions = reactionRows.length;

    // Display each emoji group as its most common raw variant
    const reactionSummaries: ReactionSummary[] = Array.from(emojiStats.values())
      .map(stats => ({
        emoji: topEmojiEntries(stats.displayVariants, 1)[0].emoji,
        count: stats.count,
        percentage: totalReactions > 0 ? (stats.count / totalReactions) * 100 : 0,
        conversation_ids: Array.from(stats.conversationIds),
        topReactors: topEntries(stats.reactors, 5),
        topTargets: topEntries(stats.targets, 5)
      }))
      .sort((a, b) => b.count - a.count);

    const senderReactionStats: SenderReactionStats[] = Array.from(senderStats.entries())
      .map(([sender, stats]) => ({
        sender,
        conversation_ids: Array.from(stats.conversationIds),
        reactionsGiven: stats.totalGiven,
        reactionsReceived: stats.totalReceived,
        topEmojisGiven: topEmojiEntries(stats.given, 5),
        topEmojisReceived: topEmojiEntries(stats.received, 5),
        reactionRatio: stats.totalReceived > 0 ? stats.totalGiven / stats.totalReceived : stats.totalGiven
      }))
      .sort((a, b) => (b.reactionsGiven + b.reactionsReceived) - (a.reactionsGiven + a.reactionsReceived));


    const totalMessages = db.prepare('SELECT COUNT(*) as count FROM messages').get() as { count: number };
    const today = localDateString(Date.now());

    // Display form for each grouped emoji key (its most common raw variant)
    const keyToDisplay = new Map<string, string>();
    for (const [key, stats] of emojiStats.entries()) {
      keyToDisplay.set(key, topEmojiEntries(stats.displayVariants, 1)[0].emoji);
    }

    const reactionsByConversation = Array.from(conversationReactions.entries()).map(([conversation_id, stats]) => {
      const emojiCounts = topEmojiEntries(stats.emojis, 50).map(({ emoji: key, count }) => ({
        emoji: keyToDisplay.get(key) || key,
        count
      }));

      const senders = Array.from(stats.senders.entries()).map(([sender, s]) => ({
        sender,
        reactionsGiven: s.given,
        reactionsReceived: s.received,
        topEmojisGiven: topEmojiEntries(s.emojisGiven, 5).map(({ emoji: key, count }) => ({
          emoji: keyToDisplay.get(key) || key,
          count
        })),
        topEmojisReceived: topEmojiEntries(s.emojisReceived, 5).map(({ emoji: key, count }) => ({
          emoji: keyToDisplay.get(key) || key,
          count
        })),
        reactionRatio: s.received > 0 ? s.given / s.received : s.given
      })).sort((a, b) => (b.reactionsGiven + b.reactionsReceived) - (a.reactionsGiven + a.reactionsReceived));

      return {
        conversation_id,
        count: stats.count,
        top_emoji: emojiCounts[0]?.emoji || 'N/A',
        emoji_counts: emojiCounts,
        senders
      };
    });

    const reactionData = {
      summary: {
        totalReactions,
        uniqueEmojis: reactionSummaries.length,
        totalMessages: totalMessages.count,
        reactionRate: totalMessages.count > 0 ? (totalReactions / totalMessages.count) * 100 : 0,
        topEmoji: reactionSummaries[0]?.emoji || 'N/A',
        dateRange: totalReactions > 0 ? {
          start: localDateString(earliestMs),
          end: localDateString(latestMs)
        } : {
          start: today,
          end: today
        }
      },
      reactionSummaries: reactionSummaries.slice(0, 20), // Top 20 emojis
      senderStats: senderReactionStats.slice(0, 15), // Top 15 most active
      reactionsByConversation, // Per-conversation totals for filtered views
      rawReactions // Sample of raw reactions for detailed analysis
    };

    writeDashData('reactionMetrics.json', reactionData);

    progressReporter.success('Reaction metrics computed and exported.');
    progressReporter.update(`Found ${totalReactions.toLocaleString()} reactions`);
    progressReporter.update(`${reactionSummaries.length} unique emojis used`);
    progressReporter.update(`Reaction rate: ${reactionData.summary.reactionRate.toFixed(2)}%`);
    progressReporter.update(`Top emoji: ${reactionData.summary.topEmoji}`);

  } catch (error) {
    console.error(chalk.red('❌ Error computing reaction metrics:'), error);
    throw error;
  } finally {
    closeDb(db);
  }
}
