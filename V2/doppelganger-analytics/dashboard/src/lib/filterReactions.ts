/** Aggregate reaction metrics for a selected conversation set from
 *  per-conversation processor output (never from global emoji counts). */

import { isKnownParticipant } from './participantFilter';

export interface ConversationReactionRow {
  conversation_id: string;
  count: number;
  top_emoji: string;
  emoji_counts?: Array<{ emoji: string; count: number }>;
  senders?: Array<{
    sender: string;
    reactionsGiven: number;
    reactionsReceived: number;
    topEmojisGiven: Array<{ emoji: string; count: number }>;
    topEmojisReceived: Array<{ emoji: string; count: number }>;
    reactionRatio: number;
  }>;
}

export interface FilteredReactionView {
  summary: {
    totalReactions: number;
    uniqueEmojis: number;
    totalMessages: number;
    reactionRate: number;
    topEmoji: string;
  };
  reactionSummaries: Array<{
    emoji: string;
    count: number;
    percentage: number;
    conversation_ids: string[];
    topReactors: Array<{ sender: string; count: number }>;
    topTargets: Array<{ sender: string; count: number }>;
  }>;
  senderStats: Array<{
    sender: string;
    conversation_ids: string[];
    reactionsGiven: number;
    reactionsReceived: number;
    topEmojisGiven: Array<{ emoji: string; count: number }>;
    topEmojisReceived: Array<{ emoji: string; count: number }>;
    reactionRatio: number;
  }>;
}

export function aggregateReactionsForConversations(
  rows: ConversationReactionRow[],
  selectedIds: string[],
  messageCountInSelection: number,
  participantIndex?: Map<string, Set<string>>
): FilteredReactionView {
  const selected = new Set(selectedIds);
  const convs = rows.filter(r => selected.has(r.conversation_id));

  const emojiTotals = new Map<string, number>();
  const senderAgg = new Map<string, {
    given: number;
    received: number;
    conversationIds: Set<string>;
    emojisGiven: Map<string, number>;
    emojisReceived: Map<string, number>;
  }>();

  for (const conv of convs) {
    for (const { emoji, count } of conv.emoji_counts || []) {
      emojiTotals.set(emoji, (emojiTotals.get(emoji) || 0) + count);
    }
    for (const s of conv.senders || []) {
      if (
        participantIndex &&
        !isKnownParticipant(participantIndex, conv.conversation_id, s.sender)
      ) {
        continue;
      }
      let agg = senderAgg.get(s.sender);
      if (!agg) {
        agg = {
          given: 0,
          received: 0,
          conversationIds: new Set(),
          emojisGiven: new Map(),
          emojisReceived: new Map()
        };
        senderAgg.set(s.sender, agg);
      }
      agg.given += s.reactionsGiven;
      agg.received += s.reactionsReceived;
      agg.conversationIds.add(conv.conversation_id);
      for (const e of s.topEmojisGiven) {
        agg.emojisGiven.set(e.emoji, (agg.emojisGiven.get(e.emoji) || 0) + e.count);
      }
      for (const e of s.topEmojisReceived) {
        agg.emojisReceived.set(e.emoji, (agg.emojisReceived.get(e.emoji) || 0) + e.count);
      }
    }
  }

  const totalReactions = convs.reduce((sum, c) => sum + c.count, 0);

  const reactionSummaries = Array.from(emojiTotals.entries())
    .map(([emoji, count]) => ({
      emoji,
      count,
      percentage: totalReactions > 0 ? (count / totalReactions) * 100 : 0,
      conversation_ids: selectedIds,
      topReactors: [] as Array<{ sender: string; count: number }>,
      topTargets: [] as Array<{ sender: string; count: number }>
    }))
    .sort((a, b) => b.count - a.count);

  // Fill top reactors/targets from aggregated sender emoji maps (approximate
  // for multi-conv selections; exact for a single conversation).
  for (const summary of reactionSummaries) {
    const reactors: Array<{ sender: string; count: number }> = [];
    const targets: Array<{ sender: string; count: number }> = [];
    for (const [sender, agg] of senderAgg.entries()) {
      const given = agg.emojisGiven.get(summary.emoji) || 0;
      const received = agg.emojisReceived.get(summary.emoji) || 0;
      if (given > 0) reactors.push({ sender, count: given });
      if (received > 0) targets.push({ sender, count: received });
    }
    summary.topReactors = reactors.sort((a, b) => b.count - a.count).slice(0, 5);
    summary.topTargets = targets.sort((a, b) => b.count - a.count).slice(0, 5);
  }

  const senderStats = Array.from(senderAgg.entries())
    .map(([sender, agg]) => ({
      sender,
      conversation_ids: Array.from(agg.conversationIds),
      reactionsGiven: agg.given,
      reactionsReceived: agg.received,
      topEmojisGiven: Array.from(agg.emojisGiven.entries())
        .map(([emoji, count]) => ({ emoji, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5),
      topEmojisReceived: Array.from(agg.emojisReceived.entries())
        .map(([emoji, count]) => ({ emoji, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5),
      reactionRatio: agg.received > 0 ? agg.given / agg.received : agg.given
    }))
    .sort((a, b) => (b.reactionsGiven + b.reactionsReceived) - (a.reactionsGiven + a.reactionsReceived));

  return {
    summary: {
      totalReactions,
      uniqueEmojis: reactionSummaries.length,
      totalMessages: messageCountInSelection,
      reactionRate: messageCountInSelection > 0
        ? (totalReactions / messageCountInSelection) * 100
        : 0,
      topEmoji: reactionSummaries[0]?.emoji || 'N/A'
    },
    reactionSummaries,
    senderStats
  };
}
