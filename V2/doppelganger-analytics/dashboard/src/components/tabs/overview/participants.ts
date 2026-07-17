// Pure participant aggregation for the Overview tab, scoped to the selected
// conversation(s). Message counts always come from per-conversation
// messages_by_sender — never global totals divided by a filtered size.

import { computeEmojiChampions } from '@/lib/overviewAggregate';
import { filterMessagesBySender, isKnownParticipant, buildParticipantIndex } from '@/lib/participantFilter';
import type { EmojiMetricsData, EngagementData, RawConversationData, RawMediaData, TurnTakingData } from './types';

export interface ParticipantAnalyticsData {
  messageContributors: Array<{ participant: string; total_messages: number }>;
  emojiUsers: Array<{ sender: string; count: number }>;
  mediaSharers: Array<{ sender: string; mediaShared: { photos: number; videos: number; attachments: number; total: number } }>;
  fastResponders: Array<{ participant: string; avg_response_time: number }>;
}

export function computeParticipantAnalytics(
  conversationData: RawConversationData | null,
  mediaData: RawMediaData | null,
  engagementData: EngagementData | null,
  turnTakingData: TurnTakingData | null,
  emojiData: EmojiMetricsData | null,
  selectedConversations: string[],
  isFiltered: boolean
): ParticipantAnalyticsData {
  const empty: ParticipantAnalyticsData = {
    messageContributors: [],
    emojiUsers: [],
    mediaSharers: [],
    fastResponders: []
  };

  if (!conversationData) return empty;

  const selectedIds = isFiltered && selectedConversations.length > 0
    ? selectedConversations
    : conversationData.conversations.map(c => c.conversation_id);

  const selectedConvs = conversationData.conversations.filter(c =>
    selectedIds.includes(c.conversation_id)
  );

  const participantIndex = buildParticipantIndex(selectedConvs);

  // Aggregate per-sender message counts from selected conversations only.
  const messagesBySender = new Map<string, number>();
  for (const conv of selectedConvs) {
    const bySender = filterMessagesBySender(conv.messages_by_sender, conv.participants);
    if (bySender && Object.keys(bySender).length > 0) {
      for (const [sender, count] of Object.entries(bySender)) {
        messagesBySender.set(sender, (messagesBySender.get(sender) || 0) + count);
      }
    } else if (!isFiltered && engagementData?.participant_scores) {
      // Legacy fallback only when viewing ALL conversations and
      // messages_by_sender is absent (pre-regeneration data).
      for (const p of engagementData.participant_scores) {
        messagesBySender.set(p.participant, p.total_messages);
      }
      break;
    }
  }

  const messageContributors = Array.from(messagesBySender.entries())
    .map(([participant, total_messages]) => ({ participant, total_messages }))
    .sort((a, b) => b.total_messages - a.total_messages);

  // Fast responders: prefer turn-taking per selected conversation; else
  // engagement scores for participants present in the selection.
  let fastResponders: ParticipantAnalyticsData['fastResponders'] = [];
  if (turnTakingData?.conversation_patterns) {
    const responseByParticipant = new Map<string, { sum: number; n: number }>();
    for (const pattern of turnTakingData.conversation_patterns) {
      if (!selectedIds.includes(pattern.conversation_id)) continue;
      for (const p of pattern.participants) {
        if (!isKnownParticipant(participantIndex, pattern.conversation_id, p.participant)) continue;
        if (p.avg_response_time <= 0) continue;
        const entry = responseByParticipant.get(p.participant) || { sum: 0, n: 0 };
        entry.sum += p.avg_response_time;
        entry.n += 1;
        responseByParticipant.set(p.participant, entry);
      }
    }
    fastResponders = Array.from(responseByParticipant.entries())
      .map(([participant, { sum, n }]) => ({ participant, avg_response_time: sum / n }))
      .sort((a, b) => a.avg_response_time - b.avg_response_time);
  } else if (engagementData?.participant_scores) {
    const present = new Set(messagesBySender.keys());
    fastResponders = engagementData.participant_scores
      .filter(p => present.has(p.participant) && p.avg_response_time > 0)
      .sort((a, b) => a.avg_response_time - b.avg_response_time)
      .map(p => ({ participant: p.participant, avg_response_time: p.avg_response_time }));
  }

  // Media sharers from per-conversation sender_media_data.
  let mediaSharers: ParticipantAnalyticsData['mediaSharers'] = [];
  if (mediaData?.sender_media_data) {
    const mediaBySender = new Map<string, { photos: number; videos: number; attachments: number; total: number }>();
    for (const row of mediaData.sender_media_data) {
      if (!selectedIds.includes(row.conversation_id)) continue;
      if (!isKnownParticipant(participantIndex, row.conversation_id, row.sender)) continue;
      const entry = mediaBySender.get(row.sender) || { photos: 0, videos: 0, attachments: 0, total: 0 };
      entry.photos += row.photo_count;
      entry.videos += row.video_count;
      entry.attachments += row.attachment_count;
      entry.total += row.total_media;
      mediaBySender.set(row.sender, entry);
    }
    mediaSharers = Array.from(mediaBySender.entries())
      .map(([sender, mediaShared]) => ({ sender, mediaShared }))
      .sort((a, b) => b.mediaShared.total - a.mediaShared.total);
  }

  // Emoji champions from honest per-conversation per-sender counts; empty when
  // the emoji dataset is unavailable (pre-regeneration exports).
  const emojiUsers = emojiData
    ? computeEmojiChampions(emojiData.senderEmojis, selectedIds, selectedConvs, 50)
    : [];

  return { messageContributors, emojiUsers, mediaSharers, fastResponders };
}
