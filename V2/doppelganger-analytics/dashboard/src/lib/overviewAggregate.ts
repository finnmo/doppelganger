// Pure aggregation for the Overview tab. Extracted from the component so it
// can be unit-tested and reused without React. No DOM or fetch dependencies.

export interface ConversationRecord {
  conversation_id: string;
  participants: string[];
  total_messages: number;
  turns: number;
  duration_ms: number;
  /** Per-sender message counts within this conversation. Required for accurate filtered contribution %. */
  messages_by_sender?: Record<string, number>;
}

export interface MediaConversationRecord {
  conversation_id: string;
  photo_count: number;
  video_count: number;
  attachment_count: number;
  total_messages: number;
}

export interface LatencyItem {
  conversation_id: string;
  bucket: string;
  count: number;
}

/** Row from emojiMetrics.json senderEmojis. */
export interface SenderEmojiRecord {
  conversation_id: string;
  sender: string;
  emoji_count: number;
  top_emojis?: Array<{ emoji: string; count: number }>;
}

/** Row from activeHours.json. */
export interface ActiveHourRecord {
  conversation_id: string;
  hour: number | string;
  day?: string;
  day_of_week?: number;
  sender?: string;
  count: number;
}

export interface RawTextData {
  summary: {
    totalMessages: number;
    totalEmojis: number;
    totalUrls: number;
    averageWordCount: number;
    averageEmojiCount: number;
    averageUrlCount: number;
  };
}

export interface RawConversationData {
  conversations: ConversationRecord[];
  summary: {
    totalConversations: number;
    totalUniqueParticipants: number;
    averageTurns: number;
    averageDuration: number;
    averageParticipants: number;
    messagesProcessed: number;
  };
}

export interface RawMediaData {
  conversation_metrics: MediaConversationRecord[];
  summary: {
    total_media_messages: number;
    total_photos: number;
    total_videos: number;
    total_attachments: number;
    media_percentage: number;
    top_media_sender: string;
    most_active_month: string;
  };
}

export interface RawTimeData {
  peak_hours: number[];
  peak_days: string[];
  activity_patterns: unknown;
}

export interface FilteredOverviewMetrics {
  textMetrics: RawTextData['summary'];
  conversationSummary: RawConversationData['summary'];
  mediaMetrics: RawMediaData['summary'];
  timeMetrics: Pick<RawTimeData, 'peak_hours' | 'peak_days' | 'activity_patterns'>;
  filteredLatency: LatencyItem[];
}

export interface OverviewInputs {
  text: RawTextData;
  conversation: RawConversationData;
  media: RawMediaData;
  time: RawTimeData;
  latency: LatencyItem[];
  /** Per-conversation per-sender emoji counts (emojiMetrics.json). Optional for pre-regeneration data. */
  emoji?: SenderEmojiRecord[];
  /** Per-conversation hourly activity (activeHours.json). Optional for pre-regeneration data. */
  activeHours?: ActiveHourRecord[];
}

const DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

function resolveDay(row: ActiveHourRecord): string | null {
  if (typeof row.day === 'string') {
    const normalized = row.day.charAt(0).toUpperCase() + row.day.slice(1).toLowerCase();
    if (DAY_NAMES.includes(normalized)) return normalized;
  }
  if (typeof row.day_of_week === 'number' && row.day_of_week >= 0 && row.day_of_week <= 6) {
    return DAY_NAMES[row.day_of_week];
  }
  return null;
}

/**
 * Peak hours and days computed from the given activeHours rows (already
 * filtered to the selection). Returns empty arrays when no data — the UI must
 * show an honest empty state rather than global peaks.
 */
export function computePeaksFromActiveHours(
  rows: ActiveHourRecord[]
): { peak_hours: number[]; peak_days: string[] } {
  const hourTotals = new Map<number, number>();
  const dayTotals = new Map<string, number>();

  for (const row of rows) {
    const hour = typeof row.hour === 'number' ? row.hour : parseInt(String(row.hour), 10);
    if (!Number.isNaN(hour)) {
      hourTotals.set(hour, (hourTotals.get(hour) || 0) + row.count);
    }
    const day = resolveDay(row);
    if (day) {
      dayTotals.set(day, (dayTotals.get(day) || 0) + row.count);
    }
  }

  const topN = <K,>(totals: Map<K, number>, n: number): K[] =>
    Array.from(totals.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([key]) => key);

  return {
    peak_hours: topN(hourTotals, 3),
    peak_days: topN(dayTotals, 3)
  };
}

/**
 * Computes the Overview tab's headline metrics for the current conversation
 * selection. When nothing is filtered, returns the precomputed global
 * summaries; otherwise recomputes from the per-conversation records.
 *
 * When filtered: emoji totals come from per-conversation emojiMetrics data and
 * peak hours/days from per-conversation activeHours data. If either dataset is
 * missing (pre-regeneration exports), the value is 0/empty — never a global
 * number presented as filtered. URL totals have no per-conversation source yet
 * and stay at 0 when filtered.
 */
export function computeFilteredOverviewMetrics(
  inputs: OverviewInputs,
  selectedConversations: string[],
  isFiltered: boolean
): FilteredOverviewMetrics {
  const { text, conversation, media, time, latency, emoji, activeHours } = inputs;

  if (!isFiltered || selectedConversations.length === 0) {
    return {
      textMetrics: text.summary,
      conversationSummary: conversation.summary,
      mediaMetrics: media.summary,
      timeMetrics: time,
      filteredLatency: latency
    };
  }

  const selected = new Set(selectedConversations);

  const filteredConversations = conversation.conversations.filter(conv => selected.has(conv.conversation_id));

  const uniqueParticipants = new Set<string>();
  filteredConversations.forEach(conv => conv.participants.forEach(p => uniqueParticipants.add(p)));

  const totalMessages = filteredConversations.reduce((sum, conv) => sum + conv.total_messages, 0);
  const count = filteredConversations.length;
  const avg = (fn: (c: ConversationRecord) => number) =>
    count > 0 ? filteredConversations.reduce((sum, conv) => sum + fn(conv), 0) / count : 0;

  const conversationSummary = {
    totalConversations: count,
    totalUniqueParticipants: uniqueParticipants.size,
    averageTurns: avg(c => c.turns),
    averageDuration: avg(c => c.duration_ms),
    averageParticipants: avg(c => c.participants.length),
    messagesProcessed: totalMessages
  };

  const filteredMedia = media.conversation_metrics.filter(conv => selected.has(conv.conversation_id));
  const totalPhotos = filteredMedia.reduce((sum, conv) => sum + conv.photo_count, 0);
  const totalVideos = filteredMedia.reduce((sum, conv) => sum + conv.video_count, 0);
  const totalAttachments = filteredMedia.reduce((sum, conv) => sum + conv.attachment_count, 0);
  const totalMediaMessages = filteredMedia.reduce(
    (sum, conv) => sum + Math.min(conv.photo_count + conv.video_count + conv.attachment_count, conv.total_messages),
    0
  );

  const mediaMetrics = {
    total_media_messages: totalMediaMessages,
    total_photos: totalPhotos,
    total_videos: totalVideos,
    total_attachments: totalAttachments,
    media_percentage: totalMessages > 0 ? (totalMediaMessages / totalMessages) * 100 : 0,
    top_media_sender: media.summary.top_media_sender,
    most_active_month: media.summary.most_active_month
  };

  // Emoji totals from per-conversation data; 0 when the dataset is missing.
  // URL totals have no per-conversation source and stay 0 (never scaled from
  // global counts). Averages stay as corpus averages, honestly labeled.
  const totalEmojis = (emoji ?? [])
    .filter(row => selected.has(row.conversation_id))
    .reduce((sum, row) => sum + row.emoji_count, 0);

  const textMetrics = {
    totalMessages,
    totalEmojis,
    totalUrls: 0,
    averageWordCount: text.summary.averageWordCount,
    averageEmojiCount: text.summary.averageEmojiCount,
    averageUrlCount: text.summary.averageUrlCount
  };

  // Peak hours/days recomputed from the selection's activeHours rows; empty
  // when that dataset is missing (never global peaks presented as filtered).
  const filteredPeaks = computePeaksFromActiveHours(
    (activeHours ?? []).filter(row => selected.has(row.conversation_id))
  );

  return {
    textMetrics,
    conversationSummary,
    mediaMetrics,
    timeMetrics: {
      peak_hours: filteredPeaks.peak_hours,
      peak_days: filteredPeaks.peak_days,
      activity_patterns: time.activity_patterns
    },
    filteredLatency: latency.filter(item => selected.has(item.conversation_id))
  };
}

/**
 * Top emoji users for the current selection, aggregated from per-conversation
 * per-sender rows. Pass all conversation ids when unfiltered.
 */
export function computeEmojiChampions(
  emoji: SenderEmojiRecord[],
  selectedConversations: string[],
  limit = 5
): Array<{ sender: string; count: number }> {
  const selected = new Set(selectedConversations);
  const bySender = new Map<string, number>();
  for (const row of emoji) {
    if (!selected.has(row.conversation_id)) continue;
    bySender.set(row.sender, (bySender.get(row.sender) || 0) + row.emoji_count);
  }
  return Array.from(bySender.entries())
    .map(([sender, count]) => ({ sender, count }))
    .filter(entry => entry.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}
