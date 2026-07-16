// Shared data shapes for the Overview tab, matching the generated JSON files.

export interface ReplyLatencyItem {
  conversation_id: string;
  bucket: string;
  count: number;
}

export interface RawTextData {
  summary: {
    totalMessages: number;
    averageWordCount: number;
    averageEmojiCount: number;
    averageUrlCount: number;
    totalEmojis: number;
    totalUrls: number;
  };
}

export interface RawConversationData {
  summary: {
    totalConversations: number;
    totalUniqueParticipants: number;
    averageTurns: number;
    averageDuration: number;
    averageParticipants: number;
    messagesProcessed: number;
  };
  conversations: Array<{
    conversation_id: string;
    source?: string;
    source_label?: string;
    participants: string[];
    first_message_ms: number;
    last_message_ms: number;
    total_messages: number;
    turns: number;
    duration_ms: number;
    avg_response_time: number;
    messages_by_sender?: Record<string, number>;
  }>;
}

export interface RawMediaData {
  summary: {
    total_media_messages: number;
    total_photos: number;
    total_videos: number;
    total_attachments: number;
    media_percentage: number;
    top_media_sender: string;
    most_active_month: string;
  };
  conversation_metrics: Array<{
    conversation_id: string;
    photo_count: number;
    video_count: number;
    attachment_count: number;
    total_messages: number;
    media_ratio: number;
  }>;
  sender_media_data?: Array<{
    conversation_id: string;
    sender: string;
    photo_count: number;
    video_count: number;
    attachment_count: number;
    total_media: number;
  }>;
}

export interface RawTimeData {
  peak_hours: number[];
  peak_days: string[];
  activity_patterns: {
    morning_peak: { hour: number; count: number };
    afternoon_peak: { hour: number; count: number };
    evening_peak: { hour: number; count: number };
  };
}

export interface EngagementData {
  summary: {
    total_participants: number;
    avg_engagement_score: number;
    high_engagement_participants: number;
    most_engaged_participant: string;
    least_engaged_participant: string;
    engagement_variance: number;
  };
  participant_scores: Array<{
    participant: string;
    overall_score: number;
    message_frequency: number;
    response_speed: number;
    conversation_initiation: number;
    message_length: number;
    consistency: number;
    social_connectivity: number;
    conversations_count: number;
    total_messages: number;
    avg_response_time: number;
    engagement_tier: string;
  }>;
}

export interface TurnTakingData {
  summary: {
    total_conversations: number;
    avg_participants: number;
    balanced_conversations: number;
    dominant_speaker_conversations: number;
    avg_turn_length: number;
    avg_response_time: number;
  };
  conversation_patterns: Array<{
    conversation_id: string;
    pattern: {
      pattern_type: string;
      participants: string[];
      turn_ratio: number;
      conversation_health: number;
      description: string;
    };
    participants: Array<{
      conversation_id: string;
      participant: string;
      turn_count: number;
      avg_turn_length: number;
      max_turn_length: number;
      turn_percentage: number;
      avg_response_time: number;
      interruption_rate: number;
    }>;
  }>;
}

export interface EmojiMetricsData {
  summary: {
    totalEmojis: number;
    uniqueEmojis: number;
    totalSenders: number;
    topSender: string | null;
    topEmoji: string | null;
  };
  senderEmojis: Array<{
    conversation_id: string;
    sender: string;
    emoji_count: number;
    top_emojis?: Array<{ emoji: string; count: number }>;
  }>;
}
