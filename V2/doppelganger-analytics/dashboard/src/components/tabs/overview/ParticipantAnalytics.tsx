'use client';

import React from 'react';
import { Camera, Heart, MessageSquare, Timer } from 'lucide-react';
import { ChartCard } from '@/components/ui/ChartCard';
import { RankedListPanel, type RankedItem } from '@/components/ui/RankedListPanel';
import { CHART_MD, GRID_GAP } from '@/lib/layout';
import type { ParticipantAnalyticsData } from './participants';

interface ParticipantAnalyticsProps {
  participants: ParticipantAnalyticsData;
  totalMessages: number;
}

function formatResponseTime(ms: number): string {
  const seconds = ms / 1000;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

function contributorItems(
  participants: ParticipantAnalyticsData['messageContributors'],
  totalMessages: number
): RankedItem[] {
  return participants.map((c) => ({
    id: c.participant,
    name: c.participant,
    value: c.total_messages.toLocaleString(),
    numericValue: c.total_messages,
    sub: totalMessages > 0 ? `${Math.round((c.total_messages / totalMessages) * 100)}%` : '—',
  }));
}

function responderItems(participants: ParticipantAnalyticsData['fastResponders']): RankedItem[] {
  return participants.map((r) => ({
    id: r.participant,
    name: r.participant,
    value: formatResponseTime(r.avg_response_time),
    numericValue: r.avg_response_time > 0 ? 1 / r.avg_response_time : 0,
    sub: 'avg response',
  }));
}

function emojiItems(participants: ParticipantAnalyticsData['emojiUsers']): RankedItem[] {
  return participants.map((u) => ({
    id: u.sender,
    name: u.sender,
    value: u.count.toLocaleString(),
    numericValue: u.count,
    sub: 'emojis used',
  }));
}

function mediaItems(participants: ParticipantAnalyticsData['mediaSharers']): RankedItem[] {
  return participants.map((s) => ({
    id: s.sender,
    name: s.sender,
    value: s.mediaShared.total.toLocaleString(),
    numericValue: s.mediaShared.total,
    sub: (
      <span className="flex space-x-1">
        <span>📸{s.mediaShared.photos}</span>
        <span>🎥{s.mediaShared.videos}</span>
      </span>
    ),
  }));
}

const CARD_LIMIT = 5;

/**
 * The four participant cards: contributors, fast responders, emoji champions,
 * media sharers. Fullscreen uses a ranked bar-list layout (like Top Words).
 */
export function ParticipantAnalytics({ participants, totalMessages }: ParticipantAnalyticsProps) {
  const contributors = contributorItems(participants.messageContributors, totalMessages);
  const responders = responderItems(participants.fastResponders);
  const emojis = emojiItems(participants.emojiUsers);
  const media = mediaItems(participants.mediaSharers);

  return (
    <div className={`grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 ${GRID_GAP}`}>
      <ChartCard
        title="Top Contributors"
        icon={MessageSquare}
        accent="blue"
        tooltip={{
          description:
            'Participants who have sent the most messages across your selected conversations, ranked by total message count. Data is filtered to only show participants from your selected conversations and recalculates in real-time when filters change.',
          calculation:
            'Sum of all messages sent by each participant in selected conversations, then ranked in descending order',
          example:
            'If Alice sent 1,500 messages, Bob sent 1,200, and Carol sent 800, they would appear in that order with percentages of total messages.',
        }}
        bodyClassName={`${CHART_MD} overflow-y-auto`}
        fullscreenChildren={
          <RankedListPanel items={contributors} tint="blue" variant="expanded" />
        }
      >
        <RankedListPanel items={contributors} tint="blue" limit={CARD_LIMIT} />
      </ChartCard>

      <ChartCard
        title="Fast Responders"
        icon={Timer}
        accent="orange"
        tooltip={{
          description:
            'Participants with the fastest average response times in conversations, showing who typically replies quickest to messages.',
          calculation:
            'Average response time calculated from time between receiving a message and sending the next message, converted from milliseconds to readable format',
          example:
            "If someone typically responds within 2 minutes on average, they'll show as '2m avg response' and rank higher than someone who takes 10 minutes.",
        }}
        bodyClassName={`${CHART_MD} overflow-y-auto`}
        fullscreenChildren={
          <RankedListPanel items={responders} tint="orange" variant="expanded" />
        }
      >
        <RankedListPanel items={responders} tint="orange" limit={CARD_LIMIT} />
      </ChartCard>

      <ChartCard
        title="Emoji Champions"
        icon={Heart}
        accent="purple"
        tooltip={{
          description:
            'Participants who use the most emojis in their messages, counted per conversation so filtering stays accurate.',
          calculation:
            'Sum of per-message emoji counts by sender within the selected conversation(s), from emojiMetrics data.',
          example:
            'If Alice used 120 emojis and Bob used 40 in this chat, Alice ranks first with 120.',
        }}
        bodyClassName={`${CHART_MD} overflow-y-auto`}
        fullscreenChildren={
          <RankedListPanel items={emojis} tint="purple" variant="expanded" />
        }
      >
        <RankedListPanel
          items={emojis}
          tint="purple"
          limit={CARD_LIMIT}
          emptyMessage="Per-sender emoji counts are not available yet. Re-run “Generate analytics” to compute them."
        />
      </ChartCard>

      <ChartCard
        title="Media Sharers"
        icon={Camera}
        accent="green"
        tooltip={{
          description:
            'Participants who shared the most photos, videos, and attachments in the selected conversation(s).',
          calculation:
            'Sum of photo_count + video_count + attachment_count from mediaMetrics.sender_media_data filtered by conversation_id.',
          example:
            'If Tia shared 40 photos and 5 videos in this chat, she shows 45 total media.',
        }}
        bodyClassName={`${CHART_MD} overflow-y-auto`}
        fullscreenChildren={
          <RankedListPanel items={media} tint="green" variant="expanded" />
        }
      >
        <RankedListPanel
          items={media}
          tint="green"
          limit={CARD_LIMIT}
          emptyMessage="No media shares in the selected conversation(s)."
        />
      </ChartCard>
    </div>
  );
}
