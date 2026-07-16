'use client';

import React from 'react';
import { Camera, Heart, MessageSquare, Timer } from 'lucide-react';
import { ChartCard } from '@/components/ui/ChartCard';
import { CHART_MD, GRID_GAP } from '@/lib/layout';
import type { ParticipantAnalyticsData } from './participants';

interface ParticipantAnalyticsProps {
  participants: ParticipantAnalyticsData;
  totalMessages: number;
}

function truncateName(name: string, max = 22): string {
  return name.length > max ? `${name.substring(0, max)}…` : name;
}

function formatResponseTime(ms: number): string {
  const seconds = ms / 1000;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

interface RankedRowProps {
  rank: number;
  name: string;
  value: string;
  sub: React.ReactNode;
  tint: 'blue' | 'orange' | 'purple' | 'green';
}

const ROW_TINT: Record<RankedRowProps['tint'], { row: string; badge: string; value: string }> = {
  blue: { row: 'bg-blue-50 border-blue-100', badge: 'bg-blue-600', value: 'text-blue-600' },
  orange: { row: 'bg-orange-50 border-orange-100', badge: 'bg-orange-600', value: 'text-orange-600' },
  purple: { row: 'bg-purple-50 border-purple-100', badge: 'bg-purple-600', value: 'text-purple-600' },
  green: { row: 'bg-green-50 border-green-100', badge: 'bg-green-600', value: 'text-green-600' },
};

function RankedRow({ rank, name, value, sub, tint }: RankedRowProps) {
  const t = ROW_TINT[tint];
  return (
    <div className={`flex items-center justify-between rounded-md border p-2 ${t.row}`}>
      <div className="flex min-w-0 items-center">
        <div className={`mr-2 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold text-white ${t.badge}`}>
          {rank}
        </div>
        <span className="truncate text-sm font-medium text-gray-800" title={name}>
          {truncateName(name)}
        </span>
      </div>
      <div className="ml-2 shrink-0 text-right">
        <div className={`text-sm font-bold ${t.value}`}>{value}</div>
        <div className="text-xs text-gray-600">{sub}</div>
      </div>
    </div>
  );
}

/**
 * The four participant cards: contributors, fast responders, emoji champions,
 * media sharers. Data is filtered to the current conversation selection and
 * recalculates when filters change (see each card's tooltip).
 */
export function ParticipantAnalytics({ participants, totalMessages }: ParticipantAnalyticsProps) {
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
      >
        <div className="space-y-2">
          {participants.messageContributors.map((contributor, index) => (
            <RankedRow
              key={contributor.participant}
              rank={index + 1}
              name={contributor.participant}
              value={contributor.total_messages.toLocaleString()}
              sub={totalMessages > 0
                ? `${Math.round((contributor.total_messages / totalMessages) * 100)}%`
                : '—'}
              tint="blue"
            />
          ))}
        </div>
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
      >
        <div className="space-y-2">
          {participants.fastResponders.map((responder, index) => (
            <RankedRow
              key={responder.participant}
              rank={index + 1}
              name={responder.participant}
              value={formatResponseTime(responder.avg_response_time)}
              sub="avg response"
              tint="orange"
            />
          ))}
        </div>
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
      >
        <div className="space-y-2">
          {participants.emojiUsers.length === 0 ? (
            <p className="text-sm text-gray-500 italic">
              Per-sender emoji counts are not available yet. Re-run “Generate analytics” to compute them.
            </p>
          ) : participants.emojiUsers.map((user, index) => (
            <RankedRow
              key={user.sender}
              rank={index + 1}
              name={user.sender}
              value={user.count.toLocaleString()}
              sub="emojis used"
              tint="purple"
            />
          ))}
        </div>
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
      >
        <div className="space-y-2">
          {participants.mediaSharers.length === 0 ? (
            <p className="text-sm text-gray-500 italic">No media shares in the selected conversation(s).</p>
          ) : participants.mediaSharers.map((sharer, index) => (
            <RankedRow
              key={sharer.sender}
              rank={index + 1}
              name={sharer.sender}
              value={sharer.mediaShared.total.toLocaleString()}
              sub={<span className="flex space-x-1"><span>📸{sharer.mediaShared.photos}</span><span>🎥{sharer.mediaShared.videos}</span></span>}
              tint="green"
            />
          ))}
        </div>
      </ChartCard>
    </div>
  );
}
