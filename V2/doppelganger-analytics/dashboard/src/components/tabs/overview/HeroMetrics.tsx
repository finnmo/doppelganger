'use client';

import React from 'react';
import { MessageCircle, Image as ImageIcon, Zap } from 'lucide-react';
import { ChartCard } from '@/components/ui/ChartCard';
import { CARD_FILL, BODY_FILL } from '@/lib/layout';
import type { FilteredOverviewMetrics } from '@/lib/overviewAggregate';

interface HeroMetricsProps {
  metrics: FilteredOverviewMetrics;
  avgResponseTime: string;
}

/**
 * The three headline cards: Total Messages, Media Shared, Communication Style.
 * Root is `contents` so the cards slot directly into the parent grid row.
 */
export function HeroMetrics({ metrics, avgResponseTime }: HeroMetricsProps) {
  const totalMedia = metrics.mediaMetrics.total_photos +
    metrics.mediaMetrics.total_videos +
    metrics.mediaMetrics.total_attachments;

  return (
    <div className="contents">
      <ChartCard
        title="Total Messages"
        icon={MessageCircle}
        accent="blue"
        className={CARD_FILL}
        bodyClassName={BODY_FILL}
        tooltip={{
          description:
            'The complete count of all messages sent across your selected conversations, including text messages, media sharing, and reactions.',
          calculation: 'Sum of all messages from all participants in selected conversations',
          example:
            'If you have 3 conversations with 1,000, 2,500, and 1,200 messages respectively, your total would be 4,700 messages.',
        }}
      >
        <div className="flex h-full min-h-0 flex-col justify-between">
          <div className="text-3xl font-bold text-blue-600">
            {metrics.textMetrics.totalMessages.toLocaleString()}
          </div>
          <div className="mt-3 space-y-1.5 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Participants</span>
              <span className="font-semibold text-gray-900">{metrics.conversationSummary.totalUniqueParticipants}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Emojis</span>
              <span className="font-semibold text-gray-900">{metrics.textMetrics.totalEmojis.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Avg Response</span>
              <span className="font-semibold text-gray-900">{avgResponseTime}</span>
            </div>
          </div>
          <div className="mt-3 border-t border-gray-200 pt-2 text-xs text-gray-500">
            Across {metrics.conversationSummary.totalConversations} conversation{metrics.conversationSummary.totalConversations !== 1 ? 's' : ''}
          </div>
        </div>
      </ChartCard>

      <ChartCard
        title="Media Shared"
        icon={ImageIcon}
        accent="orange"
        className={CARD_FILL}
        bodyClassName={BODY_FILL}
        tooltip={{
          description:
            'Consolidated overview of all media content shared across your selected conversations, broken down by type.',
          calculation:
            'Total Media = Photos + Videos + Attachments, each counted from message content analysis',
          example:
            'If your conversations contain 1,234 photos, 567 videos, and 89 attachments, Total Media shows 1,890.',
        }}
      >
        <div className="flex h-full min-h-0 flex-col justify-between">
          <div className="text-3xl font-bold text-orange-600">{totalMedia.toLocaleString()}</div>
          <div className="mt-3 space-y-1.5 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Photos</span>
              <span className="font-semibold text-gray-900">{metrics.mediaMetrics.total_photos.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Videos</span>
              <span className="font-semibold text-gray-900">{metrics.mediaMetrics.total_videos.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Attachments</span>
              <span className="font-semibold text-gray-900">{metrics.mediaMetrics.total_attachments.toLocaleString()}</span>
            </div>
          </div>
          <div className="mt-3 border-t border-gray-200 pt-2 text-xs text-gray-500">
            {metrics.mediaMetrics.media_percentage.toFixed(1)}% of messages contain media
          </div>
        </div>
      </ChartCard>

      <ChartCard
        title="Communication Style"
        icon={Zap}
        accent="green"
        className={CARD_FILL}
        bodyClassName={BODY_FILL}
        tooltip={{
          description:
            'Analysis of your communication personality and patterns based on conversation behavior, response times, and engagement levels.',
          calculation:
            'Determined by conversation count, emoji usage patterns, response speed analysis, and overall communication health metrics',
          example:
            "'Active' indicates multiple conversations with high engagement, while 'Focused' suggests concentrated communication in fewer channels.",
        }}
      >
        <div className="flex h-full min-h-0 flex-col justify-between">
          <div className="text-3xl font-bold text-green-600">
            {metrics.conversationSummary.totalConversations > 1 ? 'Active' : 'Focused'}
          </div>
          <div className="mt-3 space-y-1.5 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Health</span>
              <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-semibold text-green-800">
                {metrics.conversationSummary.totalConversations > 1 ? 'Multi-chat' : 'Single-chat'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Engagement</span>
              <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-semibold text-green-800">
                {metrics.textMetrics.totalEmojis > 1000 ? 'High' :
                 metrics.textMetrics.totalEmojis > 100 ? 'Moderate' : 'Low'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Response</span>
              <span className="rounded bg-green-100 px-2 py-0.5 text-xs font-semibold text-green-800">
                {avgResponseTime.includes('s') ? 'Instant' :
                 avgResponseTime.includes('m') && parseInt(avgResponseTime) < 30 ? 'Quick' : 'Relaxed'}
              </span>
            </div>
          </div>
          <div className="mt-3 border-t border-gray-200 pt-2 text-xs text-gray-500">
            Communication personality analysis
          </div>
        </div>
      </ChartCard>
    </div>
  );
}
