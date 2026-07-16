'use client';

import React from 'react';
import { MessageCircle, Image as ImageIcon, Zap } from 'lucide-react';
import { InfoTooltip } from '@/components/InfoTooltip';
import { useTheme } from '@/contexts/ThemeContext';
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
  const { getThemeClasses } = useTheme();
  const themeClasses = getThemeClasses();

  const totalMedia = metrics.mediaMetrics.total_photos +
    metrics.mediaMetrics.total_videos +
    metrics.mediaMetrics.total_attachments;

  return (
    <div className="contents">
      {/* Total Messages - Blue (Communication/Text) */}
      <div className={themeClasses.heroCardClass('blue')}>
        <div className="flex items-center mb-2">
          <MessageCircle className="w-5 h-5 mr-2 text-blue-600" />
          <h3 className="text-base font-bold mr-2 text-gray-900">Total Messages</h3>
          <InfoTooltip
            title="Total Messages"
            description="The complete count of all messages sent across your selected conversations, including text messages, media sharing, and reactions."
            calculation="Sum of all messages from all participants in selected conversations"
            example="If you have 3 conversations with 1,000, 2,500, and 1,200 messages respectively, your total would be 4,700 messages."
            iconColor="default"
          />
        </div>

        <div className="text-3xl font-bold mb-2 text-blue-600">
          {metrics.textMetrics.totalMessages.toLocaleString()}
        </div>

        <div className="space-y-1.5 mb-3 text-sm">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">👥 Participants</span>
            <span className="font-semibold text-gray-900">{metrics.conversationSummary.totalUniqueParticipants}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">😊 Emojis</span>
            <span className="font-semibold text-gray-900">{metrics.textMetrics.totalEmojis.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">⏱️ Avg Response</span>
            <span className="font-semibold text-gray-900">{avgResponseTime}</span>
          </div>
        </div>

        <div className="pt-2 border-t border-gray-200">
          <div className="text-xs text-gray-500">
            Across {metrics.conversationSummary.totalConversations} conversation{metrics.conversationSummary.totalConversations !== 1 ? 's' : ''}
          </div>
        </div>
      </div>

      {/* Media Shared - Orange (Visual/Creative Content) */}
      <div className={themeClasses.heroCardClass('orange')}>
        <div className="flex items-center mb-2">
          <ImageIcon className="w-5 h-5 mr-2 text-orange-600" />
          <h3 className="text-base font-bold mr-2 text-gray-900">Media Shared</h3>
          <InfoTooltip
            title="Media Summary"
            description="Consolidated overview of all media content shared across your selected conversations, broken down by type."
            calculation="Total Media = Photos + Videos + Attachments, each counted from message content analysis"
            example="If your conversations contain 1,234 photos, 567 videos, and 89 attachments, Total Media shows 1,890."
            iconColor="default"
          />
        </div>

        <div className="text-3xl font-bold mb-2 text-orange-600">{totalMedia.toLocaleString()}</div>

        <div className="space-y-1.5 mb-3 text-sm">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">📸 Photos</span>
            <span className="font-semibold text-gray-900">{metrics.mediaMetrics.total_photos.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">🎥 Videos</span>
            <span className="font-semibold text-gray-900">{metrics.mediaMetrics.total_videos.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">📎 Attachments</span>
            <span className="font-semibold text-gray-900">{metrics.mediaMetrics.total_attachments.toLocaleString()}</span>
          </div>
        </div>

        <div className="pt-2 border-t border-gray-200">
          <div className="text-xs text-gray-500">
            {metrics.mediaMetrics.media_percentage.toFixed(1)}% of messages contain media
          </div>
        </div>
      </div>

      {/* Communication Style - Green (Health/Growth/Patterns) */}
      <div className={themeClasses.heroCardClass('green')}>
        <div className="flex items-center mb-2">
          <Zap className="w-5 h-5 mr-2 text-green-600" />
          <h3 className="text-base font-bold mr-2 text-gray-900">Communication Style</h3>
          <InfoTooltip
            title="Communication Style"
            description="Analysis of your communication personality and patterns based on conversation behavior, response times, and engagement levels."
            calculation="Determined by conversation count, emoji usage patterns, response speed analysis, and overall communication health metrics"
            example="'Active' indicates multiple conversations with high engagement, while 'Focused' suggests concentrated communication in fewer channels."
            iconColor="default"
          />
        </div>

        <div className="text-3xl font-bold mb-2 text-green-600">
          {metrics.conversationSummary.totalConversations > 1 ? 'Active' : 'Focused'}
        </div>

        <div className="space-y-1.5 mb-3 text-sm">
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Health</span>
            <span className="font-semibold bg-green-100 text-green-800 px-2 py-0.5 rounded text-xs">
              {metrics.conversationSummary.totalConversations > 1 ? 'Multi-chat' : 'Single-chat'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Engagement</span>
            <span className="font-semibold bg-green-100 text-green-800 px-2 py-0.5 rounded text-xs">
              {metrics.textMetrics.totalEmojis > 1000 ? 'High' :
               metrics.textMetrics.totalEmojis > 100 ? 'Moderate' : 'Low'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Response</span>
            <span className="font-semibold bg-green-100 text-green-800 px-2 py-0.5 rounded text-xs">
              {avgResponseTime.includes('s') ? 'Instant' :
               avgResponseTime.includes('m') && parseInt(avgResponseTime) < 30 ? 'Quick' : 'Relaxed'}
            </span>
          </div>
        </div>

        <div className="pt-2 border-t border-gray-200">
          <div className="text-xs text-gray-500">
            Communication personality analysis
          </div>
        </div>
      </div>
    </div>
  );
}
