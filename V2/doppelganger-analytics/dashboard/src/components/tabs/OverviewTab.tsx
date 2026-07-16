'use client';

import React from 'react';
import { BarChart3 } from 'lucide-react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { useTheme } from '@/contexts/ThemeContext';
import { computeFilteredOverviewMetrics } from '@/lib/overviewAggregate';
import { useOverviewData } from './overview/useOverviewData';
import { computeParticipantAnalytics } from './overview/participants';
import { overallAverageResponseTime } from './overview/responseTime';
import { HeroMetrics } from './overview/HeroMetrics';
import { ParticipantAnalytics } from './overview/ParticipantAnalytics';
import { ActivityAnalytics } from './overview/ActivityAnalytics';
import { PlatformSourcesBar } from '@/components/PlatformSourcesBar';

export function OverviewTab() {
  const { data, loading } = useOverviewData();
  const { selectedConversations, isFiltered, platforms, conversations } = useConversationFilter();
  const { themeStyle, getThemeClasses } = useTheme();

  // Headline metrics scoped to the current selection.
  const filteredMetrics = React.useMemo(() => {
    if (!data.text || !data.conversation || !data.media || !data.time || !data.latency) {
      return null;
    }
    return computeFilteredOverviewMetrics(
      {
        text: data.text,
        conversation: data.conversation,
        media: data.media,
        time: data.time,
        latency: data.latency,
        emoji: data.emoji?.senderEmojis,
        activeHours: data.activeHours ?? undefined
      },
      selectedConversations,
      isFiltered
    );
  }, [data, selectedConversations, isFiltered]);

  // Participant cards scoped to the selection. Message counts come from
  // per-conversation messages_by_sender (never global engagement totals
  // divided by a filtered conversation size).
  const participants = React.useMemo(
    () => computeParticipantAnalytics(
      data.conversation,
      data.media,
      data.engagement,
      data.turnTaking,
      data.emoji,
      selectedConversations,
      isFiltered
    ),
    [data, selectedConversations, isFiltered]
  );

  const platformsForView = React.useMemo(() => {
    if (!isFiltered || selectedConversations.length === 0) return platforms;
    const map = new Map<string, { source: string; label: string; conversations: number; messages: number }>();
    for (const c of conversations) {
      if (!selectedConversations.includes(c.conversation_id)) continue;
      const entry = map.get(c.source) ?? {
        source: c.source,
        label: c.source_label,
        conversations: 0,
        messages: 0,
      };
      entry.conversations += 1;
      entry.messages += c.total_messages;
      map.set(c.source, entry);
    }
    return [...map.values()].sort((a, b) => b.messages - a.messages);
  }, [platforms, conversations, selectedConversations, isFiltered]);

  const activeSource =
    selectedConversations.length === 1
      ? conversations.find((c) => c.conversation_id === selectedConversations[0])?.source
      : null;

  if (loading) {
    return (
      <div className="space-y-8">
        {/* Header Skeleton */}
        <div>
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-2 animate-pulse"></div>
          <div className="h-4 bg-gray-200 rounded w-2/3 animate-pulse"></div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-white rounded-lg border p-6 animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
              <div className="h-8 bg-gray-200 rounded w-1/2"></div>
            </div>
          ))}
        </div>

        <div className="text-center text-gray-500 mt-8">
          Loading conversation analytics...
        </div>
      </div>
    );
  }

  if (!filteredMetrics) {
    return (
      <div className="space-y-8">
        <div className="text-center py-12">
          <div className="text-red-600 text-lg font-semibold mb-2">⚠️ Data Loading Error</div>
          <div className="text-gray-600 mb-4">
            Unable to load conversation data. Please check your data files and try refreshing the page.
          </div>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Refresh Page
          </button>
        </div>
      </div>
    );
  }

  const avgResponseTime = overallAverageResponseTime(
    data.turnTaking,
    selectedConversations,
    isFiltered,
    filteredMetrics.filteredLatency
  );

  const themeClasses = getThemeClasses();

  return (
    <div className={themeClasses.spacingClass}>
      {/* Header */}
      <div className="relative">
        <div className={`absolute inset-0 ${themeClasses.headerGradientClass}`}></div>
        <div className={`relative ${themeStyle === 'modern' ? 'p-8' : 'p-6'}`}>
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 flex flex-wrap items-center gap-2 mb-1">
            <BarChart3 className="w-8 h-8 mr-3 text-blue-600" />
            Overview of Conversation
          </h2>
          <p className="text-lg text-gray-600">
            Comprehensive insights and patterns across your conversations
          </p>
        </div>
      </div>

      <PlatformSourcesBar platforms={platformsForView} activeSource={activeSource} />

      <HeroMetrics metrics={filteredMetrics} avgResponseTime={avgResponseTime} />

      <ParticipantAnalytics
        participants={participants}
        totalMessages={filteredMetrics.textMetrics.totalMessages}
      />

      <ActivityAnalytics />
    </div>
  );
}
