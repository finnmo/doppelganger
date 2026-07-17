'use client';

import React from 'react';
import { TrendingUp } from 'lucide-react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { computeFilteredOverviewMetrics } from '@/lib/overviewAggregate';
import { useOverviewData } from './overview/useOverviewData';
import { computeParticipantAnalytics } from './overview/participants';
import { overallAverageResponseTime } from './overview/responseTime';
import { HeroMetrics } from './overview/HeroMetrics';
import { ParticipantAnalytics } from './overview/ParticipantAnalytics';
import { PlatformSourcesBar } from '@/components/PlatformSourcesBar';
import { PeakActivityChart } from '@/components/PeakActivityChart';
import { ChartCard } from '@/components/ui/ChartCard';
import { TAB_VIEWPORT, ROW_FILL, GRID_GAP, CARD_FILL, BODY_FILL } from '@/lib/layout';

export function OverviewTab() {
  const { data, loading } = useOverviewData();
  const { selectedConversations, isFiltered, platforms, conversations } = useConversationFilter();

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

  return (
    <div className={TAB_VIEWPORT}>
      <PlatformSourcesBar platforms={platformsForView} activeSource={activeSource} />

      <div className={`${ROW_FILL} min-h-0`}>
      <ParticipantAnalytics
        participants={participants}
        totalMessages={filteredMetrics.textMetrics.totalMessages}
      />
      </div>

      <div className={`grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 ${GRID_GAP} ${ROW_FILL}`}>
        <HeroMetrics metrics={filteredMetrics} avgResponseTime={avgResponseTime} />

        <ChartCard
          title="Peak Activity Patterns"
          icon={TrendingUp}
          accent="green"
          tooltip={{
            description:
              'Visual analysis showing when communication is most active during different hours and days, revealing your natural communication rhythm.',
            calculation:
              'Messages are analyzed by hour and day of week to identify peak activity periods and patterns',
            example:
              "You might discover you're most active Tuesday-Thursday from 2-4 PM, or that weekends show completely different patterns.",
          }}
          bodyClassName={BODY_FILL}
          className={CARD_FILL}
        >
          <PeakActivityChart />
        </ChartCard>
      </div>
    </div>
  );
}
