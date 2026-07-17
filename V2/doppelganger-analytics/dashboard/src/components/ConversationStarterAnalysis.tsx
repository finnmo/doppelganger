'use client';

import React, { useState, useEffect } from 'react';
import { MessageSquarePlus, Clock, Trophy } from 'lucide-react';
import { useParticipantScope } from '@/hooks/useParticipantScope';

interface StarterPattern {
  starter_sender: string;
  total_conversations_started: number;
  avg_engagement_score: number;
  preferred_starter_types: { type: string; count: number }[];
  common_topics: string[];
  time_patterns: { hour: number; count: number }[];
  starter_tier: 'prolific' | 'active' | 'occasional' | 'rare';
}

interface ConversationStarter {
  conversation_id: string;
  starter_sender: string;
  engagement_score?: number;
  starter_type?: string;
  topic_keywords?: string[];
  timestamp_ms?: number;
}

interface StarterData {
  summary: {
    total_conversations: number;
    total_starters: number;
    most_prolific_starter: string;
  };
  starter_patterns: StarterPattern[];
  conversation_starters?: ConversationStarter[];
}

const TIER_STYLES: Record<StarterPattern['starter_tier'], string> = {
  prolific: 'bg-purple-100 text-purple-700',
  active: 'bg-blue-100 text-blue-700',
  occasional: 'bg-green-100 text-green-700',
  rare: 'bg-gray-100 text-gray-600'
};

function formatHour(hour: number): string {
  return `${hour.toString().padStart(2, '0')}:00`;
}

function assignTier(count: number): StarterPattern['starter_tier'] {
  if (count >= 10) return 'prolific';
  if (count >= 5) return 'active';
  if (count >= 2) return 'occasional';
  return 'rare';
}

function rebuildPatternsFromStarters(starters: ConversationStarter[]): {
  starter_patterns: StarterPattern[];
  summary: StarterData['summary'];
} {
  const bySender = new Map<string, {
    count: number;
    engagementSum: number;
    types: Map<string, number>;
    topics: Map<string, number>;
    hours: Map<number, number>;
  }>();

  starters.forEach((starter) => {
    if (!bySender.has(starter.starter_sender)) {
      bySender.set(starter.starter_sender, {
        count: 0,
        engagementSum: 0,
        types: new Map(),
        topics: new Map(),
        hours: new Map()
      });
    }
    const stats = bySender.get(starter.starter_sender)!;
    stats.count++;
    stats.engagementSum += starter.engagement_score ?? 0;

    if (starter.starter_type) {
      stats.types.set(starter.starter_type, (stats.types.get(starter.starter_type) || 0) + 1);
    }
    (starter.topic_keywords || []).forEach((topic) => {
      stats.topics.set(topic, (stats.topics.get(topic) || 0) + 1);
    });
    if (typeof starter.timestamp_ms === 'number' && starter.timestamp_ms > 0) {
      const hour = new Date(starter.timestamp_ms).getHours();
      stats.hours.set(hour, (stats.hours.get(hour) || 0) + 1);
    }
  });

  const starter_patterns = Array.from(bySender.entries())
    .map(([starter_sender, stats]) => ({
      starter_sender,
      total_conversations_started: stats.count,
      avg_engagement_score: stats.count > 0 ? Math.round(stats.engagementSum / stats.count) : 0,
      preferred_starter_types: Array.from(stats.types.entries())
        .map(([type, count]) => ({ type, count }))
        .sort((a, b) => b.count - a.count),
      common_topics: Array.from(stats.topics.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([topic]) => topic),
      time_patterns: Array.from(stats.hours.entries())
        .map(([hour, count]) => ({ hour, count }))
        .sort((a, b) => b.count - a.count),
      starter_tier: assignTier(stats.count)
    }))
    .filter((p) => p.total_conversations_started > 0)
    .sort((a, b) => b.total_conversations_started - a.total_conversations_started);

  return {
    starter_patterns,
    summary: {
      total_conversations: new Set(starters.map(s => s.conversation_id)).size,
      total_starters: starters.length,
      most_prolific_starter: starter_patterns[0]?.starter_sender || 'N/A'
    }
  };
}

export function ConversationStarterAnalysis() {
  const [data, setData] = useState<StarterData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { filterScopedRows, scopeConversationIds, isFiltered } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/conversationStarterAnalysis.json');
        const starterData: StarterData = await response.json();

        const scopedStarters = filterScopedRows(
          (starterData.conversation_starters || []).map((s) => ({
            ...s,
            sender: s.starter_sender,
          })),
          { senderKey: 'sender' }
        ).map((starter) => {
          const { sender, ...rest } = starter;
          void sender;
          return rest;
        });

        const rebuilt = rebuildPatternsFromStarters(scopedStarters);
        setData({
          ...starterData,
          conversation_starters: scopedStarters,
          starter_patterns: rebuilt.starter_patterns,
          summary: rebuilt.summary,
        });
      } catch (error) {
        console.error('Error loading conversation starter data:', error);
        setData(null);
      } finally {
        setIsLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds]);

  if (isLoading) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="text-gray-500">Loading conversation starter analysis...</div>
      </div>
    );
  }

  if (!data || data.starter_patterns.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center bg-gray-50 rounded border-2 border-dashed border-gray-300">
        <div className="text-center">
          <MessageSquarePlus className="w-12 h-12 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-500">No conversation starter data available</p>
          <p className="text-sm text-gray-400">
            {isFiltered
              ? 'No starters found in the selected conversation(s)'
              : 'Run the analytics generation to populate this view'}
          </p>
        </div>
      </div>
    );
  }

  const topStarters = data.starter_patterns.slice(0, 6);
  const totalSessions = data.summary.total_starters;
  const singleConversation = isFiltered && scopeConversationIds.length === 1;
  const headerLabel = singleConversation
    ? `${totalSessions.toLocaleString()} chat sessions detected (4h+ gap between messages)`
    : `${totalSessions.toLocaleString()} chat sessions across ${data.summary.total_conversations.toLocaleString()} conversations`;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-sm text-gray-600">
        <span>{headerLabel}</span>
        <span className="flex items-center">
          <Trophy className="w-4 h-4 mr-1 text-amber-500" />
          Most prolific: <span className="font-medium ml-1">{data.summary.most_prolific_starter}</span>
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {topStarters.map((starter, index) => {
          const preferredType = starter.preferred_starter_types[0];
          const peakHour = starter.time_patterns[0];

          return (
            <div key={starter.starter_sender} className="bg-gray-50 p-4 rounded-lg border border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center min-w-0">
                  <span className="text-lg font-bold text-gray-400 mr-2">#{index + 1}</span>
                  <span className="font-medium text-gray-900 truncate">{starter.starter_sender}</span>
                </div>
                <span className={`text-xs px-2 py-1 rounded-full font-medium ${TIER_STYLES[starter.starter_tier]}`}>
                  {starter.starter_tier}
                </span>
              </div>

              <div className="text-2xl font-bold text-blue-600 mb-2">
                {starter.total_conversations_started}
                <span className="text-sm font-normal text-gray-500 ml-1">sessions started</span>
              </div>

              <div className="space-y-1 text-sm text-gray-600">
                {preferredType && (
                  <div className="flex justify-between">
                    <span>Usual opener</span>
                    <span className="font-medium capitalize">{preferredType.type}</span>
                  </div>
                )}
                {peakHour && (
                  <div className="flex justify-between">
                    <span className="flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      Peak start time
                    </span>
                    <span className="font-medium">{formatHour(peakHour.hour)}</span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
