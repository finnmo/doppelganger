'use client';

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import type { Payload, ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import {
  aggregateReactionsForConversations,
  type ConversationReactionRow
} from '@/lib/filterReactions';

interface ReactionSummary {
  emoji: string;
  count: number;
  percentage: number;
  conversation_ids: string[];
  topReactors: Array<{
    sender: string;
    count: number;
  }>;
  topTargets: Array<{
    sender: string;
    count: number;
  }>;
}

interface SenderReactionStats {
  sender: string;
  conversation_ids: string[];
  reactionsGiven: number;
  reactionsReceived: number;
  topEmojisGiven: Array<{
    emoji: string;
    count: number;
  }>;
  topEmojisReceived: Array<{
    emoji: string;
    count: number;
  }>;
  reactionRatio: number;
}

interface ReactionData {
  summary: {
    totalReactions: number;
    uniqueEmojis: number;
    totalMessages: number;
    reactionRate: number;
    topEmoji: string;
    dateRange: {
      start: string;
      end: string;
    };
  };
  reactionSummaries: ReactionSummary[];
  senderStats: SenderReactionStats[];
  reactionsByConversation?: ConversationReactionRow[];
}

const EMOJI_COLORS = [
  '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
  '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
  '#10ac84', '#ee5a24', '#0abde3', '#3867d6', '#8854d0'
];

export default function ReactionMetricsChart() {
  const [data, setData] = useState<ReactionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'emojis' | 'senders'>('emojis');
  const { conversations, scopeConversationIds, participantIndex } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const reactionRes = await fetch('/data/reactionMetrics.json');
        const reactionData: ReactionData = await reactionRes.json();

        if (reactionData.reactionsByConversation) {
          const messageCount = conversations
            .filter((c) => scopeConversationIds.includes(c.conversation_id))
            .reduce((sum, c) => sum + (c.total_messages ?? 0), 0);

          const filtered = aggregateReactionsForConversations(
            reactionData.reactionsByConversation,
            scopeConversationIds,
            messageCount,
            participantIndex
          );

          setData({
            ...reactionData,
            summary: {
              ...reactionData.summary,
              ...filtered.summary
            },
            reactionSummaries: filtered.reactionSummaries,
            senderStats: filtered.senderStats
          });
        } else {
          setData(reactionData);
        }
      } catch (error) {
        console.error('Error loading reaction data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [conversations, scopeConversationIds, participantIndex]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading...</div>;
  }

  if (!data || data.summary.totalReactions === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No reaction data available
      </div>
    );
  }

  const formatTooltip = (value: ValueType | undefined, entry: Payload<ValueType, NameType>) => {
    const item = entry.payload;
    if (viewMode === 'emojis') {
      return [
        <div key="tooltip" className="text-sm">
          <div className="font-semibold text-lg">{item.emoji}</div>
          <div>Count: {Number(value).toLocaleString()}</div>
          <div>Percentage: {item.percentage?.toFixed(1)}%</div>
          {item.topReactors && item.topReactors.length > 0 && (
            <div className="text-xs text-gray-500 mt-1">
              Top reactor: {item.topReactors[0].sender} ({item.topReactors[0].count})
            </div>
          )}
        </div>
      ];
    }
    return [
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{item.sender}</div>
        <div>Given: {item.reactionsGiven}</div>
        <div>Received: {item.reactionsReceived}</div>
        <div>Ratio: {item.reactionRatio?.toFixed(2)}</div>
        {item.topEmojisGiven && item.topEmojisGiven.length > 0 && (
          <div className="text-xs text-gray-500 mt-1">
            Favorite: {item.topEmojisGiven[0].emoji} ({item.topEmojisGiven[0].count})
          </div>
        )}
      </div>
    ];
  };

  const emojiChartData = data.reactionSummaries.slice(0, 10);
  const senderChartData = data.senderStats.slice(0, 10);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-blue-600 text-sm font-medium">Total Reactions</div>
          <div className="text-2xl font-bold text-blue-900">
            {data.summary.totalReactions.toLocaleString()}
          </div>
        </div>
        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-green-600 text-sm font-medium">Reaction Rate</div>
          <div className="text-2xl font-bold text-green-900">
            {data.summary.reactionRate.toFixed(2)}%
          </div>
        </div>
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="text-purple-600 text-sm font-medium">Unique Emojis</div>
          <div className="text-2xl font-bold text-purple-900">
            {data.summary.uniqueEmojis}
          </div>
        </div>
        <div className="bg-orange-50 rounded-lg p-4">
          <div className="text-orange-600 text-sm font-medium">Top Emoji</div>
          <div className="text-2xl font-bold text-orange-900">
            {data.summary.topEmoji}
          </div>
        </div>
      </div>

      <div className="flex justify-center mb-4">
        <div className="bg-gray-100 rounded-lg p-1 flex">
          <button
            onClick={() => setViewMode('emojis')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              viewMode === 'emojis'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Emoji Usage
          </button>
          <button
            onClick={() => setViewMode('senders')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              viewMode === 'senders'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Sender Stats
          </button>
        </div>
      </div>

      <div className="h-96">
        <h3 className="text-lg font-semibold mb-4 text-center">
          {viewMode === 'emojis' ? 'Most Used Reaction Emojis' : 'Most Active Reactors'}
        </h3>

        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'emojis' ? (
            <BarChart data={emojiChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="emoji" stroke="#6b7280" fontSize={16} interval={0} />
              <YAxis stroke="#6b7280" fontSize={12} tickFormatter={(value) => value.toLocaleString()} />
              <ChartTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        {formatTooltip(payload[0].value, payload[0])}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {emojiChartData.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={EMOJI_COLORS[index % EMOJI_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          ) : (
            <BarChart data={senderChartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="sender"
                stroke="#6b7280"
                fontSize={10}
                angle={-45}
                textAnchor="end"
                height={60}
                interval={0}
              />
              <YAxis stroke="#6b7280" fontSize={12} tickFormatter={(value) => value.toLocaleString()} />
              <ChartTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        {formatTooltip(payload[0].value, payload[0])}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="reactionsGiven" fill="#3b82f6" name="Given" radius={[2, 2, 0, 0]} />
              <Bar dataKey="reactionsReceived" fill="#10b981" name="Received" radius={[2, 2, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {viewMode === 'emojis' && data.reactionSummaries.length > 0 && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold mb-3">Top Emoji Details</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.reactionSummaries.slice(0, 6).map((emoji) => (
              <div key={emoji.emoji} className="bg-white rounded-lg p-3 border">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl">{emoji.emoji}</span>
                  <span className="text-sm text-gray-500">{emoji.percentage.toFixed(1)}%</span>
                </div>
                <div className="text-sm text-gray-600">
                  <div>Used {emoji.count} times</div>
                  {emoji.topReactors.length > 0 && (
                    <div className="text-xs mt-1">
                      Top user: {emoji.topReactors[0].sender}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {viewMode === 'senders' && data.senderStats.length > 0 && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-semibold mb-3">Reaction Patterns</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.senderStats.slice(0, 4).map((sender) => (
              <div key={sender.sender} className="bg-white rounded-lg p-3 border">
                <div className="font-medium mb-2">{sender.sender}</div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div className="flex justify-between">
                    <span>Reactions Given:</span>
                    <span className="font-medium">{sender.reactionsGiven}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Reactions Received:</span>
                    <span className="font-medium">{sender.reactionsReceived}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Give/Receive Ratio:</span>
                    <span className="font-medium">{sender.reactionRatio.toFixed(2)}</span>
                  </div>
                  {sender.topEmojisGiven.length > 0 && (
                    <div className="flex justify-between">
                      <span>Favorite Emoji:</span>
                      <span>{sender.topEmojisGiven[0].emoji}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
