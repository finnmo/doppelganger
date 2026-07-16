'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
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

export function ReactionSummaryCards() {
  const [data, setData] = useState<ReactionData | null>(null);
  const [loading, setLoading] = useState(true);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const [reactionRes, conversationRes] = await Promise.all([
          fetch('/data/reactionMetrics.json'),
          fetch('/data/conversationMetrics.json')
        ]);
        const reactionData: ReactionData = await reactionRes.json();
        const conversationData: {
          conversations: Array<{ conversation_id: string; total_messages: number }>;
        } = await conversationRes.json();

        if (isFiltered && selectedConversations.length > 0 && reactionData.reactionsByConversation) {
          const messageCount = conversationData.conversations
            .filter(c => selectedConversations.includes(c.conversation_id))
            .reduce((sum, c) => sum + c.total_messages, 0);

          const filtered = aggregateReactionsForConversations(
            reactionData.reactionsByConversation,
            selectedConversations,
            messageCount
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
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {[1, 2, 3].map(i => (
          <div key={i} className="bg-gray-50 p-4 rounded border animate-pulse">
            <div className="h-4 bg-gray-200 rounded mb-2"></div>
            <div className="h-8 bg-gray-200 rounded"></div>
          </div>
        ))}
      </div>
    );
  }

  if (!data || data.summary.totalReactions === 0 || data.senderStats.length === 0) {
    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="bg-gray-50 p-4 rounded border">
          <h4 className="font-medium text-gray-900 mb-2">Reaction Types</h4>
          <div className="text-center text-gray-500 py-4">
            <div className="text-sm">No reaction data available</div>
          </div>
        </div>
        <div className="bg-gray-50 p-4 rounded border">
          <h4 className="font-medium text-gray-900 mb-2">Most Reactive</h4>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-400">0</div>
            <div className="text-sm text-gray-500">reactions given</div>
          </div>
        </div>
        <div className="bg-gray-50 p-4 rounded border">
          <h4 className="font-medium text-gray-900 mb-2">Top Reacted</h4>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-400">0</div>
            <div className="text-sm text-gray-500">reactions received</div>
          </div>
        </div>
      </div>
    );
  }

  const topReactions = data.reactionSummaries.slice(0, 3);
  const mostReactiveSender = data.senderStats.reduce((prev, current) =>
    prev.reactionsGiven > current.reactionsGiven ? prev : current
  );
  const mostReactedSender = data.senderStats.reduce((prev, current) =>
    prev.reactionsReceived > current.reactionsReceived ? prev : current
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
      <div className="bg-gray-50 p-4 rounded border">
        <h4 className="font-medium text-gray-900 mb-2">Reaction Types</h4>
        <div className="space-y-2">
          {topReactions.map((reaction) => (
            <div key={reaction.emoji} className="flex justify-between items-center">
              <span className="text-sm flex items-center">
                <span className="text-lg mr-2">{reaction.emoji}</span>
                {reaction.emoji === '❤️' || reaction.emoji === '❤' ? 'Hearts' :
                 reaction.emoji === '😂' ? 'Laughs' :
                 reaction.emoji === '👍' ? 'Likes' :
                 reaction.emoji === '😮' ? 'Surprised' :
                 reaction.emoji === '😢' ? 'Sad' :
                 reaction.emoji === '😡' ? 'Angry' : 'Other'}
              </span>
              <span className="text-sm font-medium">{reaction.count.toLocaleString()}</span>
            </div>
          ))}
          {topReactions.length === 0 && (
            <div className="text-sm text-gray-500">No reactions found</div>
          )}
        </div>
      </div>

      <div className="bg-gray-50 p-4 rounded border">
        <h4 className="font-medium text-gray-900 mb-2">Most Reactive</h4>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">
            {mostReactiveSender.reactionsGiven.toLocaleString()}
          </div>
          <div className="text-sm text-gray-500">reactions given</div>
          <div className="text-xs text-gray-600 mt-1 truncate" title={mostReactiveSender.sender}>
            {mostReactiveSender.sender.length > 20
              ? mostReactiveSender.sender.substring(0, 20) + '...'
              : mostReactiveSender.sender}
          </div>
        </div>
      </div>

      <div className="bg-gray-50 p-4 rounded border">
        <h4 className="font-medium text-gray-900 mb-2">Most Reacted</h4>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">
            {mostReactedSender.reactionsReceived.toLocaleString()}
          </div>
          <div className="text-sm text-gray-500">reactions received</div>
          <div className="text-xs text-gray-600 mt-1 truncate" title={mostReactedSender.sender}>
            {mostReactedSender.sender.length > 20
              ? mostReactedSender.sender.substring(0, 20) + '...'
              : mostReactedSender.sender}
          </div>
        </div>
      </div>
    </div>
  );
}
