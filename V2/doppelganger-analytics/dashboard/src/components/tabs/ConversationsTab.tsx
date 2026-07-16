'use client';

import React, { useState, useEffect } from 'react';
import { ThreadVisualization } from '@/components/ThreadVisualization';
import { TurnTakingAnalysis } from '@/components/TurnTakingAnalysis';
import { EngagementScoring } from '@/components/EngagementScoring';
import { ConversationStarterAnalysis } from '@/components/ConversationStarterAnalysis';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { Users, MessageSquare, GitBranch, Target } from 'lucide-react';

interface ConversationSummary {
  totalConversations: number;
  averageParticipants: number;
  averageTurns: number;
  averageDuration: number;
}

function formatDuration(durationMs: number): string {
  const days = durationMs / (24 * 60 * 60 * 1000);
  if (days >= 365) return `${(days / 365).toFixed(1)}y`;
  if (days >= 1) return `${Math.round(days)}d`;
  return `${Math.round(durationMs / (60 * 60 * 1000))}h`;
}

export function ConversationsTab() {
  const [summary, setSummary] = useState<ConversationSummary | null>(null);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadSummary = async () => {
      try {
        const response = await fetch('/data/conversationMetrics.json');
        const data = await response.json();

        let conversations: Array<{
          conversation_id: string;
          participants: string[];
          turns: number;
          duration_ms: number;
        }> = data.conversations || [];

        if (isFiltered && selectedConversations.length > 0) {
          conversations = conversations.filter(conv =>
            selectedConversations.includes(conv.conversation_id)
          );
        }

        if (conversations.length === 0) {
          setSummary(null);
          return;
        }

        setSummary({
          totalConversations: conversations.length,
          averageParticipants:
            conversations.reduce((sum, conv) => sum + conv.participants.length, 0) / conversations.length,
          averageTurns:
            conversations.reduce((sum, conv) => sum + conv.turns, 0) / conversations.length,
          averageDuration:
            conversations.reduce((sum, conv) => sum + conv.duration_ms, 0) / conversations.length
        });
      } catch (error) {
        console.error('Error loading conversation metrics:', error);
      }
    };

    loadSummary();
  }, [selectedConversations, isFiltered]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-green-50 via-teal-50 to-blue-50 rounded-2xl opacity-40"></div>
        <div className="relative p-6">
          <h2 className="text-3xl font-bold text-gray-900 flex items-center mb-1">
            <Users className="w-8 h-8 mr-3 text-green-600" />
            Conversations & Thread Analysis
          </h2>
          <p className="text-lg text-gray-600">
            Conversation structure, thread depth analysis, and participant engagement patterns
          </p>
        </div>
      </div>

      {/* Conversation Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Conversation Metrics Overview */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <MessageSquare className="w-5 h-5 mr-2 text-blue-500" />
            Conversation Metrics Overview
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Key metrics for conversation engagement and participation
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 p-4 rounded border border-blue-100">
              <div className="text-sm font-medium text-gray-700">Total Conversations</div>
              <div className="text-2xl font-bold text-blue-600">
                {summary ? summary.totalConversations.toLocaleString() : '—'}
              </div>
              <div className="text-xs text-gray-500">Active conversation threads</div>
            </div>
            <div className="bg-green-50 p-4 rounded border border-green-100">
              <div className="text-sm font-medium text-gray-700">Avg Participants</div>
              <div className="text-2xl font-bold text-green-600">
                {summary ? summary.averageParticipants.toFixed(1) : '—'}
              </div>
              <div className="text-xs text-gray-500">People per conversation</div>
            </div>
            <div className="bg-purple-50 p-4 rounded border border-purple-100">
              <div className="text-sm font-medium text-gray-700">Avg Turns</div>
              <div className="text-2xl font-bold text-purple-600">
                {summary ? Math.round(summary.averageTurns).toLocaleString() : '—'}
              </div>
              <div className="text-xs text-gray-500">Back-and-forth exchanges</div>
            </div>
            <div className="bg-orange-50 p-4 rounded border border-orange-100">
              <div className="text-sm font-medium text-gray-700">Avg Duration</div>
              <div className="text-2xl font-bold text-orange-600">
                {summary ? formatDuration(summary.averageDuration) : '—'}
              </div>
              <div className="text-xs text-gray-500">Conversation lifespan</div>
            </div>
          </div>
        </div>

        {/* Thread Depth Visualization */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <GitBranch className="w-5 h-5 mr-2 text-green-500" />
            Thread Depth Visualization
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Visual representation of conversation threads and reply chains
          </p>

          <ThreadVisualization />
        </div>
      </div>

      {/* Advanced Conversation Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Turn-taking Patterns */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <MessageSquare className="w-5 h-5 mr-2 text-indigo-500" />
            Turn-taking Patterns
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            How conversation participants take turns and engage with each other
          </p>

          <TurnTakingAnalysis />
        </div>

        {/* Participant Engagement Scores */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-red-500" />
            Participant Engagement Scores
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Engagement levels and participation patterns for each conversation member
          </p>

          <EngagementScoring />
        </div>
      </div>

      {/* Conversation Starter Analysis */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2 text-orange-500" />
          Conversation Starter Analysis
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Who initiates conversations, how they usually open, and when
        </p>

        <ConversationStarterAnalysis />
      </div>
    </div>
  );
}
