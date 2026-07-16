'use client';

import React, { useState, useEffect } from 'react';
import { ThreadVisualization } from '@/components/ThreadVisualization';
import { TurnTakingAnalysis } from '@/components/TurnTakingAnalysis';
import { EngagementScoring } from '@/components/EngagementScoring';
import { ConversationStarterAnalysis } from '@/components/ConversationStarterAnalysis';
import { ChartCard } from '@/components/ui/ChartCard';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { Users, MessageSquare, GitBranch, Target } from 'lucide-react';
import { GRID_GAP, TAB_STACK } from '@/lib/layout';

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

/**
 * Conversations & Threads — single-screen grid.
 * Row 1: four summary stat tiles.
 * Row 2: thread depth · turn-taking · engagement scores · starter analysis
 * (each scrolls internally so the page never grows).
 */
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

  const tiles = [
    {
      label: 'Total Conversations',
      value: summary ? summary.totalConversations.toLocaleString() : '—',
      hint: 'Active conversation threads',
      classes: 'bg-blue-50 border-blue-100 text-blue-600',
    },
    {
      label: 'Avg Participants',
      value: summary ? summary.averageParticipants.toFixed(1) : '—',
      hint: 'People per conversation',
      classes: 'bg-green-50 border-green-100 text-green-600',
    },
    {
      label: 'Avg Turns',
      value: summary ? Math.round(summary.averageTurns).toLocaleString() : '—',
      hint: 'Back-and-forth exchanges',
      classes: 'bg-purple-50 border-purple-100 text-purple-600',
    },
    {
      label: 'Avg Duration',
      value: summary ? formatDuration(summary.averageDuration) : '—',
      hint: 'Conversation lifespan',
      classes: 'bg-orange-50 border-orange-100 text-orange-600',
    },
  ];

  return (
    <div className={TAB_STACK}>
      <div className={`grid grid-cols-2 xl:grid-cols-4 ${GRID_GAP}`}>
        {tiles.map(tile => (
          <div key={tile.label} className={`rounded-xl border p-3 sm:p-4 ${tile.classes.split(' ').slice(0, 2).join(' ')}`}>
            <div className="text-xs font-medium text-gray-700 sm:text-sm">{tile.label}</div>
            <div className={`text-xl font-bold sm:text-2xl ${tile.classes.split(' ')[2]}`}>{tile.value}</div>
            <div className="text-xs text-gray-500">{tile.hint}</div>
          </div>
        ))}
      </div>

      <div className={`grid grid-cols-1 lg:grid-cols-2 ${GRID_GAP}`}>
        <ChartCard
          title="Thread Depth Visualization"
          icon={GitBranch}
          accent="green"
          tooltip={{
            description:
              'Visual representation of conversation threads and reply chains.',
          }}
          bodyClassName='h-60 overflow-y-auto'
        >
          <ThreadVisualization />
        </ChartCard>

        <ChartCard
          title="Turn-taking Patterns"
          icon={MessageSquare}
          accent="indigo"
          tooltip={{
            description:
              'How conversation participants take turns and engage with each other.',
          }}
          bodyClassName='h-60 overflow-y-auto'
        >
          <TurnTakingAnalysis />
        </ChartCard>

        <ChartCard
          title="Participant Engagement Scores"
          icon={Users}
          accent="red"
          tooltip={{
            description:
              'Engagement levels and participation patterns for each conversation member.',
          }}
          bodyClassName='h-60 overflow-y-auto'
        >
          <EngagementScoring />
        </ChartCard>

        <ChartCard
          title="Conversation Starter Analysis"
          icon={Target}
          accent="orange"
          tooltip={{
            description:
              'Who initiates conversations, how they usually open, and when.',
          }}
          bodyClassName='h-60 overflow-y-auto'
        >
          <ConversationStarterAnalysis />
        </ChartCard>
      </div>
    </div>
  );
}
