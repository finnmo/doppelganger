'use client';

import React, { useState, useEffect } from 'react';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import { isKnownParticipant } from '@/lib/participantFilter';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import type { Payload, ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { MessageSquare, Users, TrendingUp, Clock } from 'lucide-react';

interface TurnData {
  conversation_id: string;
  participant: string;
  turn_count: number;
  avg_turn_length: number;
  max_turn_length: number;
  turn_percentage: number;
  avg_response_time: number;
  interruption_rate: number;
}

interface TurnPattern {
  pattern_type: 'balanced' | 'dominant' | 'responsive' | 'sporadic';
  participants: string[];
  turn_ratio: number;
  conversation_health: number;
  description: string;
}

interface TurnTakingMetrics {
  summary: {
    total_conversations: number;
    avg_participants: number;
    balanced_conversations: number;
    dominant_speaker_conversations: number;
    avg_turn_length: number;
    avg_response_time: number;
  };
  conversation_patterns: Array<{
    conversation_id: string;
    pattern: TurnPattern;
    participants: TurnData[];
    turn_sequence: Array<{
      participant: string;
      turn_number: number;
      message_count: number;
      duration_ms: number;
    }>;
  }>;
  participant_stats: Array<{
    participant: string;
    conversations: number;
    avg_turn_count: number;
    avg_turn_length: number;
    dominance_score: number;
    responsiveness_score: number;
  }>;
}

const PATTERN_COLORS = {
  balanced: '#10b981',
  dominant: '#ef4444',
  responsive: '#3b82f6',
  sporadic: '#f59e0b'
};

const PATTERN_DESCRIPTIONS = {
  balanced: 'Equal participation from all members',
  dominant: 'One participant dominates the conversation',
  responsive: 'Quick back-and-forth exchanges',
  sporadic: 'Irregular participation patterns'
};

export function TurnTakingAnalysis() {
  const [data, setData] = useState<TurnTakingMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'patterns' | 'participants' | 'health'>('patterns');
  const { scopeConversationIds, participantIndex, isFiltered } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/turnTakingAnalysis.json');
        const turnData: TurnTakingMetrics = await response.json();
        
        const scopeSet = new Set(scopeConversationIds);
        const filteredConversations = turnData.conversation_patterns
          .filter((pattern) => scopeSet.has(pattern.conversation_id))
          .map((pattern) => ({
            ...pattern,
            participants: pattern.participants.filter((participant) =>
              isKnownParticipant(participantIndex, pattern.conversation_id, participant.participant)
            ),
          }));

        turnData.conversation_patterns = filteredConversations;

        const totalParticipants = filteredConversations.reduce((sum, conv) => sum + conv.participants.length, 0);
        const avgParticipants = filteredConversations.length > 0 ? totalParticipants / filteredConversations.length : 0;
        const balancedCount = filteredConversations.filter(conv => conv.pattern.pattern_type === 'balanced').length;
        const dominantCount = filteredConversations.filter(conv => conv.pattern.pattern_type === 'dominant').length;

        // Recompute avg turn length / response time from filtered participants
        let turnLengthSum = 0;
        let turnLengthCount = 0;
        let responseTimeSum = 0;
        let responseTimeCount = 0;
        filteredConversations.forEach(conv => {
          conv.participants.forEach(participant => {
            if (participant.avg_turn_length > 0) {
              turnLengthSum += participant.avg_turn_length;
              turnLengthCount++;
            }
            if (participant.avg_response_time > 0) {
              responseTimeSum += participant.avg_response_time;
              responseTimeCount++;
            }
          });
        });

        turnData.summary = {
          total_conversations: filteredConversations.length,
          avg_participants: Math.round(avgParticipants * 10) / 10,
          balanced_conversations: balancedCount,
          dominant_speaker_conversations: dominantCount,
          avg_turn_length: turnLengthCount > 0 ? Math.round((turnLengthSum / turnLengthCount) * 10) / 10 : 0,
          avg_response_time: responseTimeCount > 0 ? Math.round(responseTimeSum / responseTimeCount) : 0
        };
        
        // Recalculate participant stats for filtered conversations
        const participantMap = new Map<string, {
          conversations: number;
          totalTurns: number;
          totalTurnLength: number;
          dominanceSum: number;
          responsivenessSum: number;
        }>();
        
        filteredConversations.forEach(conv => {
          conv.participants.forEach(participant => {
            if (!participantMap.has(participant.participant)) {
              participantMap.set(participant.participant, {
                conversations: 0,
                totalTurns: 0,
                totalTurnLength: 0,
                dominanceSum: 0,
                responsivenessSum: 0
              });
            }
            
            const stats = participantMap.get(participant.participant)!;
            stats.conversations++;
            stats.totalTurns += participant.turn_count;
            stats.totalTurnLength += participant.avg_turn_length;
            stats.dominanceSum += participant.turn_percentage;
            stats.responsivenessSum += (1 / Math.max(participant.avg_response_time, 1)) * 1000;
          });
        });
        
        turnData.participant_stats = Array.from(participantMap.entries()).map(([participant, stats]) => ({
          participant,
          conversations: stats.conversations,
          avg_turn_count: Math.round(stats.totalTurns / stats.conversations),
          avg_turn_length: Math.round((stats.totalTurnLength / stats.conversations) * 10) / 10,
          dominance_score: Math.round((stats.dominanceSum / stats.conversations) * 10) / 10,
          responsiveness_score: Math.round((stats.responsivenessSum / stats.conversations) * 10) / 10
        })).sort((a, b) => b.dominance_score - a.dominance_score);
        
        setData(turnData);
      } catch (error) {
        console.error('Error loading turn-taking data:', error);
        // Create fallback data structure
        setData({
          summary: {
            total_conversations: 0,
            avg_participants: 0,
            balanced_conversations: 0,
            dominant_speaker_conversations: 0,
            avg_turn_length: 0,
            avg_response_time: 0
          },
          conversation_patterns: [],
          participant_stats: []
        });
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [scopeConversationIds, participantIndex]);

  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading turn-taking analysis...</p>
        </div>
      </div>
    );
  }

  if (!data || data.summary.total_conversations === 0) {
    return (
      <div className="h-64 flex items-center justify-center bg-gray-50 rounded border-2 border-dashed border-gray-300">
        <div className="text-center">
          <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-500">No Turn-Taking Data Available</p>
          <p className="text-sm text-gray-400">
            {isFiltered ? 'No turn patterns found in selected conversations' : 'Turn-taking analysis data not available'}
          </p>
        </div>
      </div>
    );
  }

  const getPatternColor = (pattern: string): string => {
    return PATTERN_COLORS[pattern as keyof typeof PATTERN_COLORS] || '#6b7280';
  };

  const formatTooltip = (value: ValueType | undefined, entry: Payload<ValueType, NameType>) => {
    if (viewMode === 'patterns') {
      return [
        <div key="tooltip" className="text-sm">
          <div className="font-semibold">{entry.payload.pattern_type}</div>
          <div>Conversations: {value}</div>
          <div>Percentage: {((Number(value) / data.summary.total_conversations) * 100).toFixed(1)}%</div>
          <div className="text-xs text-gray-500 mt-1">
            {PATTERN_DESCRIPTIONS[entry.payload.pattern_type as keyof typeof PATTERN_DESCRIPTIONS]}
          </div>
        </div>
      ];
    } else {
      return [
        <div key="tooltip" className="text-sm">
          <div className="font-semibold">{entry.payload.participant}</div>
          <div>Dominance: {entry.payload.dominance_score}%</div>
          <div>Avg Turns: {entry.payload.avg_turn_count}</div>
          <div>Responsiveness: {entry.payload.responsiveness_score}</div>
        </div>
      ];
    }
  };

  // Process pattern distribution data
  const patternDistribution = Object.entries(
    data.conversation_patterns.reduce((acc, conv) => {
      const pattern = conv.pattern.pattern_type;
      acc[pattern] = (acc[pattern] || 0) + 1;
      return acc;
    }, {} as Record<string, number>)
  ).map(([pattern_type, count]) => ({
    pattern_type,
    count,
    percentage: (count / data.summary.total_conversations) * 100
  }));

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="flex items-center">
            <MessageSquare className="w-4 h-4 text-blue-600 mr-2" />
            <div className="text-blue-600 text-sm font-medium">Conversations</div>
          </div>
          <div className="text-xl font-bold text-blue-900">{data.summary.total_conversations}</div>
        </div>
        
        <div className="bg-green-50 rounded-lg p-3">
          <div className="flex items-center">
            <Users className="w-4 h-4 text-green-600 mr-2" />
            <div className="text-green-600 text-sm font-medium">Avg Participants</div>
          </div>
          <div className="text-xl font-bold text-green-900">{data.summary.avg_participants}</div>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-3">
          <div className="flex items-center">
            <TrendingUp className="w-4 h-4 text-purple-600 mr-2" />
            <div className="text-purple-600 text-sm font-medium">Balanced</div>
          </div>
          <div className="text-xl font-bold text-purple-900">{data.summary.balanced_conversations}</div>
        </div>
        
        <div className="bg-orange-50 rounded-lg p-3">
          <div className="flex items-center">
            <Clock className="w-4 h-4 text-orange-600 mr-2" />
            <div className="text-orange-600 text-sm font-medium">Avg Response</div>
          </div>
          <div className="text-xl font-bold text-orange-900">
            {data.summary.avg_response_time > 3600000 
              ? `${Math.round(data.summary.avg_response_time / 3600000)}h`
              : `${Math.round(data.summary.avg_response_time / 60000)}m`}
          </div>
        </div>
      </div>

      {/* View Mode Toggle */}
      <div className="flex justify-center">
        <div className="bg-gray-100 rounded-lg p-1 flex">
          <button
            onClick={() => setViewMode('patterns')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              viewMode === 'patterns'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            📊 Pattern Distribution
          </button>
          <button
            onClick={() => setViewMode('participants')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              viewMode === 'participants'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            👥 Participant Analysis
          </button>
          <button
            onClick={() => setViewMode('health')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              viewMode === 'health'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            💚 Conversation Health
          </button>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="h-64">
        {viewMode === 'health' ? (
          (() => {
            const healthRows = data.conversation_patterns
              .map((conv) => ({
                conversation_id: conv.conversation_id,
                label: conv.conversation_id.length > 24
                  ? `${conv.conversation_id.slice(0, 24)}…`
                  : conv.conversation_id,
                health: Math.round((conv.pattern.conversation_health ?? 0) * 100),
                healthRaw: conv.pattern.conversation_health ?? 0,
                pattern_type: conv.pattern.pattern_type,
                description: conv.pattern.description
              }))
              .sort((a, b) => b.health - a.health);

            const avgHealth = healthRows.length > 0
              ? Math.round(healthRows.reduce((sum, row) => sum + row.healthRaw, 0) / healthRows.length * 100)
              : 0;

            if (healthRows.length === 0) {
              return (
                <div className="flex items-center justify-center h-full bg-gray-50 rounded border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <p className="text-gray-500">No conversation health data</p>
                  </div>
                </div>
              );
            }

            return (
              <div className="h-full flex flex-col">
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-sm text-gray-600">Average health</span>
                  <span className="text-lg font-bold text-green-700">{avgHealth}%</span>
                </div>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={healthRows.slice(0, 12)} margin={{ top: 10, right: 20, left: 10, bottom: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey="label"
                        stroke="#6b7280"
                        fontSize={10}
                        angle={-35}
                        textAnchor="end"
                        height={50}
                        interval={0}
                      />
                      <YAxis
                        stroke="#6b7280"
                        fontSize={12}
                        domain={[0, 100]}
                        tickFormatter={(value) => `${value}%`}
                      />
                      <ChartTooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const row = payload[0].payload as {
                              conversation_id: string;
                              health: number;
                              pattern_type: string;
                              description: string;
                            };
                            return (
                              <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg text-sm">
                                <div className="font-semibold">{row.conversation_id}</div>
                                <div>Health: {row.health}%</div>
                                <div className="capitalize">Pattern: {row.pattern_type}</div>
                                {row.description && (
                                  <div className="text-xs text-gray-500 mt-1">{row.description}</div>
                                )}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar dataKey="health" radius={[4, 4, 0, 0]}>
                        {healthRows.slice(0, 12).map((entry, index) => (
                          <Cell
                            key={`health-${index}`}
                            fill={
                              entry.health >= 70 ? '#10b981' :
                              entry.health >= 40 ? '#f59e0b' :
                              '#ef4444'
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })()
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            {viewMode === 'patterns' ? (
              <BarChart data={patternDistribution} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="pattern_type"
                  stroke="#6b7280"
                  fontSize={12}
                  tickFormatter={(value) => value.charAt(0).toUpperCase() + value.slice(1)}
                />
                <YAxis
                  stroke="#6b7280"
                  fontSize={12}
                  tickFormatter={(value) => value.toLocaleString()}
                />
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
                  {patternDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getPatternColor(entry.pattern_type)} />
                  ))}
                </Bar>
              </BarChart>
            ) : (
              <BarChart data={data.participant_stats.slice(0, 10)} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="participant"
                  stroke="#6b7280"
                  fontSize={10}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                  interval={0}
                />
                <YAxis
                  stroke="#6b7280"
                  fontSize={12}
                  tickFormatter={(value) => `${value}%`}
                />
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
                <Bar dataKey="dominance_score" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            )}
          </ResponsiveContainer>
        )}
      </div>

      {/* Turn-Taking Insights */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 p-4 rounded-lg border border-green-200">
        <h4 className="font-semibold mb-3 flex items-center">
          <MessageSquare className="w-4 h-4 mr-2 text-green-600" />
          Turn-Taking Analysis Insights
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white p-3 rounded border border-green-100">
            <div className="text-sm font-medium text-gray-700">Conversation Health</div>
            <div className="text-lg font-bold text-green-600">
              {data.summary.balanced_conversations > data.summary.dominant_speaker_conversations ? 'Healthy' : 'Needs Balance'}
            </div>
            <div className="text-xs text-gray-500">
              {Math.round((data.summary.balanced_conversations / data.summary.total_conversations) * 100)}% balanced conversations
            </div>
          </div>
          <div className="bg-white p-3 rounded border border-green-100">
            <div className="text-sm font-medium text-gray-700">Most Common Pattern</div>
            <div className="text-lg font-bold text-blue-600">
              {patternDistribution[0]?.pattern_type.charAt(0).toUpperCase() + patternDistribution[0]?.pattern_type.slice(1) || 'N/A'}
            </div>
            <div className="text-xs text-gray-500">
              {patternDistribution[0]?.percentage.toFixed(1)}% of conversations
            </div>
          </div>
          <div className="bg-white p-3 rounded border border-green-100">
            <div className="text-sm font-medium text-gray-700">Most Active Participant</div>
            <div className="text-lg font-bold text-purple-600">
              {data.participant_stats[0]?.participant.split(' ')[0] || 'N/A'}
            </div>
            <div className="text-xs text-gray-500">
              {data.participant_stats[0]?.dominance_score}% dominance score
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 