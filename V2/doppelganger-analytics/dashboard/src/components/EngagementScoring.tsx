'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Cell } from 'recharts';
import { Users, TrendingUp, Award, MessageCircle, Zap } from 'lucide-react';

interface EngagementScore {
  participant: string;
  overall_score: number;
  message_frequency: number;
  response_speed: number;
  conversation_initiation: number;
  message_length: number;
  consistency: number;
  social_connectivity: number;
  conversations_count: number;
  total_messages: number;
  avg_response_time: number;
  engagement_tier: 'high' | 'medium' | 'low';
}

interface ConversationEngagement {
  conversation_id: string;
  participants: EngagementScore[];
  avg_engagement: number;
  engagement_distribution: {
    high: number;
    medium: number;
    low: number;
  };
}

interface EngagementMetrics {
  summary: {
    total_participants: number;
    avg_engagement_score: number;
    high_engagement_participants: number;
    most_engaged_participant: string;
    least_engaged_participant: string;
    engagement_variance: number;
  };
  participant_scores: EngagementScore[];
  conversation_engagement: ConversationEngagement[];
  engagement_trends: {
    participant: string;
    scores_over_time: { period: string; score: number }[];
  }[];
}

const COLORS = {
  high: '#10b981',
  medium: '#f59e0b', 
  low: '#ef4444'
};

const TIER_COLORS = ['#10b981', '#f59e0b', '#ef4444'];

export function EngagementScoring() {
  const [data, setData] = useState<EngagementMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | 'radar'>('overview');
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const response = await fetch('/data/engagementScoring.json');
        if (!response.ok) {
          throw new Error('Failed to load engagement scoring data');
        }
        const jsonData = await response.json();
        setData(jsonData);
      } catch (err) {
        console.error('Error loading engagement scoring data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const filteredData = React.useMemo(() => {
    if (!data) return data;

    // Unfiltered: use global participant_scores as emitted
    if (!isFiltered || selectedConversations.length === 0) {
      return data;
    }

    const filteredConversationEngagement = data.conversation_engagement.filter(
      conv => selectedConversations.includes(conv.conversation_id)
    );

    // Aggregate only from per-conversation participant rows — never fall back to global participant_scores
    const participantMap = new Map<string, {
      total_score: number;
      conversations: number;
      total_messages: number;
      total_response_time: number;
      response_count: number;
      message_frequency_sum: number;
      response_speed_sum: number;
      social_connectivity_sum: number;
    }>();

    filteredConversationEngagement.forEach(conv => {
      conv.participants.forEach(participant => {
        if (!participantMap.has(participant.participant)) {
          participantMap.set(participant.participant, {
            total_score: 0,
            conversations: 0,
            total_messages: 0,
            total_response_time: 0,
            response_count: 0,
            message_frequency_sum: 0,
            response_speed_sum: 0,
            social_connectivity_sum: 0
          });
        }

        const stats = participantMap.get(participant.participant)!;
        stats.total_score += participant.overall_score;
        stats.conversations++;
        // Use conversation-scoped total_messages only (0 if missing — do not use global scores)
        stats.total_messages += participant.total_messages ?? 0;
        stats.message_frequency_sum += participant.message_frequency ?? 0;
        stats.response_speed_sum += participant.response_speed ?? 0;
        stats.social_connectivity_sum += participant.social_connectivity ?? 0;
        if (participant.avg_response_time > 0) {
          stats.total_response_time += participant.avg_response_time;
          stats.response_count++;
        }
      });
    });

    const filteredParticipantScores = Array.from(participantMap.entries()).map(([participant, stats]) => {
      const avgScore = stats.conversations > 0 ? stats.total_score / stats.conversations : 0;
      const avgResponseTime = stats.response_count > 0 ? stats.total_response_time / stats.response_count : 0;

      return {
        participant,
        overall_score: Math.round(avgScore * 10) / 10,
        message_frequency: stats.conversations > 0
          ? Math.round((stats.message_frequency_sum / stats.conversations) * 10) / 10
          : 0,
        response_speed: stats.conversations > 0
          ? Math.round((stats.response_speed_sum / stats.conversations) * 10) / 10
          : 0,
        conversation_initiation: 0,
        message_length: 0,
        consistency: 0,
        social_connectivity: stats.conversations > 0
          ? Math.round((stats.social_connectivity_sum / stats.conversations) * 10) / 10
          : stats.conversations,
        conversations_count: stats.conversations,
        total_messages: stats.total_messages,
        avg_response_time: Math.round(avgResponseTime),
        engagement_tier: (avgScore >= 70 ? 'high' : avgScore >= 40 ? 'medium' : 'low') as EngagementScore['engagement_tier']
      };
    }).sort((a, b) => b.overall_score - a.overall_score);

    const highEngagementCount = filteredParticipantScores.filter(p => p.engagement_tier === 'high').length;
    const avgEngagement = filteredParticipantScores.length > 0
      ? filteredParticipantScores.reduce((sum, p) => sum + p.overall_score, 0) / filteredParticipantScores.length
      : 0;

    return {
      ...data,
      summary: {
        ...data.summary,
        total_participants: filteredParticipantScores.length,
        avg_engagement_score: Math.round(avgEngagement * 10) / 10,
        high_engagement_participants: highEngagementCount,
        most_engaged_participant: filteredParticipantScores[0]?.participant || '',
        least_engaged_participant: filteredParticipantScores[filteredParticipantScores.length - 1]?.participant || ''
      },
      participant_scores: filteredParticipantScores,
      conversation_engagement: filteredConversationEngagement
    };
  }, [data, selectedConversations, isFiltered]);

  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <Award className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Participant Engagement Scores</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading engagement data...</div>
        </div>
      </div>
    );
  }

  if (error || !filteredData) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <Award className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold">Participant Engagement Scores</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <MessageCircle className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">No Engagement Data Available</p>
            <p className="text-sm text-gray-400">Engagement scoring data could not be loaded</p>
          </div>
        </div>
      </div>
    );
  }

  const topParticipants = filteredData.participant_scores.slice(0, 10);
  const messageTotal = filteredData.participant_scores.reduce((sum, p) => sum + p.total_messages, 0);
  const engagementDistribution = [
    { tier: 'High', count: filteredData.participant_scores.filter(p => p.engagement_tier === 'high').length, color: COLORS.high },
    { tier: 'Medium', count: filteredData.participant_scores.filter(p => p.engagement_tier === 'medium').length, color: COLORS.medium },
    { tier: 'Low', count: filteredData.participant_scores.filter(p => p.engagement_tier === 'low').length, color: COLORS.low }
  ];

  const radarData = topParticipants.slice(0, 5).map(participant => ({
    participant: participant.participant.length > 15 ? participant.participant.substring(0, 15) + '...' : participant.participant,
    'Message Frequency': participant.message_frequency,
    'Response Speed': participant.response_speed,
    'Social Connectivity': participant.social_connectivity,
    'Overall Score': participant.overall_score
  }));

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Award className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Participant Engagement Scores</h3>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('overview')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'overview' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setViewMode('detailed')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'detailed' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Detailed
          </button>
          <button
            onClick={() => setViewMode('radar')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'radar' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Radar
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-600 font-medium">Total Participants</span>
          </div>
          <div className="text-2xl font-bold text-blue-900 mt-1">
            {filteredData.summary.total_participants}
          </div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-green-600" />
            <span className="text-sm text-green-600 font-medium">Avg Engagement</span>
          </div>
          <div className="text-2xl font-bold text-green-900 mt-1">
            {filteredData.summary.avg_engagement_score}
          </div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-purple-600" />
            <span className="text-sm text-purple-600 font-medium">High Engagement</span>
          </div>
          <div className="text-2xl font-bold text-purple-900 mt-1">
            {filteredData.summary.high_engagement_participants}
          </div>
        </div>
        
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Award className="w-4 h-4 text-orange-600" />
            <span className="text-sm text-orange-600 font-medium">Top Participant</span>
          </div>
          <div className="text-sm font-bold text-orange-900 mt-1 truncate">
            {filteredData.summary.most_engaged_participant}
          </div>
        </div>
      </div>

      {viewMode === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Participants */}
          <div>
            <h4 className="text-md font-semibold mb-3">Top Engaged Participants</h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topParticipants}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="participant" 
                    angle={-45}
                    textAnchor="end"
                    height={80}
                    fontSize={12}
                    tickFormatter={(value) => value.length > 12 ? value.substring(0, 12) + '...' : value}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value) => [value, 'Engagement Score']}
                    labelFormatter={(label) => `Participant: ${label}`}
                  />
                  <Bar dataKey="overall_score" radius={[4, 4, 0, 0]}>
                    {topParticipants.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[entry.engagement_tier]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Engagement Distribution */}
          <div>
            <h4 className="text-md font-semibold mb-3">Engagement Distribution</h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={engagementDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="tier" />
                  <YAxis />
                  <Tooltip 
                    formatter={(value) => [value, 'Participants']}
                    labelFormatter={(label) => `${label} Engagement`}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {engagementDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {viewMode === 'detailed' && (
        <div>
          <h4 className="text-md font-semibold mb-3">Detailed Engagement Metrics</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Participant
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Overall Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Messages
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Contribution
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Conversations
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg Response Time
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Tier
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredData.participant_scores.slice(0, 20).map((participant, index) => (
                  <tr key={participant.participant} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {participant.participant}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      <div className="flex items-center">
                        <div className="text-sm font-medium">{participant.overall_score}</div>
                        <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                          <div 
                            className="h-2 rounded-full" 
                            style={{ 
                              width: `${Math.min(participant.overall_score, 100)}%`,
                              backgroundColor: COLORS[participant.engagement_tier]
                            }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {participant.total_messages.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {messageTotal > 0
                        ? `${Math.round((participant.total_messages / messageTotal) * 1000) / 10}%`
                        : '0%'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {participant.conversations_count}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {participant.avg_response_time > 0 
                        ? `${Math.round(participant.avg_response_time / 1000 / 60)}m` 
                        : 'N/A'
                      }
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        participant.engagement_tier === 'high' ? 'bg-green-100 text-green-800' :
                        participant.engagement_tier === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {participant.engagement_tier.toUpperCase()}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {viewMode === 'radar' && (
        <div>
          <h4 className="text-md font-semibold mb-3">Multi-dimensional Engagement Analysis</h4>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="participant" fontSize={12} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} fontSize={10} />
                {topParticipants.slice(0, 3).map((participant, index) => (
                  <Radar
                    key={participant.participant}
                    name={participant.participant.length > 15 ? participant.participant.substring(0, 15) + '...' : participant.participant}
                    dataKey={participant.participant.length > 15 ? participant.participant.substring(0, 15) + '...' : participant.participant}
                    stroke={TIER_COLORS[index]}
                    fill={TIER_COLORS[index]}
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                ))}
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Insights Panel */}
      <div className="mt-6 bg-gray-50 p-4 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">💡 Engagement Insights</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div>
            <strong>Most Engaged:</strong> {filteredData.summary.most_engaged_participant} leads with high participation
          </div>
          <div>
            <strong>Engagement Balance:</strong> {Math.round((filteredData.summary.high_engagement_participants / filteredData.summary.total_participants) * 100)}% of participants show high engagement
          </div>
          <div>
            <strong>Average Score:</strong> {filteredData.summary.avg_engagement_score}/100 indicates {filteredData.summary.avg_engagement_score >= 70 ? 'strong' : filteredData.summary.avg_engagement_score >= 40 ? 'moderate' : 'low'} overall engagement
          </div>
          <div>
            <strong>Participation:</strong> {filteredData.summary.total_participants} active participants across conversations
          </div>
        </div>
      </div>
    </div>
  );
} 