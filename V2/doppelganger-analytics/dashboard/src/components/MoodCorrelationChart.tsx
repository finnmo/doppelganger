'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, LineChart, Line } from 'recharts';
import { Heart, BarChart3, Users, TrendingUp } from 'lucide-react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import type { SentimentDailyBySenderRow } from '@/lib/sentimentTimeline';

interface CorrelationPair {
  sender1: string;
  sender2: string;
  correlation: number;
  sharedDays: number;
  totalDays: number;
  strength: 'strong' | 'moderate' | 'weak' | 'none';
}

interface MoodPattern {
  sender: string;
  averageMood: number;
  moodVariability: number;
  positiveStreak: number;
  negativeStreak: number;
  moodTrend: 'improving' | 'declining' | 'stable';
  dominantEmotion: 'positive' | 'negative' | 'neutral';
}

interface MoodCorrelationData {
  summary: {
    totalParticipants: number;
    totalCorrelations: number;
    strongCorrelations: number;
    averageCorrelation: number;
    dateRange: {
      start: string;
      end: string;
    };
  };
  correlationMatrix: CorrelationPair[];
  moodPatterns: MoodPattern[];
  timeSeriesData: Array<{
    date: string;
    participants: Record<string, number>;
  }>;
}

interface RawSentimentData {
  sender: string;
  conversation_id: string;
  avg_sentiment: number;
  message_count: number;
  avg_positive: number;
  avg_negative: number;
  avg_neutral: number;
}

const calculateVariability = (values: number[]): number => {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
};

const getCorrelationStrength = (correlation: number): 'strong' | 'moderate' | 'weak' | 'none' => {
  const abs = Math.abs(correlation);
  if (abs >= 0.7) return 'strong';
  if (abs >= 0.4) return 'moderate';
  if (abs >= 0.2) return 'weak';
  return 'none';
};

function pearsonCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length < 2) return 0;
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
  const num = n * sumXY - sumX * sumY;
  const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  return den === 0 ? 0 : num / den;
}

function buildFilteredMoodData(
  sentimentData: RawSentimentData[],
  dailySenderRows: SentimentDailyBySenderRow[],
  selectedConversations: string[]
): MoodCorrelationData {
  const selected = new Set(selectedConversations);
  const uniqueSenders = [...new Set(sentimentData.map(item => item.sender))];

  const dateGroups = new Map<string, Map<string, { sum: number; count: number }>>();
  for (const row of dailySenderRows) {
    if (selected.size > 0 && !selected.has(row.conversation_id)) continue;
    let day = dateGroups.get(row.date);
    if (!day) {
      day = new Map();
      dateGroups.set(row.date, day);
    }
    const existing = day.get(row.sender) ?? { sum: 0, count: 0 };
    existing.sum += row.compoundSum;
    existing.count += row.messageCount;
    day.set(row.sender, existing);
  }

  const senderDaily = new Map<string, Map<string, number>>();
  for (const [date, senders] of dateGroups.entries()) {
    for (const [sender, vals] of senders.entries()) {
      if (!senderDaily.has(sender)) senderDaily.set(sender, new Map());
      senderDaily.get(sender)!.set(date, vals.sum / vals.count);
    }
  }

  const correlationMatrix: CorrelationPair[] = [];
  for (let i = 0; i < uniqueSenders.length; i++) {
    for (let j = i + 1; j < uniqueSenders.length; j++) {
      const s1 = uniqueSenders[i];
      const s2 = uniqueSenders[j];
      const dates1 = senderDaily.get(s1);
      const dates2 = senderDaily.get(s2);
      if (!dates1 || !dates2) continue;
      const sharedDates = [...dates1.keys()].filter(d => dates2.has(d)).sort();
      if (sharedDates.length < 3) continue;
      const x = sharedDates.map(d => dates1.get(d)!);
      const y = sharedDates.map(d => dates2.get(d)!);
      const correlation = pearsonCorrelation(x, y);
      correlationMatrix.push({
        sender1: s1,
        sender2: s2,
        correlation: Math.round(correlation * 1000) / 1000,
        sharedDays: sharedDates.length,
        totalDays: Math.max(dates1.size, dates2.size),
        strength: getCorrelationStrength(correlation)
      });
    }
  }

  const moodPatterns: MoodPattern[] = uniqueSenders.map(sender => {
    const senderData = sentimentData.filter(item => item.sender === sender);
    const totalMessages = senderData.reduce((sum, item) => sum + item.message_count, 0) || 1;
    const avgMood = senderData.reduce((sum, item) => sum + (item.avg_sentiment * item.message_count), 0) / totalMessages;
    const dailyVals = [...(senderDaily.get(sender)?.values() ?? [])];
    const moodVariability = calculateVariability(dailyVals.length > 1 ? dailyVals : [avgMood]);
    return {
      sender,
      averageMood: Math.round(avgMood * 1000) / 1000,
      moodVariability: Math.round(moodVariability * 1000) / 1000,
      positiveStreak: 0,
      negativeStreak: 0,
      moodTrend: avgMood > 0.05 ? 'improving' : avgMood < -0.05 ? 'declining' : 'stable',
      dominantEmotion: avgMood > 0.1 ? 'positive' : avgMood < -0.1 ? 'negative' : 'neutral'
    };
  });

  const timeSeriesData = [...dateGroups.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, senders]) => ({
      date,
      participants: Object.fromEntries(
        [...senders.entries()].map(([s, v]) => [s, Math.round((v.sum / v.count) * 1000) / 1000])
      )
    }));

  const dates = timeSeriesData.map(d => d.date);
  const strongCorrelations = correlationMatrix.filter(c => c.strength === 'strong').length;
  const averageCorrelation = correlationMatrix.length > 0
    ? correlationMatrix.reduce((sum, c) => sum + Math.abs(c.correlation), 0) / correlationMatrix.length
    : 0;

  return {
    summary: {
      totalParticipants: uniqueSenders.length,
      totalCorrelations: correlationMatrix.length,
      strongCorrelations,
      averageCorrelation: Math.round(averageCorrelation * 1000) / 1000,
      dateRange: { start: dates[0] || 'N/A', end: dates[dates.length - 1] || 'N/A' }
    },
    correlationMatrix,
    moodPatterns,
    timeSeriesData
  };
}

export default function MoodCorrelationChart() {
  const [data, setData] = useState<MoodCorrelationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'correlations' | 'patterns' | 'timeline'>('timeline');
  const { selectedConversations, isFiltered } = useConversationFilter();

  const processFilteredMoodData = useCallback((
    sentimentData: RawSentimentData[],
    dailySenderRows: SentimentDailyBySenderRow[],
    selectedConversations: string[]
  ): MoodCorrelationData => buildFilteredMoodData(sentimentData, dailySenderRows, selectedConversations), []);

  useEffect(() => {
    const loadData = async () => {
      try {
        if (!isFiltered || selectedConversations.length === 0) {
          const response = await fetch('/data/moodCorrelationMetrics.json');
          const correlationData: MoodCorrelationData = await response.json();
          setData(correlationData);
        } else {
          const [sentimentResponse, dailySenderResponse] = await Promise.all([
            fetch('/data/sentimentBySender.json'),
            fetch('/data/sentimentDailyBySender.json')
          ]);
          let sentimentData: RawSentimentData[] = await sentimentResponse.json();
          const dailySenderRows: SentimentDailyBySenderRow[] = dailySenderResponse.ok
            ? await dailySenderResponse.json()
            : [];

          sentimentData = sentimentData.filter(item =>
            selectedConversations.includes(item.conversation_id)
          );

          if (sentimentData.length === 0) {
            setData(null);
            return;
          }

          setData(processFilteredMoodData(sentimentData, dailySenderRows, selectedConversations));
        }
      } catch (error) {
        console.error('Error loading mood correlation data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered, processFilteredMoodData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-red-600"></div>
          <span className="text-gray-600">Loading mood correlation analysis...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-gray-500">
          <Users className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <div>
            <div className="text-lg mb-2">📊 No Correlation Data</div>
            <div>
              {isFiltered 
                ? 'No mood correlation data available for selected conversations'
                : 'No mood correlation data found'
              }
            </div>
          </div>
        </div>
      </div>
    );
  }

  const getCorrelationColor = (strength: string) => {
    switch (strength) {
      case 'strong': return '#ef4444';
      case 'moderate': return '#f59e0b';
      case 'weak': return '#6b7280';
      default: return '#d1d5db';
    }
  };

  const getMoodColor = (mood: number) => {
    if (mood > 0.1) return '#10b981';
    if (mood < -0.1) return '#ef4444';
    return '#6b7280';
  };

  const formatPatternTooltip = (data: MoodPattern) => {
    return (
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{data.sender}</div>
        <div>Average Mood: {data.averageMood.toFixed(3)}</div>
        <div>Variability: {data.moodVariability.toFixed(3)}</div>
        <div>Trend: {data.moodTrend}</div>
        <div>Dominant: {data.dominantEmotion}</div>
      </div>
    );
  };

  const timelineChartData = data.timeSeriesData.map(point => ({
    date: point.date,
    ...point.participants
  }));
  const timelineSenders = data.moodPatterns.map(p => p.sender).slice(0, 6);
  const LINE_COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#6b7280'];
  const correlationBarData = data.correlationMatrix.map(c => ({
    label: `${c.sender1} ↔ ${c.sender2}`,
    correlation: c.correlation,
    strength: c.strength,
    sender1: c.sender1,
    sender2: c.sender2,
    sharedDays: c.sharedDays
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 text-sm">
        <div className="bg-blue-50 p-3 rounded">
          <div className="text-blue-800 font-medium">{data.summary.totalParticipants}</div>
          <div className="text-blue-600">Participants</div>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <div className="text-green-800 font-medium">{data.summary.totalCorrelations}</div>
          <div className="text-green-600">Correlations</div>
        </div>
        <div className="bg-purple-50 p-3 rounded">
          <div className="text-purple-800 font-medium">{data.summary.strongCorrelations}</div>
          <div className="text-purple-600">Strong Links</div>
        </div>
        <div className="bg-red-50 p-3 rounded">
          <div className="text-red-800 font-medium">{data.summary.averageCorrelation.toFixed(3)}</div>
          <div className="text-red-600">Avg Correlation</div>
        </div>
      </div>

      <div className="flex bg-gray-100 rounded-lg p-1 w-fit">
          <button
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              viewMode === 'timeline'
              ? 'bg-white text-red-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          onClick={() => setViewMode('timeline')}
        >
          <TrendingUp className="w-4 h-4 inline mr-1" />
          Daily Moods
          </button>
          <button
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              viewMode === 'correlations'
              ? 'bg-white text-red-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          onClick={() => setViewMode('correlations')}
          >
          <Heart className="w-4 h-4 inline mr-1" />
          Correlations
          </button>
          <button
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              viewMode === 'patterns'
              ? 'bg-white text-red-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          onClick={() => setViewMode('patterns')}
        >
          <BarChart3 className="w-4 h-4 inline mr-1" />
          Mood Patterns
          </button>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'timeline' ? (
            timelineChartData.length > 0 ? (
              <LineChart data={timelineChartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" fontSize={10} tickFormatter={(v, i) => i % Math.ceil(timelineChartData.length / 8) === 0 ? v : ''} />
                <YAxis domain={[-1, 1]} stroke="#6b7280" fontSize={12} />
                <Tooltip />
                {timelineSenders.map((sender, i) => (
                  <Line key={sender} type="monotone" dataKey={sender} stroke={LINE_COLORS[i % LINE_COLORS.length]} strokeWidth={2} dot={false} connectNulls />
                ))}
              </LineChart>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">No daily mood data</div>
            )
          ) : viewMode === 'correlations' ? (
            correlationBarData.length > 0 ? (
              <BarChart data={correlationBarData} layout="vertical" margin={{ top: 20, right: 30, left: 80, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" domain={[-1, 1]} stroke="#6b7280" fontSize={12} />
                <YAxis type="category" dataKey="label" stroke="#6b7280" fontSize={11} width={75} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
                  {correlationBarData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getCorrelationColor(entry.strength)} />
                  ))}
                </Bar>
              </BarChart>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500 text-center px-4">
                Need at least 3 shared days between two participants for correlation
              </div>
            )
          ) : (
            <BarChart data={data.moodPatterns} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="sender" 
                stroke="#6b7280"
                fontSize={12}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis 
                stroke="#6b7280"
                fontSize={12}
                domain={[-1, 1]}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <Tooltip formatter={(value, name, props) => formatPatternTooltip(props.payload)} />
              <Bar dataKey="averageMood" radius={[2, 2, 0, 0]}>
                {data.moodPatterns.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getMoodColor(entry.averageMood)} />
                ))}
              </Bar>
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      <div className="text-xs text-gray-600">
        {viewMode === 'correlations' ? (
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Strong (≥0.7)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span>Moderate (≥0.4)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-500"></div>
              <span>Weak (≥0.2)</span>
            </div>
          </div>
        ) : viewMode === 'patterns' ? (
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>Positive Mood</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Negative Mood</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-500"></div>
              <span>Neutral Mood</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap gap-4">
            {timelineSenders.map((sender, i) => (
              <div key={sender} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: LINE_COLORS[i % LINE_COLORS.length] }} />
                <span>{sender}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: CorrelationPair }>;
}

const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
        <div className="font-semibold mb-2">{data.sender1} ↔ {data.sender2}</div>
        <div className="space-y-1 text-sm">
          <p><strong>Correlation:</strong> {data.correlation.toFixed(3)}</p>
          <p><strong>Strength:</strong> {data.strength}</p>
          <p><strong>Shared Days:</strong> {data.sharedDays}</p>
        </div>
      </div>
    );
  }
  return null;
}; 