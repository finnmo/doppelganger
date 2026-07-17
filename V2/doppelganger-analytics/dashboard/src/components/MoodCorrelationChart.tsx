'use client';

import React, { useState, useEffect } from 'react';
import { ParticipantToggleChips, allParticipantsActive, toggleParticipant } from '@/components/ParticipantToggleChips';
import { XAxis, YAxis, CartesianGrid, ResponsiveContainer, BarChart, Bar, Cell, LineChart, Line } from 'recharts';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { Heart, BarChart3, Users, TrendingUp } from 'lucide-react';
import { CHART_AREA } from '@/lib/layout';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import {
  buildFilteredMoodData,
  type CorrelationPair,
  type MoodCorrelationData,
  type MoodPattern,
  type RawSentimentData,
} from '@/lib/moodCorrelation';
import type { SentimentDailyBySenderRow } from '@/lib/sentimentTimeline';

export default function MoodCorrelationChart() {
  const [data, setData] = useState<MoodCorrelationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'correlations' | 'patterns' | 'timeline'>('timeline');
  const [activeParticipants, setActiveParticipants] = useState<Set<string>>(new Set());
  const { conversations, filterScopedRows, isFiltered, scopeConversationIds } =
    useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const [sentimentResponse, dailySenderResponse] = await Promise.all([
          fetch('/data/sentimentBySender.json'),
          fetch('/data/sentimentDailyBySender.json'),
        ]);
        let sentimentData: RawSentimentData[] = await sentimentResponse.json();
        const dailySenderRows: SentimentDailyBySenderRow[] = dailySenderResponse.ok
          ? await dailySenderResponse.json()
          : [];

        sentimentData = filterScopedRows(sentimentData, { senderKey: 'sender' });
        const scopedDaily = filterScopedRows(dailySenderRows, { senderKey: 'sender' });

        if (sentimentData.length === 0) {
          setData(null);
          return;
        }

        setData(
          buildFilteredMoodData(
            sentimentData,
            scopedDaily,
            scopeConversationIds,
            conversations
          )
        );
      } catch (error) {
        console.error('Error loading mood correlation data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [conversations, filterScopedRows, isFiltered, scopeConversationIds]);

  React.useEffect(() => {
    if (!data) return;
    setActiveParticipants(allParticipantsActive(data.moodPatterns.map((p) => p.sender)));
  }, [data]);

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
  const visibleTimelineSenders = timelineSenders.filter((s) => activeParticipants.has(s));
  const visibleCorrelations = data.correlationMatrix.filter(
    (c) => activeParticipants.has(c.sender1) && activeParticipants.has(c.sender2)
  );
  const LINE_COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#6b7280'];
  const correlationBarData = visibleCorrelations.map(c => ({
    label: `${c.sender1} ↔ ${c.sender2}`,
    correlation: c.correlation,
    strength: c.strength,
    sender1: c.sender1,
    sender2: c.sender2,
    sharedDays: c.sharedDays
  }));

  return (
    <div className="flex h-full min-h-0 flex-col space-y-3">
      <ParticipantToggleChips
        participants={timelineSenders}
        active={activeParticipants}
        onToggle={(name) =>
          setActiveParticipants((prev) => toggleParticipant(prev, name, timelineSenders))
        }
      />
      <div className="grid shrink-0 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 text-sm">
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

      <div className="flex shrink-0 bg-gray-100 rounded-lg p-1 w-fit">
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

      <div className={CHART_AREA}>
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'timeline' ? (
            timelineChartData.length > 0 ? (
              <LineChart data={timelineChartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="date" stroke="#6b7280" fontSize={10} tickFormatter={(v, i) => i % Math.ceil(timelineChartData.length / 8) === 0 ? v : ''} />
                <YAxis domain={[-1, 1]} stroke="#6b7280" fontSize={12} />
                <ChartTooltip />
                {visibleTimelineSenders.map((sender, i) => (
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
                <ChartTooltip content={<CustomTooltip />} />
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
              <ChartTooltip formatter={(value, name, props) => formatPatternTooltip(props.payload)} />
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
            {visibleTimelineSenders.map((sender, i) => (
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