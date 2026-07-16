'use client';

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import type { Payload, ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { TrendingUp, TrendingDown, Calendar, User, BarChart3 } from 'lucide-react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import {
  buildTimelineFromDailyRows,
  buildSenderTimelineFromDailyRows,
  summarizeTimeline,
  formatFullDate,
  type SentimentDailyRow,
  type SentimentDailyBySenderRow
} from '@/lib/sentimentTimeline';

interface SentimentTimePoint {
  date: string;
  timestamp: number;
  avgCompound: number;
  avgPositive: number;
  avgNegative: number;
  avgNeutral: number;
  messageCount: number;
  sentiment: 'positive' | 'negative' | 'neutral';
}

interface SentimentBySender {
  sender: string;
  timeSeries: SentimentTimePoint[];
  overallSentiment: {
    avgCompound: number;
    avgPositive: number;
    avgNegative: number;
    avgNeutral: number;
    totalMessages: number;
  };
}

interface SentimentTimelineData {
  summary: {
    totalDays: number;
    totalMessages: number;
    uniqueSenders: number;
    dateRange: {
      start: string;
      end: string;
    };
    avgDailySentiment: number;
    mostPositiveDay: SentimentTimePoint;
    mostNegativeDay: SentimentTimePoint;
  };
  overallTimeline: SentimentTimePoint[];
  senderTimelines: SentimentBySender[];
}

// Raw sentiment data interface for filtering
interface RawSentimentData {
  sender: string;
  conversation_id: string;
  avg_sentiment: number;
  message_count: number;
  avg_positive: number;
  avg_negative: number;
  avg_neutral: number;
}

const SentimentTimelineChart: React.FC = () => {
  const [data, setData] = useState<SentimentTimelineData | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'overall' | 'sender'>('overall');
  const [selectedSender, setSelectedSender] = useState<string>('');
  const [chartType, setChartType] = useState<'line' | 'area'>('area');
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        if (!isFiltered || selectedConversations.length === 0) {
          // Use pre-computed data when no filtering
          const response = await fetch('/data/sentimentTimelineMetrics.json');
          const timelineData: SentimentTimelineData = await response.json();
          setData(timelineData);
          if (timelineData.senderTimelines.length > 0) {
            setSelectedSender(timelineData.senderTimelines[0].sender);
          }
        } else {
          const [dailyResponse, dailySenderResponse, sentimentResponse] = await Promise.all([
            fetch('/data/sentimentDailyByConversation.json'),
            fetch('/data/sentimentDailyBySender.json'),
            fetch('/data/sentimentBySender.json')
          ]);

          const dailyRows: SentimentDailyRow[] = dailyResponse.ok
            ? await dailyResponse.json()
            : [];
          const dailySenderRows: SentimentDailyBySenderRow[] = dailySenderResponse.ok
            ? await dailySenderResponse.json()
            : [];

          let sentimentData: RawSentimentData[] = await sentimentResponse.json();
          sentimentData = sentimentData.filter(item =>
            selectedConversations.includes(item.conversation_id)
          );

          const overallTimeline = buildTimelineFromDailyRows(
            Array.isArray(dailyRows) ? dailyRows : [],
            selectedConversations
          );

          if (sentimentData.length === 0 && overallTimeline.length === 0) {
            setData(null);
            return;
          }

          const processedData = processFilteredSentimentData(
            sentimentData,
            overallTimeline,
            Array.isArray(dailySenderRows) ? dailySenderRows : [],
            selectedConversations
          );
          setData(processedData);
          if (processedData.senderTimelines.length > 0) {
            setSelectedSender(processedData.senderTimelines[0].sender);
          }
        }
      } catch (error) {
        console.error('Error loading sentiment timeline data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  const processFilteredSentimentData = (
    sentimentData: RawSentimentData[],
    overallTimeline: SentimentTimePoint[],
    dailySenderRows: SentimentDailyBySenderRow[],
    selectedConversations: string[]
  ): SentimentTimelineData => {
    const uniqueSenders = [...new Set(sentimentData.map(item => item.sender))];

    const senderTimelines: SentimentBySender[] = uniqueSenders.map(sender => {
      const senderData = sentimentData.filter(item => item.sender === sender);
      const totalMessages = senderData.reduce((sum, item) => sum + item.message_count, 0) || 1;
      const avgCompound = senderData.reduce((sum, item) => sum + (item.avg_sentiment * item.message_count), 0) / totalMessages;
      const avgPositive = senderData.reduce((sum, item) => sum + (item.avg_positive * item.message_count), 0) / totalMessages;
      const avgNegative = senderData.reduce((sum, item) => sum + (item.avg_negative * item.message_count), 0) / totalMessages;
      const avgNeutral = senderData.reduce((sum, item) => sum + (item.avg_neutral * item.message_count), 0) / totalMessages;

      const timeSeries = dailySenderRows.length > 0
        ? buildSenderTimelineFromDailyRows(dailySenderRows, selectedConversations, sender)
        : [];

      return {
        sender,
        timeSeries,
        overallSentiment: {
          avgCompound: Math.round(avgCompound * 1000) / 1000,
          avgPositive: Math.round(avgPositive * 1000) / 1000,
          avgNegative: Math.round(avgNegative * 1000) / 1000,
          avgNeutral: Math.round(avgNeutral * 1000) / 1000,
          totalMessages: senderData.reduce((sum, item) => sum + item.message_count, 0)
        }
      };
    });

    const timelineSummary = summarizeTimeline(overallTimeline);
    const fallbackTotal = sentimentData.reduce((sum, item) => sum + item.message_count, 0);
    const fallbackAvg = fallbackTotal > 0
      ? sentimentData.reduce((sum, item) => sum + (item.avg_sentiment * item.message_count), 0) / fallbackTotal
      : 0;
    const emptyDay: SentimentTimePoint = {
      date: 'N/A', timestamp: 0, avgCompound: 0, avgPositive: 0, avgNegative: 0, avgNeutral: 0, messageCount: 0, sentiment: 'neutral'
    };

    return {
      summary: {
        totalDays: timelineSummary.totalDays,
        totalMessages: timelineSummary.totalMessages || fallbackTotal,
        uniqueSenders: uniqueSenders.length,
        dateRange: timelineSummary.dateRange,
        avgDailySentiment: overallTimeline.length > 0
          ? timelineSummary.avgDailySentiment
          : Math.round(fallbackAvg * 1000) / 1000,
        mostPositiveDay: timelineSummary.mostPositiveDay ?? emptyDay,
        mostNegativeDay: timelineSummary.mostNegativeDay ?? emptyDay
      },
      overallTimeline,
      senderTimelines
    };
  };

  const formatDate = formatFullDate;

  const getSentimentIcon = (sentiment: number) => {
    if (sentiment > 0.1) return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (sentiment < -0.1) return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <BarChart3 className="w-4 h-4 text-gray-500" />;
  };

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<Payload<ValueType, NameType>> }) => {
    if (active && payload && payload.length > 0) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <Calendar className="w-4 h-4" />
            <span className="font-semibold">{formatDate(data.date)}</span>
            {getSentimentIcon(data.avgCompound)}
          </div>
          <div className="space-y-1 text-sm">
            <p><strong>Sentiment Score:</strong> {data.avgCompound.toFixed(3)}</p>
            <p><strong>Messages:</strong> {data.messageCount.toLocaleString()}</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-2 text-xs">
              <div className="text-green-600">
                <span className="font-medium">Positive:</span> {(data.avgPositive * 100).toFixed(1)}%
              </div>
              <div className="text-red-600">
                <span className="font-medium">Negative:</span> {(data.avgNegative * 100).toFixed(1)}%
              </div>
              <div className="text-gray-600">
                <span className="font-medium">Neutral:</span> {(data.avgNeutral * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
        <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  if (!data || data.overallTimeline.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-gray-500">
          <div className="text-lg mb-2">📊 No Timeline Data</div>
          <div>
            {isFiltered 
              ? 'No sentiment timeline data available for selected conversations'
              : 'No sentiment timeline data available'
            }
          </div>
        </div>
      </div>
    );
  }

  const currentData = viewMode === 'overall' 
    ? data.overallTimeline 
    : data.senderTimelines.find(s => s.sender === selectedSender)?.timeSeries || [];

  if (viewMode === 'sender' && currentData.length === 0) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-4 text-sm">
          <div className="bg-blue-50 p-3 rounded">
            <div className="text-blue-800 font-medium">{data.summary.totalDays}</div>
            <div className="text-blue-600">Days Analyzed</div>
          </div>
          <div className="bg-green-50 p-3 rounded">
            <div className="text-green-800 font-medium">{data.summary.avgDailySentiment.toFixed(3)}</div>
            <div className="text-green-600">Avg Sentiment</div>
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <div className="text-purple-800 font-medium">{formatDate(data.summary.mostPositiveDay.date)}</div>
            <div className="text-purple-600">Most Positive Day</div>
          </div>
          <div className="bg-red-50 p-3 rounded">
            <div className="text-red-800 font-medium">{formatDate(data.summary.mostNegativeDay.date)}</div>
            <div className="text-red-600">Most Negative Day</div>
          </div>
        </div>
        <div className="flex items-center justify-center h-64 text-gray-500">
          <div className="text-center">
            <div className="text-lg mb-2">No per-sender timeline</div>
            <div className="text-sm">Regenerate metrics to include daily sender data, or try another sender.</div>
          </div>
        </div>
      </div>
    );
  }

  // Prepare chart data with formatted dates
  const chartData = currentData.map(point => ({
    ...point,
    formattedDate: formatDate(point.date),
    displayDate: new Date(point.date).toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' })
  }));

  return (
    <div className="space-y-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-4 text-sm">
          <div className="bg-blue-50 p-3 rounded">
            <div className="text-blue-800 font-medium">{data.summary.totalDays}</div>
            <div className="text-blue-600">Days Analyzed</div>
          </div>
          <div className="bg-green-50 p-3 rounded">
            <div className="text-green-800 font-medium">{data.summary.avgDailySentiment.toFixed(3)}</div>
            <div className="text-green-600">Avg Sentiment</div>
          </div>
          <div className="bg-purple-50 p-3 rounded">
            <div className="text-purple-800 font-medium">{formatDate(data.summary.mostPositiveDay.date)}</div>
            <div className="text-purple-600">Most Positive Day</div>
          </div>
          <div className="bg-red-50 p-3 rounded">
            <div className="text-red-800 font-medium">{formatDate(data.summary.mostNegativeDay.date)}</div>
            <div className="text-red-600">Most Negative Day</div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap gap-4 items-center">
          {/* View Mode Toggle */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                viewMode === 'overall' 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setViewMode('overall')}
            >
            <BarChart3 className="w-4 h-4 inline mr-1" />
            Overall Timeline
            </button>
            <button
              className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                viewMode === 'sender' 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setViewMode('sender')}
            >
            <User className="w-4 h-4 inline mr-1" />
              By Sender
            </button>
          </div>

          {/* Sender Selection */}
          {viewMode === 'sender' && (
            <select
              value={selectedSender}
              onChange={(e) => setSelectedSender(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
            {data.senderTimelines.map(timeline => (
              <option key={timeline.sender} value={timeline.sender}>
                {timeline.sender} ({timeline.overallSentiment.totalMessages.toLocaleString()} messages)
                </option>
              ))}
            </select>
          )}

          {/* Chart Type Toggle */}
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                chartType === 'area' 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setChartType('area')}
            >
              Area
            </button>
            <button
            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                chartType === 'line' 
                  ? 'bg-white text-blue-600 shadow-sm' 
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              onClick={() => setChartType('line')}
            >
              Line
            </button>
        </div>
      </div>

      {/* Chart */}
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'area' ? (
            <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="displayDate"
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value, index) => {
                  // Show every nth tick to avoid crowding
                  return index % Math.ceil(chartData.length / 8) === 0 ? value : '';
                }}
              />
              <YAxis 
                stroke="#6b7280"
                fontSize={12}
                domain={[-1, 1]}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="avgCompound"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </AreaChart>
          ) : (
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="displayDate"
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value, index) => {
                  return index % Math.ceil(chartData.length / 8) === 0 ? value : '';
                }}
              />
              <YAxis 
                stroke="#6b7280"
                fontSize={12}
                domain={[-1, 1]}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="avgCompound"
                stroke="#8b5cf6"
                strokeWidth={2}
                dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: '#8b5cf6', strokeWidth: 2, fill: '#fff' }}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SentimentTimelineChart; 