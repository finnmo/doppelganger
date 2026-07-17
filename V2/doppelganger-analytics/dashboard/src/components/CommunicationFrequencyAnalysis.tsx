'use client';

import React, { useState, useEffect, useRef } from 'react';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { ChartPlotContext } from '@/components/ui/chartPlotContext';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import { CHART_AREA } from '@/lib/layout';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, BarChart, Bar, Cell, AreaChart, Area } from 'recharts';
import { MessageSquare, TrendingUp, Calendar, Zap, Target, Activity } from 'lucide-react';

interface MonthlyMessageData {
  conversation_id: string;
  month: string;
  messageCount: number;
}

interface FrequencyData {
  month: string;
  totalMessages: number;
  avgDaily: number;
  peakDay: number;
  consistency: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  formattedMonth: string;
  sortKey: string;
}

interface SenderFrequency {
  sender: string;
  totalMessages: number;
  avgMonthly: number;
  consistency: number;
  peakMonth: string;
  activityLevel: 'very_high' | 'high' | 'medium' | 'low';
  trendDirection: 'up' | 'down' | 'stable';
}

interface FrequencyMetrics {
  totalMessages: number;
  avgMonthlyMessages: number;
  mostActiveMonth: string;
  consistencyScore: number;
  overallTrend: 'increasing' | 'decreasing' | 'stable';
  activeSenders: number;
  peakFrequency: number;
  communicationHealth: 'excellent' | 'good' | 'moderate' | 'low';
}

export function CommunicationFrequencyAnalysis() {
  const plotRef = useRef<HTMLDivElement>(null);
  const [frequencyData, setFrequencyData] = useState<FrequencyData[]>([]);
  const [senderData, setSenderData] = useState<SenderFrequency[]>([]);
  const [metrics, setMetrics] = useState<FrequencyMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'timeline' | 'senders' | 'patterns'>('timeline');
  const { filterScopedRows, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const [monthlyResponse, activeHoursResponse] = await Promise.all([
          fetch('/data/monthly-messages.json'),
          fetch('/data/activeHours.json')
        ]);
        let monthlyData: MonthlyMessageData[] = await monthlyResponse.json();
        monthlyData = filterScopedRows(monthlyData);
        let activeHoursData: Array<{ conversation_id: string; sender?: string; count: number }> = [];
        try {
          activeHoursData = await activeHoursResponse.json();
          activeHoursData = filterScopedRows(activeHoursData, { senderKey: 'sender' });
        } catch {
          activeHoursData = [];
        }

        // Aggregate by month
        const monthlyTotals = monthlyData.reduce((acc, item) => {
          if (!acc[item.month]) {
            acc[item.month] = 0;
          }
          acc[item.month] += item.messageCount;
          return acc;
        }, {} as Record<string, number>);

        // Process frequency data
        const processedFrequencyData: FrequencyData[] = Object.entries(monthlyTotals)
          .map(([month, messageCount]) => {
            const date = new Date(month + '-01');
            const daysInMonth = new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate();
            const avgDaily = messageCount / daysInMonth;

            return {
              month,
              totalMessages: messageCount,
              avgDaily: Math.round(avgDaily * 10) / 10,
              peakDay: 0, // Unknown without daily totals — omit fake estimate
              consistency: 0,
              trend: 'stable' as const,
              formattedMonth: date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' }),
              sortKey: month
            };
          })
          .sort((a, b) => a.sortKey.localeCompare(b.sortKey));

        // Peak month from real data
        const peakMonthMessages = processedFrequencyData.reduce(
          (max, curr) => (curr.totalMessages > max ? curr.totalMessages : max),
          0
        );

        // Calculate trends and consistency
        processedFrequencyData.forEach((item, index) => {
          if (index > 0) {
            const prevMonth = processedFrequencyData[index - 1];
            const change = prevMonth.totalMessages > 0
              ? (item.totalMessages - prevMonth.totalMessages) / prevMonth.totalMessages
              : 0;

            if (change > 0.1) item.trend = 'increasing' as const;
            else if (change < -0.1) item.trend = 'decreasing' as const;
            else item.trend = 'stable' as const;
          }

          if (index >= 2) {
            const recentMonths = processedFrequencyData.slice(Math.max(0, index - 2), index + 1);
            const values = recentMonths.map(m => m.totalMessages);
            const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
            const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
            const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
            item.consistency = Math.max(0, Math.min(100, (1 - cv) * 100));
          }
        });

        // Sender frequencies from activeHours when available
        const senderTotals = new Map<string, number>();
        activeHoursData.forEach((row) => {
          if (!row.sender) return;
          senderTotals.set(row.sender, (senderTotals.get(row.sender) || 0) + (row.count || 0));
        });

        const monthCount = Math.max(processedFrequencyData.length, 1);
        const peakMonthLabel = processedFrequencyData.reduce(
          (max, curr) => (curr.totalMessages > max.totalMessages ? curr : max),
          processedFrequencyData[0] || { formattedMonth: 'Unknown', totalMessages: 0 }
        ).formattedMonth;

        const realSenderData: SenderFrequency[] = Array.from(senderTotals.entries())
          .map(([sender, totalMessages]) => {
            let activityLevel: SenderFrequency['activityLevel'] = 'low';
            const avgMonthly = totalMessages / monthCount;
            if (avgMonthly > 500) activityLevel = 'very_high';
            else if (avgMonthly > 200) activityLevel = 'high';
            else if (avgMonthly > 50) activityLevel = 'medium';

            return {
              sender,
              totalMessages,
              avgMonthly: Math.round(avgMonthly),
              consistency: 0,
              peakMonth: peakMonthLabel,
              activityLevel,
              trendDirection: 'stable' as const
            };
          })
          .sort((a, b) => b.totalMessages - a.totalMessages);

        // Calculate overall metrics
        const totalMessages = processedFrequencyData.reduce((sum, m) => sum + m.totalMessages, 0);
        const avgMonthlyMessages = processedFrequencyData.length > 0
          ? Math.round(totalMessages / processedFrequencyData.length)
          : 0;
        const mostActiveMonth = processedFrequencyData.reduce((max, curr) =>
          curr.totalMessages > max.totalMessages ? curr : max,
          processedFrequencyData[0] || { formattedMonth: 'N/A', totalMessages: 0 }
        );

        const monthlyValues = processedFrequencyData.map(m => m.totalMessages);
        const mean = monthlyValues.length > 0
          ? monthlyValues.reduce((sum, val) => sum + val, 0) / monthlyValues.length
          : 0;
        const variance = monthlyValues.length > 0
          ? monthlyValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / monthlyValues.length
          : 0;
        const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
        const consistencyScore = Math.max(0, Math.min(100, (1 - cv) * 100));

        const firstHalf = processedFrequencyData.slice(0, Math.floor(processedFrequencyData.length / 2));
        const secondHalf = processedFrequencyData.slice(Math.floor(processedFrequencyData.length / 2));
        const firstHalfAvg = firstHalf.length > 0
          ? firstHalf.reduce((sum, m) => sum + m.totalMessages, 0) / firstHalf.length
          : 0;
        const secondHalfAvg = secondHalf.length > 0
          ? secondHalf.reduce((sum, m) => sum + m.totalMessages, 0) / secondHalf.length
          : 0;
        const overallChange = firstHalfAvg > 0 ? (secondHalfAvg - firstHalfAvg) / firstHalfAvg : 0;

        let overallTrend: 'increasing' | 'decreasing' | 'stable' = 'stable';
        if (overallChange > 0.15) overallTrend = 'increasing' as const;
        else if (overallChange < -0.15) overallTrend = 'decreasing' as const;

        let communicationHealth: 'excellent' | 'good' | 'moderate' | 'low' = 'low';
        if (consistencyScore > 80 && avgMonthlyMessages > 1000) communicationHealth = 'excellent';
        else if (consistencyScore > 60 && avgMonthlyMessages > 500) communicationHealth = 'good';
        else if (consistencyScore > 40 && avgMonthlyMessages > 200) communicationHealth = 'moderate';

        const frequencyMetrics: FrequencyMetrics = {
          totalMessages,
          avgMonthlyMessages,
          mostActiveMonth: mostActiveMonth.formattedMonth,
          consistencyScore: Math.round(consistencyScore),
          overallTrend,
          activeSenders: realSenderData.length,
          peakFrequency: peakMonthMessages || mostActiveMonth.totalMessages,
          communicationHealth
        };

        setFrequencyData(processedFrequencyData);
        setSenderData(realSenderData);
        setMetrics(frequencyMetrics);

      } catch (error) {
        console.error('Error loading frequency data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds]);

  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <MessageSquare className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Communication Frequency Analysis</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading frequency analysis...</div>
        </div>
      </div>
    );
  }

  if (!metrics || frequencyData.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <MessageSquare className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold">Communication Frequency Analysis</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Activity className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">No Frequency Data Available</p>
            <p className="text-sm text-gray-400">Communication frequency could not be analyzed</p>
          </div>
        </div>
      </div>
    );
  }

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing': return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'decreasing': return <TrendingUp className="w-4 h-4 text-red-600 transform rotate-180" />;
      default: return <Target className="w-4 h-4 text-blue-600" />;
    }
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'excellent': return 'text-green-600 bg-green-50';
      case 'good': return 'text-blue-600 bg-blue-50';
      case 'moderate': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-red-600 bg-red-50';
    }
  };

  const getActivityColor = (level: string) => {
    switch (level) {
      case 'very_high': return '#10b981';
      case 'high': return '#3b82f6';
      case 'medium': return '#f59e0b';
      default: return '#ef4444';
    }
  };

  const format = (value: number, data: Partial<FrequencyData> & Partial<SenderFrequency>) => {
    return [
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{data.formattedMonth || data.sender}</div>
        <div>Messages: {value.toLocaleString()}</div>
        {data.avgDaily && <div>Daily Avg: {data.avgDaily}</div>}
        {data.consistency !== undefined && <div>Consistency: {data.consistency.toFixed(1)}%</div>}
        {data.trend && <div>Trend: {data.trend}</div>}
      </div>
    ];
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-white p-4 shadow-sm sm:p-6">
      <div className="mb-4 flex shrink-0 items-center justify-between sm:mb-6">
        <div className="flex items-center space-x-2">
          <MessageSquare className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Communication Frequency Analysis</h3>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('timeline')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'timeline' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Timeline
          </button>
          <button
            onClick={() => setViewMode('senders')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'senders' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            By Sender
          </button>
          <button
            onClick={() => setViewMode('patterns')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'patterns' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Patterns
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="mb-4 grid shrink-0 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 sm:mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <MessageSquare className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-600 font-medium">Total Messages</span>
          </div>
          <div className="text-2xl font-bold text-blue-900 mt-1">
            {metrics.totalMessages.toLocaleString()}
          </div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Calendar className="w-4 h-4 text-green-600" />
            <span className="text-sm text-green-600 font-medium">Monthly Average</span>
          </div>
          <div className="text-2xl font-bold text-green-900 mt-1">
            {metrics.avgMonthlyMessages.toLocaleString()}
          </div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-purple-600" />
            <span className="text-sm text-purple-600 font-medium">Consistency</span>
          </div>
          <div className="text-2xl font-bold text-purple-900 mt-1">
            {metrics.consistencyScore}%
          </div>
        </div>
        
        <div className={`p-4 rounded-lg ${getHealthColor(metrics.communicationHealth)}`}>
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4" />
            <span className="text-sm font-medium">Health</span>
          </div>
          <div className="text-sm font-bold mt-1 capitalize">
            {metrics.communicationHealth}
          </div>
        </div>
      </div>

      {/* Charts */}
      <ChartPlotContext.Provider value={plotRef}>
      <div ref={plotRef} className={CHART_AREA}>
        {viewMode === 'timeline' && (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={frequencyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="formattedMonth" />
              <YAxis />
              <ChartTooltip content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      {format(Number(payload[0].value), payload[0].payload)}
                    </div>
                  );
                }
                return null;
              }} />
              <Area
                type="monotone"
                dataKey="totalMessages"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}

        {viewMode === 'senders' && (
          senderData.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
              Sender breakdown unavailable
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={senderData.slice(0, 12)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="sender" angle={-35} textAnchor="end" height={70} interval={0} fontSize={10} />
                <YAxis />
                <ChartTooltip content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        {format(Number(payload[0].value), payload[0].payload)}
                      </div>
                    );
                  }
                  return null;
                }} />
                <Bar dataKey="totalMessages" radius={[4, 4, 0, 0]}>
                  {senderData.slice(0, 12).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={getActivityColor(entry.activityLevel)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )
        )}

        {viewMode === 'patterns' && (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={frequencyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="formattedMonth" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <ChartTooltip content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      <div className="text-sm">
                        <div className="font-semibold">{payload[0].payload.formattedMonth}</div>
                        <div>Messages: {payload[0].value?.toLocaleString()}</div>
                        <div>Daily Avg: {payload[0].payload.avgDaily}</div>
                        <div>Consistency: {typeof payload[1]?.value === 'number' ? payload[1].value.toFixed(1) : payload[1]?.value}%</div>
                      </div>
                    </div>
                  );
                }
                return null;
              }} />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="totalMessages"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="consistency"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      </ChartPlotContext.Provider>

      {/* Insights Panel */}
      <div className="mt-6 bg-gray-50 p-4 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">📈 Frequency Insights</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div className="flex items-center space-x-2">
            {getTrendIcon(metrics.overallTrend)}
            <span>
              <strong>Overall Trend:</strong> Communication is {metrics.overallTrend} over time
            </span>
          </div>
          <div>
            <strong>Peak Activity:</strong> {metrics.mostActiveMonth} with {metrics.peakFrequency.toLocaleString()} messages
          </div>
          <div>
            <strong>Consistency:</strong> {metrics.consistencyScore}% consistency indicates {metrics.consistencyScore > 70 ? 'regular' : metrics.consistencyScore > 50 ? 'moderate' : 'irregular'} communication patterns
          </div>
          <div>
            <strong>Health Status:</strong> Communication health is {metrics.communicationHealth} based on frequency and consistency
          </div>
        </div>
      </div>
    </div>
  );
} 