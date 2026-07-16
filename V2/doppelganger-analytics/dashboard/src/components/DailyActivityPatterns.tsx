'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Cell } from 'recharts';
import { Calendar, Clock, TrendingUp, Activity, Sun, Moon, Coffee, Sunset } from 'lucide-react';

interface ActiveHoursRow {
  conversation_id: string;
  hour: number;
  day_of_week?: number;
  day?: string;
  sender?: string;
  count: number;
}

interface ProcessedDayData {
  day: string;
  dayNumber: number;
  totalMessages: number;
  avgSentiment: number;
  mediaCount: number;
  peakHour: number;
  activityLevel: 'low' | 'medium' | 'high' | 'peak';
}

interface ProcessedHourData {
  hour: number;
  totalMessages: number;
  avgSentiment: number;
  mediaCount: number;
  timeLabel: string;
  period: 'morning' | 'afternoon' | 'evening' | 'night';
}

interface ActivitySummary {
  totalMessages: number;
  mostActiveDay: string;
  mostActiveHour: number;
  avgDailyMessages: number;
  peakActivityPeriod: string;
  activityVariance: number;
  dayDataAvailable: boolean;
}

const DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const DAY_INDEX_TO_NAME = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const DAY_COLORS: Record<string, string> = {
  Monday: '#3b82f6',
  Tuesday: '#10b981',
  Wednesday: '#f59e0b',
  Thursday: '#ef4444',
  Friday: '#8b5cf6',
  Saturday: '#06b6d4',
  Sunday: '#f97316'
};

function resolveDayName(row: ActiveHoursRow): string | null {
  if (row.day && typeof row.day === 'string') {
    const normalized = row.day.charAt(0).toUpperCase() + row.day.slice(1).toLowerCase();
    if (DAY_ORDER.includes(normalized)) return normalized;
    if (DAY_INDEX_TO_NAME.includes(row.day)) return row.day;
  }
  if (typeof row.day_of_week === 'number' && row.day_of_week >= 0 && row.day_of_week <= 6) {
    return DAY_INDEX_TO_NAME[row.day_of_week];
  }
  return null;
}

function getPeriod(hour: number): 'morning' | 'afternoon' | 'evening' | 'night' {
  if (hour >= 6 && hour < 12) return 'morning';
  if (hour >= 12 && hour < 18) return 'afternoon';
  if (hour >= 18 && hour < 22) return 'evening';
  return 'night';
}

export function DailyActivityPatterns() {
  const [dayData, setDayData] = useState<ProcessedDayData[]>([]);
  const [hourData, setHourData] = useState<ProcessedHourData[]>([]);
  const [summary, setSummary] = useState<ActivitySummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'days' | 'hours' | 'radar'>('days');
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const response = await fetch('/data/activeHours.json');
        let rows: ActiveHoursRow[] = await response.json();

        if (!Array.isArray(rows)) {
          rows = [];
        }

        if (isFiltered && selectedConversations.length > 0) {
          rows = rows.filter((row) => selectedConversations.includes(row.conversation_id));
        }

        const dayMap = new Map<string, { totalMessages: number; hourCounts: Map<number, number> }>();
        const hourMap = new Map<number, number>();
        let hasDayField = false;

        rows.forEach((row) => {
          const hour = typeof row.hour === 'number' ? row.hour : Number(row.hour);
          const count = typeof row.count === 'number' ? row.count : 0;
          if (Number.isNaN(hour) || hour < 0 || hour > 23) return;

          hourMap.set(hour, (hourMap.get(hour) || 0) + count);

          const dayName = resolveDayName(row);
          if (dayName) {
            hasDayField = true;
            if (!dayMap.has(dayName)) {
              dayMap.set(dayName, { totalMessages: 0, hourCounts: new Map() });
            }
            const dayStats = dayMap.get(dayName)!;
            dayStats.totalMessages += count;
            dayStats.hourCounts.set(hour, (dayStats.hourCounts.get(hour) || 0) + count);
          }
        });

        const maxDayMessages = hasDayField
          ? Math.max(0, ...DAY_ORDER.map((d) => dayMap.get(d)?.totalMessages || 0))
          : 0;

        const processedDayData: ProcessedDayData[] = DAY_ORDER.map((day, index) => {
          const stats = dayMap.get(day) || { totalMessages: 0, hourCounts: new Map<number, number>() };
          const peakHour = Array.from(stats.hourCounts.entries())
            .sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;

          let activityLevel: ProcessedDayData['activityLevel'] = 'low';
          if (hasDayField && maxDayMessages > 0) {
            if (stats.totalMessages > maxDayMessages * 0.8) activityLevel = 'peak';
            else if (stats.totalMessages > maxDayMessages * 0.6) activityLevel = 'high';
            else if (stats.totalMessages > maxDayMessages * 0.3) activityLevel = 'medium';
          }

          return {
            day,
            dayNumber: index,
            totalMessages: hasDayField ? stats.totalMessages : 0,
            avgSentiment: 0,
            mediaCount: 0,
            peakHour,
            activityLevel
          };
        });

        const processedHourData: ProcessedHourData[] = Array.from({ length: 24 }, (_, hour) => ({
          hour,
          totalMessages: hourMap.get(hour) || 0,
          avgSentiment: 0,
          mediaCount: 0,
          timeLabel: `${hour.toString().padStart(2, '0')}:00`,
          period: getPeriod(hour)
        }));

        const totalMessages = processedHourData.reduce((sum, h) => sum + h.totalMessages, 0);
        const mostActiveDay = hasDayField
          ? processedDayData.reduce((max, day) =>
              day.totalMessages > max.totalMessages ? day : max
            ).day
          : 'N/A';
        const mostActiveHour = processedHourData.reduce((max, hour) =>
          hour.totalMessages > max.totalMessages ? hour : max
        ).hour;
        const avgDailyMessages = Math.round(totalMessages / 7);

        const periodTotals = processedHourData.reduce((acc, hour) => {
          acc[hour.period] = (acc[hour.period] || 0) + hour.totalMessages;
          return acc;
        }, {} as Record<string, number>);
        const peakActivityPeriod = Object.entries(periodTotals)
          .sort((a, b) => b[1] - a[1])[0]?.[0] || 'morning';

        const dayTotals = processedDayData.map((d) => d.totalMessages);
        const mean = dayTotals.reduce((sum, val) => sum + val, 0) / dayTotals.length;
        const variance = dayTotals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / dayTotals.length;

        setDayData(processedDayData);
        setHourData(processedHourData);
        setSummary({
          totalMessages,
          mostActiveDay,
          mostActiveHour,
          avgDailyMessages,
          peakActivityPeriod,
          activityVariance: Math.round(Math.sqrt(variance)),
          dayDataAvailable: hasDayField
        });
      } catch (error) {
        console.error('Error loading daily activity data:', error);
        setDayData([]);
        setHourData([]);
        setSummary(null);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <Calendar className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Daily Activity Patterns</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading activity patterns...</div>
        </div>
      </div>
    );
  }

  if (!summary || (summary.totalMessages === 0 && hourData.every((h) => h.totalMessages === 0))) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <Calendar className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold">Daily Activity Patterns</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Activity className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">No Activity Data Available</p>
            <p className="text-sm text-gray-400">Activity patterns could not be loaded</p>
          </div>
        </div>
      </div>
    );
  }

  const getActivityIcon = (period: string) => {
    switch (period) {
      case 'morning': return <Sun className="w-4 h-4" />;
      case 'afternoon': return <Coffee className="w-4 h-4" />;
      case 'evening': return <Sunset className="w-4 h-4" />;
      case 'night': return <Moon className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const formatTooltip = (value: number, data: Partial<ProcessedDayData> & Partial<ProcessedHourData>) => {
    return [
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{data.day || data.timeLabel}</div>
        <div>Messages: {value.toLocaleString()}</div>
        {data.peakHour !== undefined && data.day && <div>Peak Hour: {data.peakHour}:00</div>}
      </div>
    ];
  };

  const radarData = DAY_ORDER.map((day) => {
    const dayStats = dayData.find((d) => d.day === day) || { totalMessages: 0 };
    return {
      day: day.substring(0, 3),
      messages: dayStats.totalMessages,
      sentiment: 0,
      media: 0
    };
  });

  const effectiveViewMode =
    !summary.dayDataAvailable && (viewMode === 'days' || viewMode === 'radar') ? 'hours' : viewMode;

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Calendar className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Daily Activity Patterns</h3>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('days')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'days' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            By Day
          </button>
          <button
            onClick={() => setViewMode('hours')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'hours' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            By Hour
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

      {!summary.dayDataAvailable && (
        <div className="mb-4 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          Day-of-week breakdown unavailable in current data — showing hourly activity only. Regenerate analytics to enable day charts.
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-600 font-medium">Most Active Day</span>
          </div>
          <div className="text-2xl font-bold text-blue-900 mt-1">
            {summary.mostActiveDay}
          </div>
        </div>

        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-green-600" />
            <span className="text-sm text-green-600 font-medium">Peak Hour</span>
          </div>
          <div className="text-2xl font-bold text-green-900 mt-1">
            {summary.mostActiveHour}:00
          </div>
        </div>

        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            {getActivityIcon(summary.peakActivityPeriod)}
            <span className="text-sm text-purple-600 font-medium">Peak Period</span>
          </div>
          <div className="text-sm font-bold text-purple-900 mt-1 capitalize">
            {summary.peakActivityPeriod}
          </div>
        </div>

        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-orange-600" />
            <span className="text-sm text-orange-600 font-medium">Daily Average</span>
          </div>
          <div className="text-2xl font-bold text-orange-900 mt-1">
            {summary.avgDailyMessages}
          </div>
        </div>
      </div>

      <div className="h-80">
        {effectiveViewMode === 'days' && (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={dayData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      {formatTooltip(Number(payload[0].value), payload[0].payload)}
                    </div>
                  );
                }
                return null;
              }} />
              <Bar dataKey="totalMessages" radius={[4, 4, 0, 0]}>
                {dayData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={DAY_COLORS[entry.day]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}

        {effectiveViewMode === 'hours' && (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={hourData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timeLabel" interval={2} />
              <YAxis />
              <Tooltip content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      {formatTooltip(Number(payload[0].value), payload[0].payload)}
                    </div>
                  );
                }
                return null;
              }} />
              <Line
                type="monotone"
                dataKey="totalMessages"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}

        {effectiveViewMode === 'radar' && (
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="day" />
              <PolarRadiusAxis angle={90} domain={[0, 'dataMax']} />
              <Radar
                name="Messages"
                dataKey="messages"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.3}
                strokeWidth={2}
              />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="mt-6 bg-gray-50 p-4 rounded-lg">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Activity Insights</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
          <div>
            <strong>Peak Activity:</strong>{' '}
            {summary.dayDataAvailable
              ? `${summary.mostActiveDay} is your most active day with peak activity at ${summary.mostActiveHour}:00`
              : `Peak activity at ${summary.mostActiveHour}:00`}
          </div>
          <div>
            <strong>Activity Pattern:</strong> Most active during {summary.peakActivityPeriod} hours
          </div>
          {summary.dayDataAvailable && (
            <div>
              <strong>Consistency:</strong>{' '}
              {summary.activityVariance < 50 ? 'Very consistent' : summary.activityVariance < 100 ? 'Moderately consistent' : 'Highly variable'}{' '}
              activity levels across days
            </div>
          )}
          <div>
            <strong>Daily Volume:</strong> Average of {summary.avgDailyMessages} messages per day
          </div>
        </div>
      </div>
    </div>
  );
}
