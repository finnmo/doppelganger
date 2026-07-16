'use client';

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface ActivityData {
  conversation_id: string;
  hour: number | string;
  sender?: string;
  count: number;
  day?: string;
  day_of_week?: number;
}

interface HourlyData {
  hour: number;
  count: number;
  label: string;
}

interface DailyData {
  day: string;
  count: number;
  dayNumber: number;
}

const DAY_ORDER_SUN = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
const DAY_INDEX_TO_NAME = DAY_ORDER_SUN;

function formatHour(hour: number): string {
  if (hour === 0) return '12 AM';
  if (hour === 12) return '12 PM';
  if (hour < 12) return `${hour} AM`;
  return `${hour - 12} PM`;
}

function resolveDayName(row: ActivityData): string | null {
  if (row.day && typeof row.day === 'string') {
    const normalized = row.day.charAt(0).toUpperCase() + row.day.slice(1).toLowerCase();
    if (DAY_ORDER_SUN.includes(normalized)) return normalized;
    if (DAY_INDEX_TO_NAME.includes(row.day)) return row.day;
  }
  if (typeof row.day_of_week === 'number' && row.day_of_week >= 0 && row.day_of_week <= 6) {
    return DAY_INDEX_TO_NAME[row.day_of_week];
  }
  return null;
}

function topNKeys(totals: Record<number | string, number>, n: number): Array<number | string> {
  return Object.entries(totals)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([key]) => (typeof key === 'string' && /^\d+$/.test(key) ? Number(key) : key));
}

export function PeakActivityChart() {
  const [hourlyData, setHourlyData] = useState<HourlyData[]>([]);
  const [dailyData, setDailyData] = useState<DailyData[]>([]);
  const [peakInfo, setPeakInfo] = useState<{ peak_hours: number[]; peak_days: string[]; dayDataAvailable: boolean }>({
    peak_hours: [],
    peak_days: [],
    dayDataAvailable: false
  });
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'hourly' | 'daily'>('hourly');
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const activeHoursResponse = await fetch('/data/activeHours.json');
        let activeHoursData: ActivityData[] = await activeHoursResponse.json();

        if (!Array.isArray(activeHoursData)) {
          activeHoursData = [];
        }

        if (isFiltered && selectedConversations.length > 0) {
          activeHoursData = activeHoursData.filter((item) =>
            selectedConversations.includes(item.conversation_id)
          );
        }

        const hourlyTotals = activeHoursData.reduce((acc, item) => {
          const hour = typeof item.hour === 'number' ? item.hour : parseInt(String(item.hour), 10);
          if (Number.isNaN(hour)) return acc;
          acc[hour] = (acc[hour] || 0) + item.count;
          return acc;
        }, {} as Record<number, number>);

        const processedHourlyData = Array.from({ length: 24 }, (_, hour) => ({
          hour,
          count: hourlyTotals[hour] || 0,
          label: formatHour(hour)
        }));

        // Day-of-week only from real day / day_of_week fields — never invent
        let hasDayField = false;
        const dailyTotals: Record<string, number> = {};
        activeHoursData.forEach((item) => {
          const dayName = resolveDayName(item);
          if (!dayName) return;
          hasDayField = true;
          dailyTotals[dayName] = (dailyTotals[dayName] || 0) + item.count;
        });

        const processedDailyData: DailyData[] = DAY_ORDER_SUN.map((dayName, index) => ({
          day: dayName,
          count: hasDayField ? (dailyTotals[dayName] || 0) : 0,
          dayNumber: index
        }));

        const peakHours = topNKeys(hourlyTotals, 3).map(Number).filter((h) => !Number.isNaN(h));
        const peakDays = hasDayField
          ? (topNKeys(dailyTotals, 3) as string[])
          : [];

        // When not filtered, optionally enrich peak labels from timeMetrics if day fields missing
        if (!isFiltered && !hasDayField) {
          try {
            const timeMetricsResponse = await fetch('/data/timeMetrics.json');
            const timeMetricsData: {
              peak_hours?: number[];
              peak_days?: string[];
              daily_activity?: Record<string, number>;
            } = await timeMetricsResponse.json();

            if (timeMetricsData.daily_activity) {
              hasDayField = true;
              processedDailyData.forEach((row) => {
                row.count = timeMetricsData.daily_activity?.[row.day] || 0;
              });
              setDailyData([...processedDailyData]);
              setPeakInfo({
                peak_hours: peakHours.length > 0 ? peakHours : (timeMetricsData.peak_hours || []),
                peak_days: timeMetricsData.peak_days || topNKeys(
                  timeMetricsData.daily_activity,
                  3
                ) as string[],
                dayDataAvailable: true
              });
              setHourlyData(processedHourlyData);
              return;
            }
          } catch {
            // keep activeHours-derived peaks
          }
        }

        setHourlyData(processedHourlyData);
        setDailyData(processedDailyData);
        setPeakInfo({
          peak_hours: peakHours,
          peak_days: peakDays,
          dayDataAvailable: hasDayField
        });
      } catch (error) {
        console.error('Error loading activity data:', error);
        setHourlyData(Array.from({ length: 24 }, (_, hour) => ({
          hour,
          count: 0,
          label: formatHour(hour)
        })));
        setDailyData(DAY_ORDER_SUN.map((day, index) => ({
          day,
          count: 0,
          dayNumber: index
        })));
        setPeakInfo({ peak_hours: [], peak_days: [], dayDataAvailable: false });
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  const getDayColor = (day: string): string => {
    if (peakInfo.peak_days.includes(day)) return '#ef4444';
    if (day === 'Saturday' || day === 'Sunday') return '#10b981';
    return '#3b82f6';
  };

  if (loading) {
    return <div className="flex items-center justify-center h-full">Loading...</div>;
  }

  const effectiveView = viewMode === 'daily' && !peakInfo.dayDataAvailable ? 'hourly' : viewMode;

  return (
    <div className="h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('hourly')}
            className={`px-3 py-1 text-sm rounded-lg transition-colors ${
              viewMode === 'hourly'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            By Hour
          </button>
          <button
            onClick={() => setViewMode('daily')}
            className={`px-3 py-1 text-sm rounded-lg transition-colors ${
              viewMode === 'daily'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            By Day
          </button>
        </div>

        <div className="text-right text-sm">
          {effectiveView === 'hourly' ? (
            <>
              <div className="text-gray-600">Peak Hours:</div>
              <div className="font-medium text-red-600">
                {peakInfo.peak_hours.length > 0
                  ? peakInfo.peak_hours.map((h) => formatHour(h)).join(', ')
                  : 'None'}
              </div>
            </>
          ) : (
            <>
              <div className="text-gray-600">Peak Days:</div>
              <div className="font-medium text-red-600">
                {peakInfo.peak_days.length > 0
                  ? peakInfo.peak_days.join(', ')
                  : 'None'}
              </div>
            </>
          )}
        </div>
      </div>

      {viewMode === 'daily' && !peakInfo.dayDataAvailable && (
        <div className="mb-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
          Day-of-week data unavailable — showing hourly peaks instead.
        </div>
      )}

      <div className="min-h-0 flex-1">
        <ResponsiveContainer width="100%" height="100%">
          {effectiveView === 'hourly' ? (
            <LineChart data={hourlyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="label"
                stroke="#6b7280"
                fontSize={11}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value) => value.toLocaleString()}
              />
              <Tooltip
                formatter={(value, name, props) => [
                  Number(value).toLocaleString() + ' messages',
                  `${props.payload.label} Activity`
                ]}
                labelFormatter={(label) => `Time: ${label}`}
              />
              <Line
                type="monotone"
                dataKey="count"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, fill: '#ef4444' }}
              />
            </LineChart>
          ) : (
            <BarChart data={dailyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis
                dataKey="day"
                stroke="#6b7280"
                fontSize={12}
              />
              <YAxis
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value) => value.toLocaleString()}
              />
              <Tooltip
                formatter={(value) => [Number(value).toLocaleString() + ' messages', 'Activity']}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {dailyData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getDayColor(entry.day)} />
                ))}
              </Bar>
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}
