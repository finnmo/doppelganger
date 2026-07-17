'use client';

import React, { useMemo } from 'react';
import { Clock, Zap, Sun } from 'lucide-react';
import { useDashData } from '@/hooks/useDashData';
import { useParticipantScope } from '@/hooks/useParticipantScope';

interface ActiveHour {
  conversation_id: string;
  hour: number;
  sender: string;
  count: number;
}

interface LatencyBucket {
  conversation_id: string;
  bucket: string;
  count: number;
}

function formatHour12(hour: number): string {
  if (hour === 0) return '12 AM';
  if (hour === 12) return '12 PM';
  return hour < 12 ? `${hour} AM` : `${hour - 12} PM`;
}

// Buckets from the latency processor, grouped into fast/medium/slow.
const FAST_BUCKETS = new Set(['0-10s', '10-30s', '30-60s']);
const MEDIUM_BUCKETS = new Set(['1-5m', '5-15m']);

const DAY_PARTS = [
  { label: 'Morning', emoji: '🌅', from: 6, to: 12 },
  { label: 'Afternoon', emoji: '🌞', from: 12, to: 18 },
  { label: 'Evening', emoji: '🌙', from: 18, to: 24 },
  { label: 'Night', emoji: '😴', from: 0, to: 6 }
];

export function ActivityInsightsPanel() {
  const { data: activeHours } = useDashData<ActiveHour[]>('activeHours.json');
  const { data: latency } = useDashData<LatencyBucket[]>('replyLatencyDistribution.json');
  const { filterScopedRows } = useParticipantScope();

  const scopedActiveHours = useMemo(
    () => filterScopedRows(activeHours || [], { senderKey: 'sender' }),
    [activeHours, filterScopedRows]
  );
  const scopedLatency = useMemo(
    () => filterScopedRows(latency || []),
    [latency, filterScopedRows]
  );

  // Hourly totals for the selected conversation(s), participants only
  const hourTotals = new Array(24).fill(0);
  for (const row of scopedActiveHours) {
    hourTotals[row.hour] += row.count;
  }
  const totalActivity = hourTotals.reduce((a, b) => a + b, 0);
  const peakHour = totalActivity > 0 ? hourTotals.indexOf(Math.max(...hourTotals)) : null;

  // Peak within each part of day
  const partPeak = DAY_PARTS.map(part => {
    let bestHour = part.from, bestCount = -1;
    for (let h = part.from; h < part.to; h++) {
      if (hourTotals[h] > bestCount) { bestCount = hourTotals[h]; bestHour = h; }
    }
    return { ...part, hour: bestHour, count: bestCount };
  });

  // Busiest part of day
  const busiestPart = totalActivity > 0
    ? DAY_PARTS.map(part => ({
        label: part.label,
        total: hourTotals.slice(part.from, part.to).reduce((a, b) => a + b, 0)
      })).sort((a, b) => b.total - a.total)[0]
    : null;

  // Response-speed split from the latency distribution
  let fast = 0, medium = 0, slow = 0, totalReplies = 0;
  for (const row of scopedLatency) {
    totalReplies += row.count;
    if (FAST_BUCKETS.has(row.bucket)) fast += row.count;
    else if (MEDIUM_BUCKETS.has(row.bucket)) medium += row.count;
    else slow += row.count;
  }
  const pctReplies = (n: number) => (totalReplies > 0 ? Math.round((n / totalReplies) * 100) : 0);
  const speedLabel = totalReplies === 0 ? '—' : fast / totalReplies >= 0.5 ? 'Fast' : fast / totalReplies >= 0.25 ? 'Moderate' : 'Slow';

  return (
    <div className="grid grid-cols-1 gap-3">
      {/* Peak Hour */}
      <div className="bg-gradient-to-br from-orange-400 to-orange-500 rounded-xl p-4 text-white shadow-sm">
        <div className="flex items-center mb-4">
          <Clock className="w-6 h-6 mr-2" />
          <h3 className="text-xl font-bold">Peak Hour</h3>
        </div>
        <div className="text-4xl font-bold mb-4">
          {peakHour !== null ? formatHour12(peakHour) : '—'}
        </div>
        <div className="space-y-3 mb-6">
          {partPeak.slice(0, 3).map(part => (
            <div key={part.label} className="flex justify-between items-center">
              <span className="text-orange-100">{part.emoji} {part.label} peak</span>
              <span className="font-semibold">{part.count > 0 ? formatHour12(part.hour) : '—'}</span>
            </div>
          ))}
        </div>
        <div className="pt-4 border-t border-orange-300 border-opacity-40">
          <div className="text-orange-100 text-sm">Most active communication time</div>
        </div>
      </div>

      {/* Response Speed */}
      <div className="bg-gradient-to-br from-green-400 to-green-500 rounded-xl p-4 text-white shadow-sm">
        <div className="flex items-center mb-4">
          <Zap className="w-6 h-6 mr-2" />
          <h3 className="text-xl font-bold">Response Speed</h3>
        </div>
        <div className="text-4xl font-bold mb-4">{speedLabel}</div>
        <div className="space-y-3 mb-6">
          <div className="flex justify-between items-center">
            <span className="text-green-100">⚡ Under 1 min</span>
            <span className="font-semibold">{pctReplies(fast)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-green-100">🏃 1–15 min</span>
            <span className="font-semibold">{pctReplies(medium)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-green-100">🚶 Over 15 min</span>
            <span className="font-semibold">{pctReplies(slow)}%</span>
          </div>
        </div>
        <div className="pt-4 border-t border-green-300 border-opacity-40">
          <div className="text-green-100 text-sm">Reply latency across the conversation</div>
        </div>
      </div>

      {/* Busiest Time of Day */}
      <div className="bg-gradient-to-br from-blue-400 to-blue-500 rounded-xl p-4 text-white shadow-sm">
        <div className="flex items-center mb-4">
          <Sun className="w-6 h-6 mr-2" />
          <h3 className="text-xl font-bold">Busiest Window</h3>
        </div>
        <div className="text-4xl font-bold mb-4">{busiestPart ? busiestPart.label : '—'}</div>
        <div className="space-y-3 mb-6">
          {DAY_PARTS.map(part => {
            const total = hourTotals.slice(part.from, part.to).reduce((a, b) => a + b, 0);
            return (
              <div key={part.label} className="flex justify-between items-center">
                <span className="text-blue-100">{part.emoji} {part.label}</span>
                <span className="font-semibold">{totalActivity > 0 ? `${Math.round((total / totalActivity) * 100)}%` : '—'}</span>
              </div>
            );
          })}
        </div>
        <div className="pt-4 border-t border-blue-300 border-opacity-40">
          <div className="text-blue-100 text-sm">Share of messages by time of day</div>
        </div>
      </div>
    </div>
  );
}
