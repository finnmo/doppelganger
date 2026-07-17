'use client';

import React, { useState, useEffect } from 'react';
import { useParticipantScope } from '@/hooks/useParticipantScope';

interface ActivityData {
  conversation_id: string;
  hour: string;
  sender: string;
  count: number;
}

interface HeatmapCell {
  hour: number;
  sender: string;
  fullSender: string;
  count: number;
  intensity: number;
}

interface ActivityHeatmapProps {
  /** Compact card view vs expanded fullscreen grid. */
  variant?: 'card' | 'expanded';
}

function getIntensityColor(intensity: number): string {
  if (intensity === 0) return 'bg-gray-100';
  if (intensity < 0.2) return 'bg-blue-200';
  if (intensity < 0.4) return 'bg-blue-300';
  if (intensity < 0.6) return 'bg-blue-400';
  if (intensity < 0.8) return 'bg-blue-500';
  return 'bg-blue-600';
}

export function ActivityHeatmap({ variant = 'card' }: ActivityHeatmapProps) {
  const [cells, setCells] = useState<HeatmapCell[]>([]);
  const [senders, setSenders] = useState<string[]>([]);
  const [fullSenders, setFullSenders] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const { filterScopedRows, scopeConversationIds } = useParticipantScope();

  const maxSenders = variant === 'expanded' ? 20 : 8;
  const cellHeight = variant === 'expanded' ? 'h-8' : 'h-7 sm:h-8';
  const labelWidth = variant === 'expanded' ? 'w-32' : 'w-24';

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/activeHours.json');
        let activityData: ActivityData[] = await response.json();
        activityData = filterScopedRows(activityData, { senderKey: 'sender' });

        const aggregatedData = activityData.reduce((acc, item) => {
          const key = `${item.sender}-${item.hour}`;
          if (!acc[key]) {
            acc[key] = { sender: item.sender, hour: item.hour, count: 0 };
          }
          acc[key].count += item.count;
          return acc;
        }, {} as Record<string, { sender: string; hour: string; count: number }>);

        const processed = Object.values(aggregatedData);
        const senderCounts = processed.reduce((acc, item) => {
          acc[item.sender] = (acc[item.sender] || 0) + item.count;
          return acc;
        }, {} as Record<string, number>);

        const topSenders = Object.entries(senderCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, maxSenders)
          .map(([sender]) => sender);

        const filtered = processed.filter((item) => topSenders.includes(item.sender));
        const maxCount = Math.max(...filtered.map((item) => item.count), 1);

        const heatmapData: HeatmapCell[] = [];
        for (let hour = 0; hour < 24; hour++) {
          for (const sender of topSenders) {
            const item = filtered.find(
              (d) => parseInt(d.hour, 10) === hour && d.sender === sender
            );
            const display =
              variant === 'expanded'
                ? sender
                : sender.length > 12
                  ? `${sender.substring(0, 12)}…`
                  : sender;
            heatmapData.push({
              hour,
              sender: display,
              fullSender: sender,
              count: item?.count ?? 0,
              intensity: item ? item.count / maxCount : 0,
            });
          }
        }

        setCells(heatmapData);
        setFullSenders(topSenders);
        setSenders(
          topSenders.map((s) =>
            variant === 'expanded' ? s : s.length > 12 ? `${s.substring(0, 12)}…` : s
          )
        );
      } catch (error) {
        console.error('Error loading activity data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds, maxSenders, variant]);

  if (loading) {
    return <div className="flex h-full items-center justify-center text-gray-500">Loading...</div>;
  }

  if (senders.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-gray-500 text-sm">
        No activity data for selected participants
      </div>
    );
  }

  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {variant === 'card' && (
        <p className="shrink-0 text-xs text-gray-500 sm:hidden">Scroll the grid to see all hours</p>
      )}

      <div
        className={
          variant === 'expanded'
            ? 'min-h-0 flex-1 overflow-auto'
            : 'shrink-0 overflow-x-auto overflow-y-hidden'
        }
      >
        <div className="w-full min-w-0">
          {/* Hour labels */}
          <div
            className="mb-1 grid gap-0.5"
            style={{ gridTemplateColumns: `${variant === 'expanded' ? '8rem' : '6rem'} repeat(24, minmax(0, 1fr))` }}
          >
            <div />
            {hours.map((hour) => (
              <div key={hour} className="text-center text-[10px] text-gray-500 sm:text-xs">
                {hour.toString().padStart(2, '0')}
              </div>
            ))}
          </div>

          {/* Rows */}
          {senders.map((sender, rowIdx) => (
            <div
              key={fullSenders[rowIdx]}
              className="mb-0.5 grid items-center gap-0.5"
              style={{ gridTemplateColumns: `${variant === 'expanded' ? '8rem' : '6rem'} repeat(24, minmax(0, 1fr))` }}
            >
              <div
                className={`truncate pr-1 text-xs text-gray-700 ${labelWidth}`}
                title={fullSenders[rowIdx]}
              >
                {sender}
              </div>
              {hours.map((hour) => {
                const cell = cells.find(
                  (d) => d.hour === hour && d.fullSender === fullSenders[rowIdx]
                );
                return (
                  <div
                    key={`${fullSenders[rowIdx]}-${hour}`}
                    className={`${cellHeight} min-w-0 rounded-sm border border-gray-200/80 ${getIntensityColor(cell?.intensity ?? 0)}`}
                    title={`${fullSenders[rowIdx]} at ${hour}:00 — ${cell?.count ?? 0} messages`}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex shrink-0 items-center justify-center text-xs text-gray-600">
        <span className="mr-2">Less</span>
        <div className="flex gap-1">
          {[0, 0.2, 0.4, 0.6, 0.8, 1].map((intensity, index) => (
            <div
              key={index}
              className={`h-3 w-3 rounded-sm border border-gray-200 ${getIntensityColor(intensity)}`}
            />
          ))}
        </div>
        <span className="ml-2">More</span>
      </div>
    </div>
  );
}

export function ActivityHeatmapFullscreen() {
  return <ActivityHeatmap variant="expanded" />;
}
