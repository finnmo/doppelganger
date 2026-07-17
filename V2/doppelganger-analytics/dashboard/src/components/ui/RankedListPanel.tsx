'use client';

import React from 'react';

export type RankedTint = 'blue' | 'orange' | 'purple' | 'green';

export interface RankedItem {
  id: string;
  name: string;
  value: string;
  /** Used for relative bar width in expanded view. */
  numericValue: number;
  sub: React.ReactNode;
}

interface RankedListPanelProps {
  items: RankedItem[];
  tint: RankedTint;
  /** Compact rows in card; scrollable bar chart list for fullscreen. */
  variant?: 'card' | 'expanded';
  limit?: number;
  emptyMessage?: string;
}

const ROW_TINT: Record<RankedTint, { row: string; badge: string; value: string; bar: string }> = {
  blue: { row: 'bg-blue-50 border-blue-100', badge: 'bg-blue-600', value: 'text-blue-600', bar: 'bg-blue-400' },
  orange: { row: 'bg-orange-50 border-orange-100', badge: 'bg-orange-600', value: 'text-orange-600', bar: 'bg-orange-400' },
  purple: { row: 'bg-purple-50 border-purple-100', badge: 'bg-purple-600', value: 'text-purple-600', bar: 'bg-purple-400' },
  green: { row: 'bg-green-50 border-green-100', badge: 'bg-green-600', value: 'text-green-600', bar: 'bg-green-400' },
};

function truncateName(name: string, max = 22): string {
  return name.length > max ? `${name.substring(0, max)}…` : name;
}

function RankedRow({
  rank,
  name,
  value,
  sub,
  tint,
}: {
  rank: number;
  name: string;
  value: string;
  sub: React.ReactNode;
  tint: RankedTint;
}) {
  const t = ROW_TINT[tint];
  return (
    <div className={`flex items-center justify-between rounded-md border p-2 ${t.row}`}>
      <div className="flex min-w-0 items-center">
        <div className={`mr-2 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold text-white ${t.badge}`}>
          {rank}
        </div>
        <span className="truncate text-sm font-medium text-gray-800" title={name}>
          {truncateName(name)}
        </span>
      </div>
      <div className="ml-2 shrink-0 text-right">
        <div className={`text-sm font-bold ${t.value}`}>{value}</div>
        <div className="text-xs text-gray-600">{sub}</div>
      </div>
    </div>
  );
}

export function RankedListPanel({
  items,
  tint,
  variant = 'card',
  limit,
  emptyMessage = 'No data available.',
}: RankedListPanelProps) {
  const visible = limit ? items.slice(0, limit) : items;

  if (visible.length === 0) {
    return <p className="text-sm italic text-gray-500">{emptyMessage}</p>;
  }

  if (variant === 'expanded') {
    const maxVal = Math.max(...visible.map((i) => i.numericValue), 1);
    const t = ROW_TINT[tint];

    return (
      <div className="flex h-full min-h-0 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto pr-1">
          <div className="space-y-1">
            {visible.map((item, index) => {
              const barPct = (item.numericValue / maxVal) * 100;
              return (
                <div
                  key={item.id}
                  className="flex items-center gap-3 rounded-md px-2 py-2 hover:bg-gray-50"
                >
                  <span className="w-8 shrink-0 text-right text-sm font-medium text-gray-400">
                    {index + 1}
                  </span>
                  <span className="w-48 shrink-0 truncate text-base font-medium text-gray-900" title={item.name}>
                    {item.name}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div
                      className={`h-3 rounded-full opacity-80 ${t.bar}`}
                      style={{ width: `${Math.max(barPct, 3)}%` }}
                    />
                  </div>
                  <div className="w-24 shrink-0 text-right">
                    <div className={`text-base font-bold ${t.value}`}>{item.value}</div>
                    <div className="text-xs text-gray-500">{item.sub}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <p className="shrink-0 pt-2 text-xs text-gray-400">
          Showing {visible.length} participant{visible.length !== 1 ? 's' : ''}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {visible.map((item, index) => (
        <RankedRow
          key={item.id}
          rank={index + 1}
          name={item.name}
          value={item.value}
          sub={item.sub}
          tint={tint}
        />
      ))}
    </div>
  );
}
