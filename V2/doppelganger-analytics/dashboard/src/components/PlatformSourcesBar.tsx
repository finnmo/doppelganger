'use client';

import React from 'react';
import { PlatformBadge } from '@/components/PlatformBadge';
import { platformStyles } from '@/lib/platforms';
import type { PlatformSummary } from '@/contexts/ConversationContext';

interface PlatformSourcesBarProps {
  platforms: PlatformSummary[];
  /** When filtering a single conversation, pass its source to highlight it. */
  activeSource?: string | null;
}

/** Shows how messages break down across imported platforms. */
export function PlatformSourcesBar({ platforms, activeSource }: PlatformSourcesBarProps) {
  if (platforms.length <= 1) return null;

  const totalMessages = platforms.reduce((sum, p) => sum + p.messages, 0) || 1;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">Data sources</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            Messages imported from {platforms.length} platforms
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5 justify-end">
          {platforms.map((p) => (
            <PlatformBadge key={p.source} source={p.source} />
          ))}
        </div>
      </div>

      <div className="flex h-2.5 rounded-full overflow-hidden bg-gray-100">
        {platforms.map((p) => {
          const pct = (p.messages / totalMessages) * 100;
          const styles = platformStyles(p.source);
          return (
            <div
              key={p.source}
              className={`${styles.dot} transition-all ${
                activeSource && activeSource !== p.source ? 'opacity-30' : ''
              }`}
              style={{ width: `${pct}%` }}
              title={`${p.label}: ${p.messages.toLocaleString()} messages (${Math.round(pct)}%)`}
            />
          );
        })}
      </div>

      <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
        {platforms.map((p) => {
          const pct = Math.round((p.messages / totalMessages) * 100);
          const dimmed = activeSource && activeSource !== p.source;
          return (
            <div
              key={p.source}
              className={`text-sm ${dimmed ? 'opacity-40' : ''}`}
            >
              <div className="font-medium text-gray-900">{p.label}</div>
              <div className="text-gray-500 tabular-nums">
                {p.messages.toLocaleString()} msgs · {pct}%
              </div>
              <div className="text-xs text-gray-400">
                {p.conversations} conversation{p.conversations === 1 ? '' : 's'}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
