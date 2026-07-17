'use client';

import React from 'react';
import { ActivityHeatmap, ActivityHeatmapFullscreen } from '@/components/ActivityHeatmap';
import { LatencyChart } from '@/components/LatencyChart';
import { DailyActivityPatterns } from '@/components/DailyActivityPatterns';
import { CommunicationFrequencyAnalysis } from '@/components/CommunicationFrequencyAnalysis';
import { ActivityInsightsPanel } from '@/components/ActivityInsightsPanel';
import { ChartCard } from '@/components/ui/ChartCard';
import { Calendar, Zap, Activity } from 'lucide-react';
import { TAB_VIEWPORT, CARD_GRID_ROW, CARD_FILL, BODY_FILL } from '@/lib/layout';

/**
 * Activity Patterns — single-screen grid.
 * Row 1: hourly heatmap · response times · activity insights.
 * Row 2: daily patterns · communication frequency.
 */
export function ActivityTab() {
  return (
    <div className={TAB_VIEWPORT}>
      <div className={CARD_GRID_ROW('grid-cols-1 lg:grid-cols-2 xl:grid-cols-3')}>
        <ChartCard
          title="Activity Heatmap by Hour"
          icon={Calendar}
          accent="blue"
          className={CARD_FILL}
          tooltip={{
            description:
              'When people are most active during the day across different senders.',
          }}
          bodyClassName={BODY_FILL}
          fullscreenChildren={<ActivityHeatmapFullscreen />}
        >
          <ActivityHeatmap />
        </ChartCard>

        <ChartCard
          title="Response Time Distribution"
          icon={Zap}
          accent="green"
          className={CARD_FILL}
          tooltip={{
            description:
              'How quickly people respond to messages across different time buckets.',
          }}
          bodyClassName={BODY_FILL}
        >
          <LatencyChart />
        </ChartCard>

        <ChartCard
          title="Activity Insights"
          icon={Activity}
          accent="orange"
          className={CARD_FILL}
          enableFullscreen={false}
          tooltip={{
            description:
              'Key activity statistics computed from your real data: peak hour, busiest day, and time-of-day split.',
          }}
          bodyClassName={BODY_FILL}
        >
          <ActivityInsightsPanel />
        </ChartCard>
      </div>

      <div className={CARD_GRID_ROW('grid-cols-1 xl:grid-cols-2')}>
        <div className="min-h-0 h-full overflow-hidden">
          <DailyActivityPatterns />
        </div>
        <div className="min-h-0 h-full overflow-hidden">
          <CommunicationFrequencyAnalysis />
        </div>
      </div>
    </div>
  );
}
