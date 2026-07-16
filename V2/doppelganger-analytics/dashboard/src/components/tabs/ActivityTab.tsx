'use client';

import React from 'react';
import { ActivityHeatmap } from '@/components/ActivityHeatmap';
import { LatencyChart } from '@/components/LatencyChart';
import { DailyActivityPatterns } from '@/components/DailyActivityPatterns';
import { CommunicationFrequencyAnalysis } from '@/components/CommunicationFrequencyAnalysis';
import { ActivityInsightsPanel } from '@/components/ActivityInsightsPanel';
import { ChartCard } from '@/components/ui/ChartCard';
import { Calendar, Zap, Activity } from 'lucide-react';
import { CHART_MD, GRID_GAP, TAB_STACK } from '@/lib/layout';

/**
 * Activity Patterns — single-screen grid.
 * Row 1: hourly heatmap · response times · activity insights.
 * Row 2: daily patterns · communication frequency (self-carded components
 * with their own controls; scroll internally so the page never grows).
 */
export function ActivityTab() {
  return (
    <div className={TAB_STACK}>
      <div className={`grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 ${GRID_GAP}`}>
        <ChartCard
          title="Activity Heatmap by Hour"
          icon={Calendar}
          accent="blue"
          tooltip={{
            description:
              'When people are most active during the day across different senders.',
          }}
          bodyClassName={`${CHART_MD} overflow-y-auto`}
        >
          <ActivityHeatmap />
        </ChartCard>

        <ChartCard
          title="Response Time Distribution"
          icon={Zap}
          accent="green"
          tooltip={{
            description:
              'How quickly people respond to messages across different time buckets.',
          }}
          bodyClassName={`${CHART_MD} overflow-y-auto`}
        >
          <LatencyChart />
        </ChartCard>

        <ChartCard
          title="Activity Insights"
          icon={Activity}
          accent="orange"
          tooltip={{
            description:
              'Key activity statistics computed from your real data: peak hour, busiest day, and time-of-day split.',
          }}
          bodyClassName={`${CHART_MD} overflow-y-auto`}
        >
          <ActivityInsightsPanel />
        </ChartCard>
      </div>

      <div className={`grid grid-cols-1 xl:grid-cols-2 ${GRID_GAP}`}>
        <div className="min-w-0 max-h-96 overflow-y-auto rounded-lg">
          <DailyActivityPatterns />
        </div>
        <div className="min-w-0 max-h-96 overflow-y-auto rounded-lg">
          <CommunicationFrequencyAnalysis />
        </div>
      </div>
    </div>
  );
}
