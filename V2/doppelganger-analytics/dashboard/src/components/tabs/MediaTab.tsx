'use client';

import React from 'react';
import { MediaChart } from '@/components/MediaChart';
import { AttachmentTypeChart } from '@/components/AttachmentTypeChart';
import { MediaBySenderChart } from '@/components/MediaBySenderChart';
import ReactionMetricsChart from '@/components/ReactionMetricsChart';
import { MediaInsightsPanel } from '@/components/MediaInsightsPanel';
import { ChartCard } from '@/components/ui/ChartCard';
import { Video, ThumbsUp, Camera, FileImage, Sparkles } from 'lucide-react';
import { CHART_MD, CHART_LG, GRID_GAP, TAB_STACK } from '@/lib/layout';

/**
 * Media & Engagement — single-screen grid.
 * Row 1: media trends · media by sender · attachment types.
 * Row 2: reaction metrics dashboard (wide) · media insights.
 */
export function MediaTab() {
  return (
    <div className={TAB_STACK}>
      <div className={`grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 ${GRID_GAP}`}>
        <ChartCard
          title="Media Sharing Trends"
          icon={Camera}
          accent="blue"
          tooltip={{
            description:
              'Photos and videos shared over time showing content patterns across your conversations.',
          }}
          bodyClassName={CHART_MD}
        >
          <MediaChart />
        </ChartCard>

        <ChartCard
          title="Media Breakdown by Sender"
          icon={FileImage}
          accent="green"
          tooltip={{
            description:
              'Who shares the most visual content and what types they prefer.',
          }}
          bodyClassName={CHART_MD}
        >
          <MediaBySenderChart />
        </ChartCard>

        <ChartCard
          title="Attachment Type Analysis"
          icon={Video}
          accent="red"
          tooltip={{
            description:
              'Breakdown of different media types and formats shared in your conversations.',
          }}
          bodyClassName={CHART_MD}
        >
          <AttachmentTypeChart />
        </ChartCard>
      </div>

      <div className={`grid grid-cols-1 xl:grid-cols-3 ${GRID_GAP}`}>
        <ChartCard
          title="Reaction Metrics Dashboard"
          icon={ThumbsUp}
          accent="yellow"
          tooltip={{
            description:
              'Message reactions, engagement patterns, and most reacted content.',
          }}
          className="xl:col-span-2"
          bodyClassName={`${CHART_LG} overflow-y-auto`}
        >
          <ReactionMetricsChart />
        </ChartCard>

        <ChartCard
          title="Media Insights"
          icon={Sparkles}
          accent="orange"
          tooltip={{
            description:
              'Key media statistics computed from your real data: share rate, top sharer, and preferred formats.',
          }}
          bodyClassName={`${CHART_LG} overflow-y-auto`}
        >
          <MediaInsightsPanel />
        </ChartCard>
      </div>
    </div>
  );
}
