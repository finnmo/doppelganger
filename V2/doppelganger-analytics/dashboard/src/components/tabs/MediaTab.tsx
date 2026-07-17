'use client';

import React from 'react';
import { MediaChart } from '@/components/MediaChart';
import { AttachmentTypeChart } from '@/components/AttachmentTypeChart';
import { MediaBySenderChart } from '@/components/MediaBySenderChart';
import ReactionMetricsChart from '@/components/ReactionMetricsChart';
import { MediaInsightsPanel } from '@/components/MediaInsightsPanel';
import { ChartCard } from '@/components/ui/ChartCard';
import { Video, ThumbsUp, Camera, FileImage, Sparkles } from 'lucide-react';
import { TAB_VIEWPORT, CARD_GRID_ROW, CARD_FILL, BODY_FILL } from '@/lib/layout';

/**
 * Media & Engagement — single-screen grid.
 * Row 1: media trends · media by sender · attachment types.
 * Row 2: reaction metrics dashboard (wide) · media insights.
 */
export function MediaTab() {
  return (
    <div className={TAB_VIEWPORT}>
      <div className={CARD_GRID_ROW('grid-cols-1 lg:grid-cols-2 xl:grid-cols-3')}>
        <ChartCard
          title="Media Sharing Trends"
          icon={Camera}
          accent="blue"
          className={CARD_FILL}
          tooltip={{
            description:
              'Photos and videos shared over time showing content patterns across your conversations.',
          }}
          bodyClassName={BODY_FILL}
        >
          <MediaChart />
        </ChartCard>

        <ChartCard
          title="Media Breakdown by Sender"
          icon={FileImage}
          accent="green"
          className={CARD_FILL}
          tooltip={{
            description:
              'Who shares the most visual content and what types they prefer.',
          }}
          bodyClassName={BODY_FILL}
        >
          <MediaBySenderChart />
        </ChartCard>

        <ChartCard
          title="Attachment Type Analysis"
          icon={Video}
          accent="red"
          className={CARD_FILL}
          tooltip={{
            description:
              'Breakdown of different media types and formats shared in your conversations.',
          }}
          bodyClassName={BODY_FILL}
        >
          <AttachmentTypeChart />
        </ChartCard>
      </div>

      <div className={CARD_GRID_ROW('grid-cols-1 xl:grid-cols-3')}>
        <ChartCard
          title="Reaction Metrics Dashboard"
          icon={ThumbsUp}
          accent="yellow"
          className={`${CARD_FILL} xl:col-span-2`}
          tooltip={{
            description:
              'Message reactions, engagement patterns, and most reacted content.',
          }}
          bodyClassName={BODY_FILL}
        >
          <ReactionMetricsChart />
        </ChartCard>

        <ChartCard
          title="Media Insights"
          icon={Sparkles}
          accent="orange"
          className={CARD_FILL}
          enableFullscreen={false}
          tooltip={{
            description:
              'Key media statistics computed from your real data: share rate, top sharer, and preferred formats.',
          }}
          bodyClassName={BODY_FILL}
        >
          <MediaInsightsPanel />
        </ChartCard>
      </div>
    </div>
  );
}
