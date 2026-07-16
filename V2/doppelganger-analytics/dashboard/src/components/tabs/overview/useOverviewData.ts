'use client';

import { useState, useEffect } from 'react';
import type { ActiveHourRecord } from '@/lib/overviewAggregate';
import type {
  EmojiMetricsData,
  EngagementData,
  RawConversationData,
  RawMediaData,
  RawTextData,
  RawTimeData,
  ReplyLatencyItem,
  TurnTakingData
} from './types';

export interface OverviewData {
  text: RawTextData | null;
  conversation: RawConversationData | null;
  media: RawMediaData | null;
  time: RawTimeData | null;
  latency: ReplyLatencyItem[] | null;
  engagement: EngagementData | null;
  turnTaking: TurnTakingData | null;
  emoji: EmojiMetricsData | null;
  activeHours: ActiveHourRecord[] | null;
}

const EMPTY: OverviewData = {
  text: null,
  conversation: null,
  media: null,
  time: null,
  latency: null,
  engagement: null,
  turnTaking: null,
  emoji: null,
  activeHours: null
};

/** Loads every JSON file the Overview tab needs, once. */
export function useOverviewData(): { data: OverviewData; loading: boolean } {
  const [data, setData] = useState<OverviewData>(EMPTY);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      // emojiMetrics/activeHours may not exist in pre-regeneration exports;
      // they resolve to null instead of failing the whole tab.
      const optional = async <T,>(path: string): Promise<T | null> => {
        try {
          const res = await fetch(path);
          if (!res.ok) return null;
          return (await res.json()) as T;
        } catch {
          return null;
        }
      };

      try {
        const [text, conversation, media, time, latency, engagement, turnTaking, emoji, activeHours] =
          await Promise.all([
            fetch('/data/textMetrics.json').then(r => r.json()),
            fetch('/data/conversationMetrics.json').then(r => r.json()),
            fetch('/data/mediaMetrics.json').then(r => r.json()),
            fetch('/data/timeMetrics.json').then(r => r.json()),
            fetch('/data/replyLatencyDistribution.json').then(r => r.json()),
            fetch('/data/engagementScoring.json').then(r => r.json()),
            fetch('/data/turnTakingAnalysis.json').then(r => r.json()),
            optional<EmojiMetricsData>('/data/emojiMetrics.json'),
            optional<ActiveHourRecord[]>('/data/activeHours.json')
          ]);

        setData({
          text,
          conversation,
          media,
          time,
          latency,
          engagement,
          turnTaking,
          emoji,
          activeHours: Array.isArray(activeHours) ? activeHours : null
        });
      } catch (error) {
        console.error('Error loading overview data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  return { data, loading };
}
