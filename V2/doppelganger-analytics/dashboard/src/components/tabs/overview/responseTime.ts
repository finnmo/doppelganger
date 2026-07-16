// Response-time formatting/aggregation helpers for the Overview tab.

import type { ReplyLatencyItem, TurnTakingData } from './types';

function formatMs(avgMs: number): string {
  const avgMinutes = avgMs / (1000 * 60);
  if (avgMinutes < 1) return `${(avgMs / 1000).toFixed(0)}s`;
  if (avgMinutes < 60) return `${avgMinutes.toFixed(1)}m`;
  return `${(avgMinutes / 60).toFixed(1)}h`;
}

/** Average response time from latency distribution buckets. */
export function averageResponseTimeFromLatency(latencyData: ReplyLatencyItem[]): string {
  if (!latencyData || latencyData.length === 0) return '0s';

  const bucketMidpoints: Record<string, number> = {
    '0-10s': 5000,
    '10-30s': 20000,
    '30-60s': 45000,
    '1-5m': 180000,
    '5-15m': 600000,
    '15-60m': 2250000,
    '>1h': 10800000
  };

  let totalWeightedTime = 0;
  let totalCount = 0;
  for (const item of latencyData) {
    const midpoint = bucketMidpoints[item.bucket];
    if (midpoint) {
      totalWeightedTime += midpoint * item.count;
      totalCount += item.count;
    }
  }

  if (totalCount === 0) return '0s';
  return formatMs(totalWeightedTime / totalCount);
}

/**
 * Overall average response time, averaged at the participant level (matching
 * how individual "fast responders" are computed, avoiding Simpson's paradox
 * against conversation-level averages). Falls back to latency buckets.
 */
export function overallAverageResponseTime(
  turnTakingData: TurnTakingData | null,
  selectedConversations: string[],
  isFiltered: boolean,
  fallbackLatency: ReplyLatencyItem[]
): string {
  if (turnTakingData?.conversation_patterns) {
    const selectedIds = !isFiltered || selectedConversations.length === 0
      ? turnTakingData.conversation_patterns.map(p => p.conversation_id)
      : selectedConversations;

    const times: number[] = [];
    for (const pattern of turnTakingData.conversation_patterns) {
      if (!selectedIds.includes(pattern.conversation_id)) continue;
      for (const participant of pattern.participants) {
        if (participant.avg_response_time > 0) {
          times.push(participant.avg_response_time);
        }
      }
    }

    if (times.length > 0) {
      return formatMs(times.reduce((sum, t) => sum + t, 0) / times.length);
    }
  }

  return averageResponseTimeFromLatency(fallbackLatency);
}
