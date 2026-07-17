'use client';

import React, { useState, useEffect } from 'react';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { useParticipantScope } from '@/hooks/useParticipantScope';

interface URLMetric {
  domain: string;
  count: number;
  sender: string;
  conversation_id: string;
  first_seen: number;
  last_seen: number;
}

interface AggregatedURLData {
  domain: string;
  total_count: number;
  unique_senders: number;
  conversations: number;
  first_seen: number;
  last_seen: number;
}

interface URLDomainChartProps {
  /** Bar chart in card; ranked list for fullscreen. */
  variant?: 'bar' | 'list';
}

function formatDomainLabel(domain: string): string {
  let formatted = domain.replace(/^(https?:\/\/)?(www\.)?/, '');
  if (formatted.includes('tiktok')) return 'TikTok';
  if (formatted.includes('youtube') || formatted.includes('youtu.be')) return 'YouTube';
  if (formatted.includes('instagram')) return 'Instagram';
  if (formatted.includes('facebook')) return 'Facebook';
  if (formatted.includes('google')) return 'Google';
  if (formatted.includes('apple')) return 'Apple Music';
  if (formatted.includes('spotify')) return 'Spotify';
  if (formatted.includes('netflix')) return 'Netflix';
  if (formatted.includes('amazon')) return 'Amazon';
  const firstDot = formatted.indexOf('.');
  if (firstDot > 0 && firstDot < 12) formatted = formatted.substring(0, firstDot);
  else if (formatted.length > 12) formatted = `${formatted.substring(0, 10)}…`;
  return formatted;
}

function getBarColor(domain: string): string {
  if (domain.includes('youtube') || domain.includes('youtu.be')) return '#ff0000';
  if (domain.includes('instagram') || domain.includes('ig')) return '#e4405f';
  if (domain.includes('twitter') || domain.includes('x.com')) return '#1da1f2';
  if (domain.includes('tiktok')) return '#000000';
  if (domain.includes('facebook')) return '#1877f2';
  if (domain.includes('spotify')) return '#1db954';
  if (domain.includes('netflix')) return '#e50914';
  if (domain.includes('amazon')) return '#ff9900';
  if (domain.includes('google')) return '#4285f4';
  if (domain.includes('apple')) return '#007aff';
  return '#6366f1';
}

export function URLDomainChart({ variant = 'bar' }: URLDomainChartProps) {
  const [data, setData] = useState<AggregatedURLData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { filterScopedRows, isFiltered, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch('/data/urlMetrics.json');
        if (!response.ok) {
          throw new Error(`Failed to load URL data: ${response.status}`);
        }
        
        let urlData: URLMetric[] = await response.json();
        urlData = filterScopedRows(urlData, { senderKey: 'sender' });
        
        if (!Array.isArray(urlData)) {
          throw new Error('Invalid URL data format');
        }
        
        // Aggregate URL data by domain
        const domainMap = new Map<string, {
          totalCount: number;
          senders: Set<string>;
          conversations: Set<string>;
          firstSeen: number;
          lastSeen: number;
        }>();

        urlData.forEach(item => {
          if (!domainMap.has(item.domain)) {
            domainMap.set(item.domain, {
              totalCount: 0,
              senders: new Set(),
              conversations: new Set(),
              firstSeen: item.first_seen,
              lastSeen: item.last_seen
            });
          }
          
          const domainStats = domainMap.get(item.domain)!;
          domainStats.totalCount += item.count;
          domainStats.senders.add(item.sender);
          domainStats.conversations.add(item.conversation_id);
          domainStats.firstSeen = Math.min(domainStats.firstSeen, item.first_seen);
          domainStats.lastSeen = Math.max(domainStats.lastSeen, item.last_seen);
        });

        // Convert to array and sort by total count
        const aggregatedData = Array.from(domainMap.entries())
          .map(([domain, stats]) => ({
            domain: domain.replace(/^(https?:\/\/)?(www\.)?/, ''), // Clean up domain names
            total_count: stats.totalCount,
            unique_senders: stats.senders.size,
            conversations: stats.conversations.size,
            first_seen: stats.firstSeen,
            last_seen: stats.lastSeen
          }))
          .sort((a, b) => b.total_count - a.total_count)
          .slice(0, variant === 'list' ? 50 : 12);

        setData(aggregatedData);
      } catch (error) {
        console.error('Error loading URL data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load URL data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds, variant]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading URL data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500 text-center">
          <div>Error loading URL data</div>
          <div className="text-xs mt-1">{error}</div>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <div>No URL data available</div>
          {isFiltered && (
            <div className="text-xs mt-1">
              for selected conversations
            </div>
          )}
        </div>
      </div>
    );
  }

  const chartData = data.map((item) => ({
    ...item,
    displayDomain: formatDomainLabel(item.domain),
    fullDomain: item.domain,
  }));

  if (variant === 'list') {
    const maxCount = data[0]?.total_count || 1;
    return (
      <div className="flex h-full min-h-0 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="space-y-1 pr-1">
            {data.map((item, index) => {
              const barPct = (item.total_count / maxCount) * 100;
              return (
                <div key={item.domain} className="flex items-center gap-2 rounded-md px-1 py-1 hover:bg-gray-50">
                  <span className="w-6 shrink-0 text-right text-xs text-gray-400">{index + 1}</span>
                  <span className="w-40 shrink-0 truncate text-sm font-medium text-gray-900" title={item.domain}>
                    {item.domain}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div
                      className="h-2 rounded-full"
                      style={{
                        width: `${Math.max(barPct, 4)}%`,
                        backgroundColor: getBarColor(item.domain),
                        opacity: 0.75,
                      }}
                    />
                  </div>
                  <span className="w-10 shrink-0 text-right text-xs text-gray-500">
                    {item.total_count}
                  </span>
                  <span className="w-16 shrink-0 text-right text-xs text-gray-400">
                    {item.unique_senders} sender{item.unique_senders !== 1 ? 's' : ''}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={chartData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis 
            dataKey="displayDomain"
            tick={{ fontSize: 10 }}
            height={60}
            interval={0}
            axisLine={{ stroke: '#6b7280' }}
            tickLine={{ stroke: '#6b7280' }}
            angle={-45}
          />
          <YAxis 
            tickFormatter={(value) => value.toLocaleString()}
            tick={{ fontSize: 11 }}
            axisLine={{ stroke: '#6b7280' }}
            tickLine={{ stroke: '#6b7280' }}
            label={{ value: 'Shares', angle: -90, position: 'insideLeft' }}
          />
          <ChartTooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length > 0) {
                const data = payload[0].payload;
                const firstSeenDate = new Date(data.first_seen).toLocaleDateString();
                const lastSeenDate = new Date(data.last_seen).toLocaleDateString();

                return (
                  <div className="p-3 text-sm">
                    <div className="font-semibold text-gray-900">{data.fullDomain}</div>
                    <div className="text-gray-700">Shared: {data.total_count.toLocaleString()} time{data.total_count !== 1 ? 's' : ''}</div>
                    <div className="text-gray-700">By: {data.unique_senders} sender{data.unique_senders !== 1 ? 's' : ''}</div>
                    <div className="text-gray-700">In: {data.conversations} conversation{data.conversations !== 1 ? 's' : ''}</div>
                    <div className="mt-1 border-t pt-1 text-xs text-gray-500">
                      First: {firstSeenDate} | Last: {lastSeenDate}
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar 
            dataKey="total_count" 
            radius={[4, 4, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.domain)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}

export function URLDomainFullscreen() {
  return <URLDomainChart variant="list" />;
}