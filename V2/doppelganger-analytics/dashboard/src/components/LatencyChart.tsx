'use client';

import React, { useState, useEffect } from 'react';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { useParticipantScope } from '@/hooks/useParticipantScope';

interface LatencyData {
  conversation_id: string;
  bucket: string;
  count: number;
}

interface ProcessedLatencyData {
  bucket: string;
  count: number;
}

export function LatencyChart() {
  const [data, setData] = useState<ProcessedLatencyData[]>([]);
  const [loading, setLoading] = useState(true);
  const { filterScopedRows, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/replyLatencyDistribution.json');
        let latencyData: LatencyData[] = await response.json();
        latencyData = filterScopedRows(latencyData);
        
        // Aggregate counts by bucket across filtered conversations
        const bucketTotals = latencyData.reduce((acc, item) => {
          if (!acc[item.bucket]) {
            acc[item.bucket] = 0;
          }
          acc[item.bucket] += item.count;
          return acc;
        }, {} as Record<string, number>);
        
        // Sort buckets in logical order
        const bucketOrder = ['0-10s', '10-30s', '30-60s', '1-5m', '5-15m', '15-60m', '>1h'];
        const sortedData = bucketOrder
          .map(bucket => ({
            bucket,
            count: bucketTotals[bucket] || 0
          }))
          .filter(item => item.count > 0);
        
        setData(sortedData);
      } catch (error) {
        console.error('Error loading latency data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading...</div>;
  }

  const getBarColor = (bucket: string) => {
    switch (bucket) {
      case '0-10s': return '#10b981';    // Fast - Green
      case '10-30s': return '#84cc16';   // Quick - Light Green
      case '30-60s': return '#eab308';   // Medium - Yellow
      case '1-5m': return '#f97316';     // Slow - Orange
      case '5-15m': return '#ef4444';    // Very Slow - Red
      case '15-60m': return '#dc2626';   // Extremely Slow - Dark Red
      case '>1h': return '#991b1b';      // Dead - Very Dark Red
      default: return '#6b7280';
    }
  };

  return (
    <div className="h-full min-h-0">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="bucket" 
            stroke="#6b7280"
            fontSize={12}
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickFormatter={(value) => value.toLocaleString()}
          />
          <ChartTooltip 
            formatter={(value) => [Number(value).toLocaleString(), 'Reply Count']}
            labelStyle={{ color: '#374151' }}
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e5e7eb',
              borderRadius: '8px'
            }}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.bucket)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
} 