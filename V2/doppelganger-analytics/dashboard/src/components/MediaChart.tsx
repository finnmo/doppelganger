'use client';

import React, { useState, useEffect } from 'react';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend } from 'recharts';
import { useParticipantScope } from '@/hooks/useParticipantScope';

interface AttachmentData {
  conversation_id: string;
  month: string;
  sender: string;
  photo_count: number;
  video_count: number;
  total_count: number;
}

interface MonthlyMedia {
  month: string;
  photos: number;
  videos: number;
  total: number;
}

export function MediaChart() {
  const [data, setData] = useState<MonthlyMedia[]>([]);
  const [loading, setLoading] = useState(true);
  const { filterScopedRows, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/attachmentTimeSeries.json');
        let attachmentData: AttachmentData[] = await response.json();
        attachmentData = filterScopedRows(attachmentData, { senderKey: 'sender' });
        
        // Group by month and sum across filtered conversations
        const monthlyTotals = attachmentData.reduce((acc, item) => {
          if (!acc[item.month]) {
            acc[item.month] = { photos: 0, videos: 0 };
          }
          acc[item.month].photos += item.photo_count;
          acc[item.month].videos += item.video_count;
          return acc;
        }, {} as Record<string, { photos: number; videos: number }>);
        
        // Convert to array and sort by date
        const processedData = Object.entries(monthlyTotals)
          .map(([month, counts]) => ({
            month: new Date(month + '-01').toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'short' 
            }),
            photos: counts.photos,
            videos: counts.videos,
            total: counts.photos + counts.videos,
            sortKey: month
          }))
          .sort((a, b) => a.sortKey.localeCompare(b.sortKey))
          .slice(-24); // Last 24 months for readability
        
        setData(processedData);
      } catch (error) {
        console.error('Error loading media data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading...</div>;
  }

  return (
    <div className="h-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="month" 
            stroke="#6b7280"
            fontSize={12}
            interval="preserveStartEnd"
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickFormatter={(value) => value.toLocaleString()}
          />
          <ChartTooltip 
            formatter={(value, name) => {
              const label = name === 'photos' ? 'Photos' : name === 'videos' ? 'Videos' : 'Total';
              return [Number(value).toLocaleString(), label];
            }}
            labelStyle={{ color: '#374151' }}
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e5e7eb',
              borderRadius: '8px'
            }}
          />
          <Legend />
          <Area
            type="monotone"
            dataKey="photos"
            stackId="1"
            stroke="#10b981"
            fill="#10b981"
            fillOpacity={0.6}
            name="Photos"
          />
          <Area
            type="monotone"
            dataKey="videos"
            stackId="1"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.6}
            name="Videos"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
} 