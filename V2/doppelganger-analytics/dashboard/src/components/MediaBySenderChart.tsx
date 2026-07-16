'use client';

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface SenderMediaData {
  conversation_id: string;
  sender: string;
  photo_count: number;
  video_count: number;
  attachment_count: number;
  total_media: number;
}

interface ProcessedSenderData {
  sender: string;
  total_media: number;
  media_types: Record<string, number>;
  top_type: string;
}

export function MediaBySenderChart() {
  const [data, setData] = useState<ProcessedSenderData[]>([]);
  const [loading, setLoading] = useState(true);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/mediaMetrics.json');
        const mediaData = await response.json();
        
        let senderMediaData: SenderMediaData[] = mediaData.sender_media_data || [];
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          senderMediaData = senderMediaData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
        }
        
        // Aggregate filtered data by sender
        const senderTotals = new Map<string, {
          total_media: number;
          types: Record<string, number>;
        }>();

        senderMediaData.forEach(data => {
          if (!senderTotals.has(data.sender)) {
            senderTotals.set(data.sender, {
              total_media: 0,
              types: { image: 0, video: 0, attachment: 0 }
            });
          }
          
          const senderTotal = senderTotals.get(data.sender)!;
          senderTotal.total_media += data.total_media;
          senderTotal.types.image += data.photo_count;
          senderTotal.types.video += data.video_count;
          senderTotal.types.attachment += data.attachment_count;
        });

        // Convert to chart format and sort
        const processedData: ProcessedSenderData[] = Array.from(senderTotals.entries())
          .map(([sender, data]) => {
            const topType = Object.entries(data.types)
              .sort(([,a], [,b]) => b - a)[0]?.[0] || 'unknown';
            
            return {
              sender: sender,
              total_media: data.total_media,
              media_types: data.types,
              top_type: topType
            };
          })
          .sort((a, b) => b.total_media - a.total_media)
          .slice(0, 10); // Top 10 senders
        
        setData(processedData);
        
      } catch (error) {
        console.error('Error loading media data:', error);
        // Fallback data if file doesn't exist
        const fallbackData: ProcessedSenderData[] = [
          { sender: 'No Data', total_media: 0, media_types: {}, top_type: 'unknown' }
        ];
        setData(fallbackData);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading...</div>;
  }

  const getBarColor = (topType: string): string => {
    const typeColors: Record<string, string> = {
      'image': '#3b82f6',      // Blue
      'video': '#ef4444',      // Red  
      'attachment': '#10b981',  // Green
      'gif': '#8b5cf6',        // Purple
      'sticker': '#f97316',    // Orange
    };
    return typeColors[topType.toLowerCase()] || '#6b7280';
  };

  const formatTooltip = (value: number, data: ProcessedSenderData) => {
    const mediaTypes = Object.entries(data.media_types)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([type, count]) => `${type}: ${count}`)
      .join(', ');
    
    return [
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{data.sender}</div>
        <div>Total Media: {value.toLocaleString()}</div>
        <div>Top Type: {data.top_type}</div>
        <div className="text-xs text-gray-500 mt-1">
          {mediaTypes}
        </div>
      </div>
    ];
  };

  const truncateName = (name: string, maxLength: number = 15): string => {
    return name.length > maxLength ? name.substring(0, maxLength) + '...' : name;
  };

  return (
    <div className="h-96">
      <div className="mb-4 text-center">
        <div className="text-sm text-gray-600">
          Top Media Sharers
        </div>
        <div className="text-xs text-gray-500">
          Most active: {data[0]?.sender || 'N/A'} ({data[0]?.total_media.toLocaleString() || 0} items)
        </div>
      </div>
      
      {isFiltered && (
        <div className="mb-2 text-center text-xs text-blue-600">
          Filtered across {selectedConversations.length} selected conversation{selectedConversations.length !== 1 ? 's' : ''}
        </div>
      )}
      
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={data} 
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="sender"
            stroke="#6b7280"
            fontSize={11}
            angle={-45}
            textAnchor="end"
            height={60}
            tickFormatter={(value) => truncateName(value)}
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickFormatter={(value) => value.toLocaleString()}
          />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                    {formatTooltip(Number(payload[0].value), payload[0].payload)}
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar 
            dataKey="total_media" 
            radius={[4, 4, 0, 0]}
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={getBarColor(entry.top_type)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      <div className="mt-4 text-xs text-gray-500 text-center">
        Bar colors indicate each sender&apos;s most shared media type
        {data.length === 0 && (
          <div className="mt-2 text-orange-600">
            No media sharing data available for selected conversations
          </div>
        )}
      </div>
    </div>
  );
} 