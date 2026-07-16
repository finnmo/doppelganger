'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface ActivityData {
  conversation_id: string;
  hour: string;
  sender: string;
  count: number;
}

interface HeatmapCell {
  hour: number;
  sender: string;
  count: number;
  intensity: number;
}

export function ActivityHeatmap() {
  const [data, setData] = useState<HeatmapCell[]>([]);
  const [senders, setSenders] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/activeHours.json');
        let activityData: ActivityData[] = await response.json();
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          activityData = activityData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
        }
        
        // Aggregate by sender and hour across filtered conversations
        const aggregatedData = activityData.reduce((acc, item) => {
          const key = `${item.sender}-${item.hour}`;
          if (!acc[key]) {
            acc[key] = {
              sender: item.sender,
              hour: item.hour,
              count: 0
            };
          }
          acc[key].count += item.count;
          return acc;
        }, {} as Record<string, { sender: string; hour: string; count: number }>);
        
        const processedActivityData = Object.values(aggregatedData);
        
        // Get top 10 most active senders
        const senderCounts = processedActivityData.reduce((acc, item) => {
          acc[item.sender] = (acc[item.sender] || 0) + item.count;
          return acc;
        }, {} as Record<string, number>);
        
        const topSenders = Object.entries(senderCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10)
          .map(([sender]) => sender);
        
        // Filter data for top senders and create heatmap data
        const filteredData = processedActivityData.filter(item => topSenders.includes(item.sender));
        
        // Find max count for normalization
        const maxCount = Math.max(...filteredData.map(item => item.count));
        
        // Create heatmap cells
        const heatmapData: HeatmapCell[] = [];
        for (let hour = 0; hour < 24; hour++) {
          for (const sender of topSenders) {
            const item = filteredData.find(d => 
              parseInt(d.hour) === hour && d.sender === sender
            );
            heatmapData.push({
              hour,
              sender: sender.length > 12 ? sender.substring(0, 12) + '...' : sender,
              count: item ? item.count : 0,
              intensity: item ? item.count / maxCount : 0
            });
          }
        }
        
        setData(heatmapData);
        setSenders(topSenders.map(s => s.length > 12 ? s.substring(0, 12) + '...' : s));
      } catch (error) {
        console.error('Error loading activity data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading...</div>;
  }

  const getIntensityColor = (intensity: number) => {
    if (intensity === 0) return 'bg-gray-100';
    if (intensity < 0.2) return 'bg-blue-200';
    if (intensity < 0.4) return 'bg-blue-300';
    if (intensity < 0.6) return 'bg-blue-400';
    if (intensity < 0.8) return 'bg-blue-500';
    return 'bg-blue-600';
  };

  const hours = Array.from({ length: 24 }, (_, i) => i);

  return (
    <div className="overflow-x-auto">
      <div className="min-w-max">
        {/* Hour labels */}
        <div className="flex mb-2">
          <div className="w-24"></div> {/* Space for sender names */}
          {hours.map(hour => (
            <div key={hour} className="w-8 text-xs text-center text-gray-600">
              {hour.toString().padStart(2, '0')}
            </div>
          ))}
        </div>
        
        {/* Heatmap grid */}
        {senders.map(sender => (
          <div key={sender} className="flex items-center mb-1">
            <div className="w-24 text-xs text-gray-700 pr-2 truncate" title={sender}>
              {sender}
            </div>
            {hours.map(hour => {
              const cell = data.find(d => d.hour === hour && d.sender === sender);
              return (
                <div
                  key={`${sender}-${hour}`}
                  className={`w-8 h-6 mr-0.5 rounded-sm ${getIntensityColor(cell?.intensity || 0)} border border-gray-200`}
                  title={`${sender} at ${hour}:00 - ${cell?.count || 0} messages`}
                />
              );
            })}
          </div>
        ))}
        
        {/* Legend */}
        <div className="mt-4 flex items-center text-xs text-gray-600">
          <span className="mr-2">Less</span>
          <div className="flex space-x-1">
            {[0, 0.2, 0.4, 0.6, 0.8, 1].map((intensity, index) => (
              <div
                key={index}
                className={`w-3 h-3 rounded-sm ${getIntensityColor(intensity)} border border-gray-200`}
              />
            ))}
          </div>
          <span className="ml-2">More</span>
        </div>
      </div>
    </div>
  );
} 