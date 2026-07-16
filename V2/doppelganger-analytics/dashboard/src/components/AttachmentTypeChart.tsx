'use client';

import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface AttachmentTypeData {
  conversation_id: string;
  type: string;
  count: number;
  percentage: number;
  examples: string[];
}

interface AggregatedAttachmentType {
  type: string;
  count: number;
  percentage: number;
  examples: string[];
}

const COLORS = {
  'image': '#3b82f6',      // Blue
  'video': '#ef4444',      // Red
  'audio': '#10b981',      // Green
  'link': '#0ea5e9',       // Sky
  'document': '#f59e0b',   // Amber
  'gif': '#8b5cf6',        // Purple
  'sticker': '#f97316',    // Orange
  'other': '#6b7280'       // Gray
};

export function AttachmentTypeChart() {
  const [data, setData] = useState<AggregatedAttachmentType[]>([]);
  const [loading, setLoading] = useState(true);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/attachmentTypeMetrics.json');
        let attachmentData: AttachmentTypeData[] = await response.json();
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          attachmentData = attachmentData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
        }
        
        // Aggregate by attachment type across filtered conversations
        const typeMap = new Map<string, {
          count: number;
          examples: Set<string>;
        }>();

        attachmentData.forEach(item => {
          if (!typeMap.has(item.type)) {
            typeMap.set(item.type, { count: 0, examples: new Set() });
          }
          
          const typeData = typeMap.get(item.type)!;
          typeData.count += item.count;
          item.examples.forEach(ex => typeData.examples.add(ex));
        });

        // Calculate total and percentages
        const totalCount = Array.from(typeMap.values()).reduce((sum, data) => sum + data.count, 0);
        
        // Convert to final format
        const processedData = Array.from(typeMap.entries()).map(([type, data]) => ({
          type,
          count: data.count,
          percentage: totalCount > 0 ? (data.count / totalCount) * 100 : 0,
          examples: Array.from(data.examples).slice(0, 3) // Limit to 3 examples
        }));

        // Sort by count and take top 10
        const sortedData = processedData
          .sort((a, b) => b.count - a.count)
          .slice(0, 10);
        
        setData(sortedData);
      } catch (error) {
        console.error('Error loading attachment data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading...</div>;
  }

  const getColor = (type: string): string => {
    const lowerType = type.toLowerCase();
    for (const [key, color] of Object.entries(COLORS)) {
      if (lowerType.includes(key)) {
        return color;
      }
    }
    return COLORS.other;
  };

  const formatTooltip = (value: number, data: AggregatedAttachmentType) => {
    return [
      <div key="tooltip" className="text-sm">
        <div className="font-semibold capitalize">{data.type}</div>
        <div>Count: {value.toLocaleString()}</div>
        <div>Percentage: {data.percentage?.toFixed(1)}%</div>
        {data.examples && data.examples.length > 0 && (
          <div className="text-xs text-gray-500 mt-1">
            Examples: {data.examples.slice(0, 2).join(', ')}
          </div>
        )}
      </div>
    ];
  };

  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: {
    cx: number;
    cy: number;
    midAngle: number;
    innerRadius: number;
    outerRadius: number;
    percent: number;
  }) => {
    if (percent < 0.05) return null; // Don't show labels for slices smaller than 5%
    
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        fontSize={12}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  const totalCount = data.reduce((sum, item) => sum + item.count, 0);

  return (
    <div className="flex h-full flex-col">
      <div className="mb-2 shrink-0 text-center">
        <div className="text-sm text-gray-600">
          Total Attachments: {totalCount.toLocaleString()}
        </div>
        <div className="text-xs text-gray-500">
          Most shared: {data[0]?.type || 'N/A'} ({data[0]?.count.toLocaleString() || 0})
        </div>
        {isFiltered && (
          <div className="text-xs text-blue-600 mt-1">
            Filtered across {selectedConversations.length} selected conversation{selectedConversations.length !== 1 ? 's' : ''}
          </div>
        )}
      </div>
      
      <div className="min-h-0 flex-1">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderCustomLabel}
            outerRadius={80}
            fill="#8884d8"
            dataKey="count"
            nameKey="type"
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={getColor(entry.type)}
              />
            ))}
          </Pie>
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
          <Legend 
            verticalAlign="bottom" 
            height={36}
            formatter={(value, entry) => {
              // Find the corresponding data entry
              const dataEntry = data.find(d => d.type === value);
              const percentage = dataEntry?.percentage || 0;
              return (
                <span style={{ color: entry.color }} className="text-sm capitalize">
                  {value} ({percentage.toFixed(1)}%)
                </span>
              );
            }}
          />
        </PieChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}