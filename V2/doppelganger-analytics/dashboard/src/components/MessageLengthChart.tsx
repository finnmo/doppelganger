'use client';

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface MessageLengthDistribution {
  conversation_id: string;
  bucket: string;
  count: number;
  percentage: number;
  label: string;
  minWords: number;
  maxWords: number;
}

interface LengthBucket {
  range: string;
  count: number;
  percentage: number;
  label: string;
  minWords: number;
  maxWords: number;
}

export function MessageLengthChart() {
  const [data, setData] = useState<LengthBucket[]>([]);
  const [totalMessages, setTotalMessages] = useState(0);
  const [averageLength, setAverageLength] = useState(0);
  const [loading, setLoading] = useState(true);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/messageLengthDistribution.json');
        let lengthData: MessageLengthDistribution[] = await response.json();
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          lengthData = lengthData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
        }
        
        // Aggregate by bucket across filtered conversations
        const bucketMap = new Map<string, {
          count: number;
          label: string;
          minWords: number;
          maxWords: number;
        }>();

        lengthData.forEach(item => {
          if (!bucketMap.has(item.bucket)) {
            bucketMap.set(item.bucket, {
              count: 0,
              label: item.label,
              minWords: item.minWords,
              maxWords: item.maxWords
            });
          }
          
          const bucketData = bucketMap.get(item.bucket)!;
          bucketData.count += item.count;
        });

        // Calculate total and percentages
        const totalMsgs = Array.from(bucketMap.values()).reduce((sum, data) => sum + data.count, 0);
        
        // Convert to final format (preserve bucket order)
        const bucketOrder = ['1-2', '3-5', '6-10', '11-20', '21-50', '50+'];
        const processedData: LengthBucket[] = bucketOrder.map(bucket => {
          const data = bucketMap.get(bucket);
          return {
            range: bucket,
            count: data?.count || 0,
            percentage: totalMsgs > 0 ? ((data?.count || 0) / totalMsgs) * 100 : 0,
            label: data?.label || bucket,
            minWords: data?.minWords || 1,
            maxWords: data?.maxWords || 2
          };
        }).filter(bucket => bucket.count > 0); // Only show buckets with data

        // Calculate average word count based on distribution
        let totalWords = 0;
        processedData.forEach(bucket => {
          const avgWordsInBucket = bucket.maxWords === 999 ? 75 : (bucket.minWords + bucket.maxWords) / 2;
          totalWords += bucket.count * avgWordsInBucket;
        });
        const avgLength = totalMsgs > 0 ? totalWords / totalMsgs : 0;
        
        setData(processedData);
        setTotalMessages(totalMsgs);
        setAverageLength(avgLength);
        
      } catch (error) {
        console.error('Error loading text metrics:', error);
        // Fallback to empty data
        setData([]);
        setTotalMessages(0);
        setAverageLength(0);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading...</div>;
  }

  const getBarColor = (range: string): string => {
    if (range.includes('1-2')) return '#dbeafe'; // Very light blue
    if (range.includes('3-5')) return '#93c5fd'; // Light blue
    if (range.includes('6-10')) return '#3b82f6'; // Blue
    if (range.includes('11-20')) return '#1d4ed8'; // Dark blue
    if (range.includes('21-50')) return '#1e40af'; // Darker blue
    return '#1e3a8a'; // Darkest blue
  };

  const formatTooltip = (value: number, data: LengthBucket) => {
    return [
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{data.label}</div>
        <div>Messages: {value.toLocaleString()}</div>
        <div>Percentage: {data.percentage.toFixed(1)}%</div>
        <div className="text-xs text-gray-500 mt-1">
          Word range: {data.minWords}-{data.maxWords === 999 ? '50+' : data.maxWords}
        </div>
      </div>
    ];
  };

  const mostCommonRange = data.length > 0 ? data.reduce((prev, current) => 
    prev.count > current.count ? prev : current
  ) : { range: 'N/A', percentage: 0 };

  return (
    <div className="h-80">
      {/* Stats Header */}
      <div className="mb-3 grid grid-cols-3 gap-4 text-center">
        <div className="bg-gray-50 p-2 rounded">
          <div className="text-lg font-bold text-blue-600">{averageLength.toFixed(1)}</div>
          <div className="text-xs text-gray-500">Avg Words/Message</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <div className="text-lg font-bold text-green-600">{totalMessages.toLocaleString()}</div>
          <div className="text-xs text-gray-500">Total Messages</div>
        </div>
        <div className="bg-gray-50 p-2 rounded">
          <div className="text-lg font-bold text-purple-600">{mostCommonRange.range}</div>
          <div className="text-xs text-gray-500">Most Common Length</div>
        </div>
      </div>
      
      {isFiltered && (
        <div className="mb-2 text-center text-xs text-blue-600">
          Filtered across {selectedConversations.length} selected conversation{selectedConversations.length !== 1 ? 's' : ''}
        </div>
      )}
      
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart 
            data={data} 
            margin={{ top: 5, right: 30, left: 20, bottom: 40 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="range"
              stroke="#6b7280"
              fontSize={12}
              label={{ value: 'Message Length (words)', position: 'insideBottom', offset: -10 }}
            />
            <YAxis 
              stroke="#6b7280"
              fontSize={12}
              tickFormatter={(value) => {
                if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
                if (value >= 1000) return `${(value / 1000).toFixed(0)}K`;
                return value.toString();
              }}
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
              dataKey="count" 
              radius={[4, 4, 0, 0]}
            >
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={getBarColor(entry.range)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-2 text-center">
        <div className="text-xs text-gray-500">
          {data.length > 0 ? (
            `Most messages are short (${mostCommonRange.percentage.toFixed(1)}% are ${mostCommonRange.range} words)`
          ) : (
            'No message length data available for selected conversations'
          )}
        </div>
      </div>
    </div>
  );
} 