'use client';

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { format, parseISO } from 'date-fns';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface MonthlyMessageData {
  conversation_id: string;
  month: string;
  messageCount: number;
}

interface ProcessedMonthlyData {
  month: string;
  messageCount: number;
  formattedMonth: string;
}

export function MessageTrendChart() {
  const [data, setData] = useState<ProcessedMonthlyData[]>([]);
  const [loading, setLoading] = useState(true);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/monthly-messages.json');
        let monthlyData: MonthlyMessageData[] = await response.json();
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          monthlyData = monthlyData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
        }
        
        // Aggregate by month across filtered conversations
        const monthlyTotals = monthlyData.reduce((acc, item) => {
          if (!acc[item.month]) {
            acc[item.month] = 0;
          }
          acc[item.month] += item.messageCount;
          return acc;
        }, {} as Record<string, number>);
        
        // Process and sort the data
        const processedData = Object.entries(monthlyTotals)
          .map(([month, messageCount]) => ({
            month,
            messageCount,
            formattedMonth: format(parseISO(month + '-01'), 'MMM yyyy')
          }))
          .sort((a, b) => a.month.localeCompare(b.month));
        
        setData(processedData);
      } catch (error) {
        console.error('Error loading monthly messages data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return <div className="flex items-center justify-center h-full">Loading...</div>;
  }

  return (
    <div className="h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="formattedMonth" 
            stroke="#6b7280"
            fontSize={12}
            interval="preserveStartEnd"
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickFormatter={(value) => value.toLocaleString()}
          />
          <Tooltip 
            formatter={(value) => [Number(value).toLocaleString(), 'Messages']}
            labelStyle={{ color: '#374151' }}
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e5e7eb',
              borderRadius: '8px'
            }}
          />
          <Line 
            type="monotone" 
            dataKey="messageCount" 
            stroke="#3b82f6" 
            strokeWidth={2}
            dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
} 