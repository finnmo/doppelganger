'use client';

import React, { useState, useEffect } from 'react';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import { AlertCircle, Smile } from 'lucide-react';

interface EmotionData {
  message_id: number;
  conversation_id?: string; // Optional for backward compatibility
  emotions: {
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
  };
}

interface EmotionTotals {
  emotion: string;
  total: number;
  percentage: number;
}

const EMOTION_COLORS = {
  joy: '#fbbf24',      // Yellow
  sadness: '#60a5fa',  // Blue
  anger: '#f87171',    // Red
  fear: '#a78bfa',     // Purple
  surprise: '#34d399'  // Green
};

export function EmotionChart() {
  const [data, setData] = useState<EmotionTotals[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { filterScopedRows, isFiltered, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch('/data/emotionMetrics.json');
        if (!response.ok) {
          throw new Error(`Failed to load emotion data: ${response.status}`);
        }
        
        let emotionData: EmotionData[] = await response.json();
        
        if (!Array.isArray(emotionData)) {
          throw new Error('Invalid emotion data format');
        }

        emotionData = filterScopedRows(
          emotionData.filter(
            (e): e is EmotionData & { conversation_id: string } => !!e.conversation_id
          )
        );
        
        if (emotionData.length === 0) {
          setData([]);
          return;
        }
        
        // Calculate emotion totals
        const emotionTotals = {
          joy: 0,
          sadness: 0,
          anger: 0,
          fear: 0,
          surprise: 0
        };
        
        emotionData.forEach(item => {
          if (item.emotions) {
            emotionTotals.joy += item.emotions.joy || 0;
            emotionTotals.sadness += item.emotions.sadness || 0;
            emotionTotals.anger += item.emotions.anger || 0;
            emotionTotals.fear += item.emotions.fear || 0;
            emotionTotals.surprise += item.emotions.surprise || 0;
          }
        });
        
        const total = Object.values(emotionTotals).reduce((sum, value) => sum + value, 0);
        
        if (total === 0) {
          setData([]);
          return;
        }
        
        const processedData = Object.entries(emotionTotals)
          .map(([emotion, value]) => ({
            emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
            total: Math.round(value * 100) / 100,
            percentage: Math.round((value / total) * 10000) / 100
          }))
          .filter(item => item.total > 0)
          .sort((a, b) => b.total - a.total);
        
        setData(processedData);
      } catch (error) {
        console.error('Error loading emotion data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load emotion data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-yellow-600"></div>
          <span className="text-gray-600">Loading emotion analysis...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h4 className="text-lg font-semibold text-red-700 mb-2">Error Loading Data</h4>
          <p className="text-red-600 mb-4">{error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-gray-500">
          <Smile className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <h4 className="text-lg font-semibold text-gray-600 mb-2">No Emotion Data</h4>
          <p className="text-gray-500">
            {isFiltered 
              ? 'No emotion data available for selected conversations'
              : 'No emotion data found'
            }
          </p>
        </div>
      </div>
    );
  }

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
        fontSize={11}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
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
            dataKey="total"
            nameKey="emotion"
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={EMOTION_COLORS[entry.emotion.toLowerCase() as keyof typeof EMOTION_COLORS] || '#6b7280'} 
              />
            ))}
          </Pie>
            <ChartTooltip 
              formatter={(value, _name, item) => {
                const total = data.reduce((sum, row) => sum + row.total, 0);
                const pct = total > 0 ? (Number(value) / total) * 100 : 0;
                const emotionName = String(item?.payload?.emotion ?? 'Emotion');
                return [`${Number(value).toFixed(2)} (${pct.toFixed(1)}%)`, emotionName];
              }}
              labelStyle={{ color: '#374151' }}
              contentStyle={{ 
                backgroundColor: 'white', 
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
            />
        </PieChart>
      </ResponsiveContainer>
      </div>

      <div className="mt-1 grid shrink-0 grid-cols-2 justify-items-center gap-x-3 gap-y-0.5 text-xs">
        {data.map((entry, index) => (
          <div key={index} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ 
                backgroundColor: EMOTION_COLORS[entry.emotion.toLowerCase() as keyof typeof EMOTION_COLORS] || '#6b7280' 
              }}
            />
            <span className="text-gray-700">
              {entry.emotion} ({entry.percentage.toFixed(1)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
} 