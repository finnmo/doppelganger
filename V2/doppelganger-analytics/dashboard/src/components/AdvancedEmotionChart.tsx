'use client';

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { AlertCircle, Brain } from 'lucide-react';

interface EmotionData {
  message_id: number;
  conversation_id?: string;
  emotions: {
    joy: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
  };
}

interface AdvancedEmotionCategory {
  category: string;
  value: number;
  intensity: 'low' | 'medium' | 'high';
  color: string;
}

export function AdvancedEmotionChart() {
  const [data, setData] = useState<AdvancedEmotionCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { selectedConversations, isFiltered } = useConversationFilter();

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
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          emotionData = emotionData.filter(emotion => 
            emotion.conversation_id && selectedConversations.includes(emotion.conversation_id)
          );
        }
        
        if (emotionData.length === 0) {
          setData([]);
          return;
        }
        
        // Calculate advanced emotion categories
        const categoryTotals = {
          positive: 0,
          negative: 0,
          anxiety: 0,
          affection: 0,
          neutral: 0
        };
        
        let totalMessages = 0;
        
        emotionData.forEach(item => {
          if (item.emotions) {
            totalMessages++;
            
            // Map basic emotions to advanced categories
            const joy = item.emotions.joy || 0;
            const sadness = item.emotions.sadness || 0;
            const anger = item.emotions.anger || 0;
            const fear = item.emotions.fear || 0;
            const surprise = item.emotions.surprise || 0;
        
            // Positive emotions (joy + positive surprise)
            categoryTotals.positive += joy + (surprise > 0.3 ? surprise * 0.7 : 0);
            
            // Negative emotions (sadness + anger)
            categoryTotals.negative += sadness + anger;
          
            // Anxiety (fear + negative surprise)
            categoryTotals.anxiety += fear + (surprise > 0.3 ? surprise * 0.3 : surprise);
            
            // Affection (subset of joy with higher threshold)
            categoryTotals.affection += joy > 0.5 ? joy * 0.8 : 0;
            
            // Neutral (when all emotions are low)
            const totalEmotion = joy + sadness + anger + fear + surprise;
            if (totalEmotion < 0.2) {
              categoryTotals.neutral += 1;
            }
          }
        });
        
        if (totalMessages === 0) {
          setData([]);
          return;
        }
        
        // Calculate percentages and intensity levels
        const processedData: AdvancedEmotionCategory[] = Object.entries(categoryTotals)
          .map(([category, value]) => {
            const percentage = (value / totalMessages) * 100;
            let intensity: 'low' | 'medium' | 'high' = 'low';
            
            if (percentage > 30) intensity = 'high';
            else if (percentage > 15) intensity = 'medium';
            
            return {
              category: category.charAt(0).toUpperCase() + category.slice(1),
              value: Math.round(percentage * 10) / 10,
              intensity,
              color: getCategoryColor(category, intensity)
            };
          })
          .filter(item => item.value > 0)
          .sort((a, b) => b.value - a.value);
        
        setData(processedData);
      } catch (error) {
        console.error('Error loading advanced emotion data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load emotion data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  const getCategoryColor = (category: string, intensity: 'low' | 'medium' | 'high'): string => {
    const colors = {
      positive: { low: '#86efac', medium: '#4ade80', high: '#22c55e' },
      negative: { low: '#fca5a5', medium: '#f87171', high: '#ef4444' },
      anxiety: { low: '#c4b5fd', medium: '#a78bfa', high: '#8b5cf6' },
      affection: { low: '#fdba74', medium: '#fb923c', high: '#f97316' },
      neutral: { low: '#d1d5db', medium: '#9ca3af', high: '#6b7280' }
    };
    
    return colors[category as keyof typeof colors]?.[intensity] || '#6b7280';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-pink-600"></div>
          <span className="text-gray-600">Loading advanced emotion analysis...</span>
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
          <Brain className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <h4 className="text-lg font-semibold text-gray-600 mb-2">No Advanced Emotion Data</h4>
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

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="category" 
            stroke="#6b7280"
            fontSize={12}
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis 
            stroke="#6b7280"
            fontSize={12}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip 
            formatter={(value, name, props) => {
              if (name === 'value') {
                const intensity = props.payload?.intensity || 'unknown';
                return [`${Number(value).toFixed(1)}% (${intensity} intensity)`, 'Percentage'];
              }
              return [value, name];
            }}
            labelStyle={{ color: '#374151' }}
            contentStyle={{ 
              backgroundColor: 'white', 
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
          />
          <Bar 
            dataKey="value"
            radius={[4, 4, 0, 0]}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      
      {/* Intensity Legend */}
      <div className="mt-4 flex justify-center">
        <div className="flex items-center gap-6 text-xs text-gray-600">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gray-300"></div>
            <span>Low Intensity</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gray-500"></div>
            <span>Medium Intensity</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gray-700"></div>
            <span>High Intensity</span>
          </div>
        </div>
      </div>
    </div>
  );
} 