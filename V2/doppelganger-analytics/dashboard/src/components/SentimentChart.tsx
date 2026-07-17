'use client';

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import type { ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import { CHART_AREA } from '@/lib/layout';
import { AlertCircle, TrendingUp } from 'lucide-react';

interface SentimentData {
  conversation_id: string;
  sender: string;
  avg_sentiment: number;
  message_count: number;
  avg_positive: number;
  avg_negative: number;
  avg_neutral: number;
}

interface ProcessedSentimentData {
  sender: string;
  sentiment_score: number;
  message_count: number;
  display_name: string;
}

export function SentimentChart() {
  const [data, setData] = useState<ProcessedSentimentData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { filterScopedRows, isFiltered, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch('/data/sentimentBySender.json');
        if (!response.ok) {
          throw new Error(`Failed to load sentiment data: ${response.status}`);
        }
        
        let sentimentData: SentimentData[] = await response.json();
        
        if (!Array.isArray(sentimentData)) {
          throw new Error('Invalid sentiment data format');
        }
        sentimentData = filterScopedRows(sentimentData, { senderKey: 'sender' });
        
        if (sentimentData.length === 0) {
          setData([]);
          return;
        }
        
        // Aggregate by sender across filtered conversations
        const senderAggregates = sentimentData.reduce((acc, item) => {
          if (!acc[item.sender]) {
            acc[item.sender] = {
              total_sentiment: 0,
              total_messages: 0,
              count: 0
            };
          }
          acc[item.sender].total_sentiment += item.avg_sentiment * item.message_count;
          acc[item.sender].total_messages += item.message_count;
          acc[item.sender].count += 1;
          return acc;
        }, {} as Record<string, { total_sentiment: number; total_messages: number; count: number }>);
        
        // Calculate weighted averages and format data
        const processedData = Object.entries(senderAggregates)
          .map(([sender, data]) => ({
            sender,
            sentiment_score: Math.round((data.total_sentiment / data.total_messages) * 1000) / 1000,
            message_count: data.total_messages,
            display_name: sender.length > 15 ? sender.substring(0, 15) + '...' : sender
          }))
          .sort((a, b) => {
            // Sort by sender name first for consistency, then by message count
            if (a.sender < b.sender) return -1;
            if (a.sender > b.sender) return 1;
            return b.message_count - a.message_count;
          })
          .slice(0, 10);
        
        setData(processedData);
      } catch (error) {
        console.error('Error loading sentiment data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load sentiment data');
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
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600"></div>
          <span className="text-gray-600">Loading sentiment analysis...</span>
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
          <TrendingUp className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <h4 className="text-lg font-semibold text-gray-600 mb-2">No Sentiment Data</h4>
          <p className="text-gray-500">
            {isFiltered 
              ? 'No sentiment data available for selected conversations'
              : 'No sentiment data found'
            }
          </p>
        </div>
      </div>
    );
  }

  const formatTooltip = (value: ValueType, name: NameType) => {
    if (name === 'sentiment_score') {
      const score = Number(value);
      const sentiment = score > 0.05 ? 'Positive' : score < -0.05 ? 'Negative' : 'Neutral';
      const intensity = Math.abs(score) > 0.3 ? 'Strong' : Math.abs(score) > 0.1 ? 'Moderate' : 'Mild';
      return [`${score.toFixed(3)} (${intensity} ${sentiment})`, 'Sentiment Score'];
    }
    return [value, name];
  };

  const getBarColor = (value: number) => {
    if (value > 0.05) return '#10b981'; // Green for positive
    if (value < -0.05) return '#ef4444'; // Red for negative
    return '#6b7280'; // Gray for neutral
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      {/* Summary Statistics */}
      {data.length > 0 && (
        <div className="grid shrink-0 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-blue-800 font-medium">{data.length}</div>
            <div className="text-blue-600 text-sm">Participants</div>
          </div>
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="text-green-800 font-medium">
              {data.filter(d => d.sentiment_score > 0.05).length}
            </div>
            <div className="text-green-600 text-sm">Positive</div>
          </div>
          <div className="bg-red-50 p-3 rounded-lg">
            <div className="text-red-800 font-medium">
              {data.filter(d => d.sentiment_score < -0.05).length}
            </div>
            <div className="text-red-600 text-sm">Negative</div>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg">
            <div className="text-gray-800 font-medium">
              {data.filter(d => d.sentiment_score >= -0.05 && d.sentiment_score <= 0.05).length}
            </div>
            <div className="text-gray-600 text-sm">Neutral</div>
          </div>
        </div>
      )}

      <div className={CHART_AREA}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis 
              dataKey="display_name" 
              stroke="#6b7280"
              fontSize={12}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis 
              stroke="#6b7280"
              fontSize={12}
              domain={[-1, 1]}
              tickFormatter={(value) => value.toFixed(1)}
            />
            <ChartTooltip 
              formatter={formatTooltip}
              labelStyle={{ color: '#374151' }}
              contentStyle={{ 
                backgroundColor: 'white', 
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
              }}
            />
            <Bar 
              dataKey="sentiment_score"
              radius={[2, 2, 0, 0]}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(entry.sentiment_score)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
} 