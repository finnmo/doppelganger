'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Cell } from 'recharts';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import { CHART_AREA } from '@/lib/layout';
import { AlertCircle, Brain } from 'lucide-react';

interface EmotionData {
  message_id: number;
  conversation_id?: string;
  sender?: string;
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

function computeCategories(items: EmotionData[]): AdvancedEmotionCategory[] {
  const categoryTotals = {
    positive: 0,
    negative: 0,
    anxiety: 0,
    affection: 0,
    neutral: 0
  };
  let totalMessages = 0;

  items.forEach(item => {
    if (!item.emotions) return;
    totalMessages++;
    const joy = item.emotions.joy || 0;
    const sadness = item.emotions.sadness || 0;
    const anger = item.emotions.anger || 0;
    const fear = item.emotions.fear || 0;
    const surprise = item.emotions.surprise || 0;

    categoryTotals.positive += joy + (surprise > 0.3 ? surprise * 0.7 : 0);
    categoryTotals.negative += sadness + anger;
    categoryTotals.anxiety += fear + (surprise > 0.3 ? surprise * 0.3 : surprise);
    categoryTotals.affection += joy > 0.5 ? joy * 0.8 : 0;
    const totalEmotion = joy + sadness + anger + fear + surprise;
    if (totalEmotion < 0.2) categoryTotals.neutral += 1;
  });

  if (totalMessages === 0) return [];

  return Object.entries(categoryTotals)
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
}

function getCategoryColor(category: string, intensity: 'low' | 'medium' | 'high'): string {
  const colors = {
    positive: { low: '#86efac', medium: '#4ade80', high: '#22c55e' },
    negative: { low: '#fca5a5', medium: '#f87171', high: '#ef4444' },
    anxiety: { low: '#c4b5fd', medium: '#a78bfa', high: '#8b5cf6' },
    affection: { low: '#fdba74', medium: '#fb923c', high: '#f97316' },
    neutral: { low: '#d1d5db', medium: '#9ca3af', high: '#6b7280' }
  };
  return colors[category as keyof typeof colors]?.[intensity] || '#6b7280';
}

export function AdvancedEmotionChart() {
  const [rawData, setRawData] = useState<EmotionData[]>([]);
  const [selectedSender, setSelectedSender] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { filterScopedRows, isFiltered, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch('/data/emotionMetrics.json');
        if (!response.ok) throw new Error(`Failed to load emotion data: ${response.status}`);

        let emotionData: EmotionData[] = await response.json();
        if (!Array.isArray(emotionData)) throw new Error('Invalid emotion data format');

        emotionData = filterScopedRows(
          emotionData.filter(
            (e): e is EmotionData & { conversation_id: string; sender: string } =>
              !!e.conversation_id && !!e.sender
          ),
          { senderKey: 'sender' }
        );

        setRawData(emotionData);
        const senders = [...new Set(emotionData.map(e => e.sender).filter(Boolean))] as string[];
        if (senders.length > 0) setSelectedSender(senders[0]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load emotion data');
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [filterScopedRows, scopeConversationIds]);

  const senders = useMemo(
    () => [...new Set(rawData.map(e => e.sender).filter(Boolean))].sort() as string[],
    [rawData]
  );

  const data = useMemo(() => {
    if (!selectedSender) return [];
    return computeCategories(rawData.filter(e => e.sender === selectedSender));
  }, [rawData, selectedSender]);

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
          <p className="text-red-600 mb-4">{error}</p>
        </div>
      </div>
    );
  }

  if (senders.length === 0 || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <Brain className="w-12 h-12 mx-auto mb-4 text-gray-400" />
        <p>No emotion data {isFiltered ? 'for selected conversations' : 'found'}</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex shrink-0 items-center gap-3">
        <label htmlFor="emotion-sender" className="text-sm text-gray-600">Sender:</label>
        <select
          id="emotion-sender"
          value={selectedSender}
          onChange={(e) => setSelectedSender(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-pink-500"
        >
          {senders.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <div className={CHART_AREA}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="category" stroke="#6b7280" fontSize={12} angle={-45} textAnchor="end" height={80} />
            <YAxis stroke="#6b7280" fontSize={12} tickFormatter={(v) => `${v}%`} />
            <ChartTooltip
              formatter={(value, name, props) => {
                if (name === 'value') {
                  const intensity = props.payload?.intensity || 'unknown';
                  return [`${Number(value).toFixed(1)}% (${intensity} intensity)`, 'Percentage'];
                }
                return [value, name];
              }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
