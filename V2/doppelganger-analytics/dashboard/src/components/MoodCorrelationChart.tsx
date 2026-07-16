'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { Heart, BarChart3, Users } from 'lucide-react';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface CorrelationPair {
  sender1: string;
  sender2: string;
  correlation: number;
  sharedDays: number;
  totalDays: number;
  strength: 'strong' | 'moderate' | 'weak' | 'none';
}

interface MoodPattern {
  sender: string;
  averageMood: number;
  moodVariability: number;
  positiveStreak: number;
  negativeStreak: number;
  moodTrend: 'improving' | 'declining' | 'stable';
  dominantEmotion: 'positive' | 'negative' | 'neutral';
}

interface MoodCorrelationData {
  summary: {
    totalParticipants: number;
    totalCorrelations: number;
    strongCorrelations: number;
    averageCorrelation: number;
    dateRange: {
      start: string;
      end: string;
    };
  };
  correlationMatrix: CorrelationPair[];
  moodPatterns: MoodPattern[];
  timeSeriesData: Array<{
    date: string;
    participants: Record<string, number>;
  }>;
}

interface RawSentimentData {
  sender: string;
  conversation_id: string;
  avg_sentiment: number;
  message_count: number;
  avg_positive: number;
  avg_negative: number;
  avg_neutral: number;
}

const calculateVariability = (values: number[]): number => {
  if (values.length < 2) return 0;
  const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
};

const getCorrelationStrength = (correlation: number): 'strong' | 'moderate' | 'weak' | 'none' => {
  const abs = Math.abs(correlation);
  if (abs >= 0.7) return 'strong';
  if (abs >= 0.4) return 'moderate';
  if (abs >= 0.2) return 'weak';
  return 'none';
};

export default function MoodCorrelationChart() {
  const [data, setData] = useState<MoodCorrelationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'correlations' | 'patterns'>('correlations');
  const { selectedConversations, isFiltered } = useConversationFilter();

  const processFilteredMoodData = useCallback((sentimentData: RawSentimentData[]): MoodCorrelationData => {
    // Get unique senders from filtered data
    const uniqueSenders = [...new Set(sentimentData.map(item => item.sender))];
    
    // Create mood patterns for filtered participants
    const moodPatterns: MoodPattern[] = uniqueSenders.map(sender => {
      const senderData = sentimentData.filter(item => item.sender === sender);
      const totalMessages = senderData.reduce((sum, item) => sum + item.message_count, 0);
      const avgMood = senderData.reduce((sum, item) => sum + (item.avg_sentiment * item.message_count), 0) / totalMessages;
      
      // Calculate mood variability from sentiment scores
      const sentimentScores = senderData.flatMap(item => 
        Array(item.message_count).fill(item.avg_sentiment)
      );
      const moodVariability = calculateVariability(sentimentScores);
      
      // Determine mood trend based on average sentiment
      let moodTrend: 'improving' | 'declining' | 'stable' = 'stable';
      if (avgMood > 0.1) moodTrend = 'improving';
      else if (avgMood < -0.1) moodTrend = 'declining';
      
      // Determine dominant emotion
      let dominantEmotion: 'positive' | 'negative' | 'neutral' = 'neutral';
      if (avgMood > 0.1) dominantEmotion = 'positive';
      else if (avgMood < -0.1) dominantEmotion = 'negative';
      
      return {
        sender,
        averageMood: Math.round(avgMood * 1000) / 1000,
        moodVariability: Math.round(moodVariability * 1000) / 1000,
        positiveStreak: avgMood > 0 ? Math.round(avgMood * 10) : 0,
        negativeStreak: avgMood < 0 ? Math.round(Math.abs(avgMood) * 10) : 0,
        moodTrend,
        dominantEmotion
      };
    });

    // Create correlation matrix for filtered participants
    const correlationMatrix: CorrelationPair[] = [];
    for (let i = 0; i < uniqueSenders.length; i++) {
      for (let j = i + 1; j < uniqueSenders.length; j++) {
        const sender1 = uniqueSenders[i];
        const sender2 = uniqueSenders[j];
        
        // Find shared conversations between these two senders
        const sender1Conversations = new Set(sentimentData.filter(item => item.sender === sender1).map(item => item.conversation_id));
        const sender2Conversations = new Set(sentimentData.filter(item => item.sender === sender2).map(item => item.conversation_id));
        const sharedConversations = [...sender1Conversations].filter(conv => sender2Conversations.has(conv));
        
        if (sharedConversations.length > 0) {
          // Calculate correlation based on shared conversations
          const sender1Data = sentimentData.filter(item => item.sender === sender1 && sharedConversations.includes(item.conversation_id));
          const sender2Data = sentimentData.filter(item => item.sender === sender2 && sharedConversations.includes(item.conversation_id));
          
          const sender1Avg = sender1Data.reduce((sum, item) => sum + (item.avg_sentiment * item.message_count), 0) / 
                            sender1Data.reduce((sum, item) => sum + item.message_count, 0);
          const sender2Avg = sender2Data.reduce((sum, item) => sum + (item.avg_sentiment * item.message_count), 0) / 
                            sender2Data.reduce((sum, item) => sum + item.message_count, 0);
          
          // Simple correlation calculation
          const correlation = Math.round((sender1Avg + sender2Avg) / 2 * 1000) / 1000;
          
          correlationMatrix.push({
            sender1,
            sender2,
            correlation,
            sharedDays: sharedConversations.length,
            totalDays: Math.max(sender1Data.length, sender2Data.length),
            strength: getCorrelationStrength(correlation)
          });
        }
      }
    }

    // Calculate summary statistics
    const totalCorrelations = correlationMatrix.length;
    const strongCorrelations = correlationMatrix.filter(corr => corr.strength === 'strong').length;
    const averageCorrelation = totalCorrelations > 0 ? 
      correlationMatrix.reduce((sum, corr) => sum + corr.correlation, 0) / totalCorrelations : 0;

    return {
      summary: {
        totalParticipants: uniqueSenders.length,
        totalCorrelations,
        strongCorrelations,
        averageCorrelation: Math.round(averageCorrelation * 1000) / 1000,
        dateRange: {
          start: 'Filtered Data',
          end: 'Filtered Data'
        }
      },
      correlationMatrix,
      moodPatterns,
      timeSeriesData: [] // No time series data for filtered view
    };
  }, []);

  useEffect(() => {
    const loadData = async () => {
      try {
        if (!isFiltered || selectedConversations.length === 0) {
          const response = await fetch('/data/moodCorrelationMetrics.json');
          const correlationData: MoodCorrelationData = await response.json();
          setData(correlationData);
        } else {
          const sentimentResponse = await fetch('/data/sentimentBySender.json');
          let sentimentData: RawSentimentData[] = await sentimentResponse.json();
          
          sentimentData = sentimentData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );

          if (sentimentData.length === 0) {
            setData(null);
            return;
          }

          const processedData = processFilteredMoodData(sentimentData);
          setData(processedData);
        }
      } catch (error) {
        console.error('Error loading mood correlation data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered, processFilteredMoodData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-red-600"></div>
          <span className="text-gray-600">Loading mood correlation analysis...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-gray-500">
          <Users className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <div>
            <div className="text-lg mb-2">📊 No Correlation Data</div>
            <div>
              {isFiltered 
                ? 'No mood correlation data available for selected conversations'
                : 'No mood correlation data found'
              }
            </div>
          </div>
        </div>
      </div>
    );
  }

  const getCorrelationColor = (strength: string) => {
    switch (strength) {
      case 'strong': return '#ef4444';
      case 'moderate': return '#f59e0b';
      case 'weak': return '#6b7280';
      default: return '#d1d5db';
    }
  };

  const getMoodColor = (mood: number) => {
    if (mood > 0.1) return '#10b981';
    if (mood < -0.1) return '#ef4444';
    return '#6b7280';
  };

  const formatPatternTooltip = (data: MoodPattern) => {
    return (
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{data.sender}</div>
        <div>Average Mood: {data.averageMood.toFixed(3)}</div>
        <div>Variability: {data.moodVariability.toFixed(3)}</div>
        <div>Trend: {data.moodTrend}</div>
        <div>Dominant: {data.dominantEmotion}</div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="bg-blue-50 p-3 rounded">
          <div className="text-blue-800 font-medium">{data.summary.totalParticipants}</div>
          <div className="text-blue-600">Participants</div>
        </div>
        <div className="bg-green-50 p-3 rounded">
          <div className="text-green-800 font-medium">{data.summary.totalCorrelations}</div>
          <div className="text-green-600">Correlations</div>
        </div>
        <div className="bg-purple-50 p-3 rounded">
          <div className="text-purple-800 font-medium">{data.summary.strongCorrelations}</div>
          <div className="text-purple-600">Strong Links</div>
        </div>
        <div className="bg-red-50 p-3 rounded">
          <div className="text-red-800 font-medium">{data.summary.averageCorrelation.toFixed(3)}</div>
          <div className="text-red-600">Avg Correlation</div>
        </div>
      </div>

      <div className="flex bg-gray-100 rounded-lg p-1 w-fit">
          <button
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              viewMode === 'correlations'
              ? 'bg-white text-red-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          onClick={() => setViewMode('correlations')}
          >
          <Heart className="w-4 h-4 inline mr-1" />
          Correlations
          </button>
          <button
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              viewMode === 'patterns'
              ? 'bg-white text-red-600 shadow-sm' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          onClick={() => setViewMode('patterns')}
        >
          <BarChart3 className="w-4 h-4 inline mr-1" />
          Mood Patterns
          </button>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'correlations' ? (
            <ScatterChart data={data.correlationMatrix} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                type="number"
                dataKey="correlation"
                domain={[-1, 1]}
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <YAxis 
                type="number"
                dataKey="sharedDays"
                stroke="#6b7280"
                fontSize={12}
              />
              <Tooltip content={<CustomTooltip />} />
              <Scatter dataKey="correlation" fill="#8884d8">
                {data.correlationMatrix.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getCorrelationColor(entry.strength)} />
                ))}
              </Scatter>
            </ScatterChart>
          ) : (
            <BarChart data={data.moodPatterns} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="sender" 
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
              <Tooltip formatter={(value, name, props) => formatPatternTooltip(props.payload)} />
              <Bar dataKey="averageMood" radius={[2, 2, 0, 0]}>
                {data.moodPatterns.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getMoodColor(entry.averageMood)} />
                ))}
              </Bar>
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      <div className="text-xs text-gray-600">
        {viewMode === 'correlations' ? (
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Strong (≥0.7)</span>
                </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span>Moderate (≥0.4)</span>
                  </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-500"></div>
              <span>Weak (≥0.2)</span>
                </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-300"></div>
              <span>None (&lt;0.2)</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span>Positive Mood</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span>Negative Mood</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-500"></div>
              <span>Neutral Mood</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: CorrelationPair }>;
}

const CustomTooltip = ({ active, payload }: CustomTooltipProps) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload;
    return (
      <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
        <div className="font-semibold mb-2">{data.sender1} ↔ {data.sender2}</div>
        <div className="space-y-1 text-sm">
          <p><strong>Correlation:</strong> {data.correlation.toFixed(3)}</p>
          <p><strong>Strength:</strong> {data.strength}</p>
          <p><strong>Shared Days:</strong> {data.sharedDays}</p>
        </div>
      </div>
    );
  }
  return null;
}; 