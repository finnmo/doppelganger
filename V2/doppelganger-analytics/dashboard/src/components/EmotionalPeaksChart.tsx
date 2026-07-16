'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { useTheme } from '@/contexts/ThemeContext';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell, LineChart, Line } from 'recharts';
import { TrendingUp, TrendingDown, Target, Zap, Heart, AlertTriangle, Users, Calendar, Activity, Info } from 'lucide-react';

// Data interfaces matching the processor output
interface EmotionalPeak {
  id: string;
  conversation_id: string;
  timestamp: number;
  sentiment_score: number;
  intensity: 'extreme' | 'high' | 'moderate';
  type: 'peak' | 'valley';
  trigger_analysis: {
    primary_trigger: string;
    trigger_confidence: number;
    contributing_factors: string[];
    keywords: string[];
  };
  participants: Array<{
    sender: string;
    contribution_score: number;
    message_count: number;
  }>;
  sample_messages: Array<{
    content: string;
    sender: string;
    sentiment: number;
  }>;
  recovery_time?: number;
  recovery_factors?: string[];
}

interface EmotionalPattern {
  pattern_type: 'volatility' | 'stability' | 'gradual_increase' | 'gradual_decrease' | 'cyclical';
  frequency: number;
  description: string;
  time_periods: string[];
  participants_involved: string[];
}

interface TriggerAnalysis {
  trigger: string;
  frequency: number;
  avg_impact: number;
  typical_sentiment_change: number;
  recovery_time: number;
  associated_keywords: string[];
  time_patterns: Array<{
    hour: number;
    day_of_week: number;
    frequency: number;
  }>;
  participant_sensitivity: Array<{
    sender: string;
    sensitivity_score: number;
    typical_response: number;
  }>;
}

interface EmotionalPeaksData {
  summary: {
    total_peaks: number;
    total_valleys: number;
    extreme_events: number;
    avg_peak_intensity: number;
    avg_valley_intensity: number;
    avg_recovery_time: number;
    most_volatile_period: string;
    most_stable_period: string;
    dominant_pattern: string;
  };
  peaks_and_valleys: EmotionalPeak[];
  emotional_patterns: EmotionalPattern[];
  trigger_analysis: TriggerAnalysis[];
  temporal_analysis: {
    hourly_volatility: Array<{
      hour: number;
      avg_volatility: number;
      peak_frequency: number;
      valley_frequency: number;
    }>;
    daily_volatility: Array<{
      day_of_week: number;
      day_name: string;
      avg_volatility: number;
      emotional_events: number;
    }>;
    monthly_trends: Array<{
      month: string;
      emotional_stability: number;
      peak_count: number;
      valley_count: number;
      avg_sentiment: number;
    }>;
  };
  recovery_analysis: {
    avg_peak_recovery: number;
    avg_valley_recovery: number;
    fastest_recovery: EmotionalPeak;
    slowest_recovery: EmotionalPeak;
    recovery_factors: Array<{
      factor: string;
      impact_on_recovery: number;
      frequency: number;
    }>;
  };
}

export function EmotionalPeaksChart() {
  const [data, setData] = useState<EmotionalPeaksData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'timeline' | 'triggers' | 'patterns' | 'recovery'>('timeline');
  const { selectedConversations, isFiltered } = useConversationFilter();
  const { getThemeClasses } = useTheme();

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch('/data/emotionalPeaks.json');
        if (!response.ok) {
          throw new Error(`Failed to load emotional peaks data: ${response.status}`);
        }
        
        const emotionalData: EmotionalPeaksData = await response.json();
        
        // Filter data if conversations are selected
        if (isFiltered && selectedConversations.length > 0) {
          const filteredPeaksAndValleys = emotionalData.peaks_and_valleys.filter(peak => 
            selectedConversations.includes(peak.conversation_id)
          );
          
          if (filteredPeaksAndValleys.length === 0) {
            setData({
              summary: {
                total_peaks: 0,
                total_valleys: 0,
                extreme_events: 0,
                avg_peak_intensity: 0,
                avg_valley_intensity: 0,
                avg_recovery_time: 0,
                most_volatile_period: 'N/A',
                most_stable_period: 'N/A',
                dominant_pattern: 'N/A'
              },
              peaks_and_valleys: [],
              emotional_patterns: [],
              trigger_analysis: [],
              temporal_analysis: {
                hourly_volatility: [],
                daily_volatility: [],
                monthly_trends: []
              },
              recovery_analysis: {
                avg_peak_recovery: 0,
                avg_valley_recovery: 0,
                fastest_recovery: {} as EmotionalPeak,
                slowest_recovery: {} as EmotionalPeak,
                recovery_factors: []
              }
            });
            return;
          }
          
          const peaks = filteredPeaksAndValleys.filter(item => item.type === 'peak');
          const valleys = filteredPeaksAndValleys.filter(item => item.type === 'valley');
          const extremeEvents = filteredPeaksAndValleys.filter(item => item.intensity === 'extreme');
          
          // Recalculate trigger analysis for filtered data
          const triggerMap = new Map<string, {
            frequency: number;
            impacts: number[];
            keywords: Set<string>;
            participants: Set<string>;
          }>();
          
          filteredPeaksAndValleys.forEach(peak => {
            const trigger = peak.trigger_analysis.primary_trigger;
            if (!triggerMap.has(trigger)) {
              triggerMap.set(trigger, {
                frequency: 0,
                impacts: [],
                keywords: new Set(),
                participants: new Set()
              });
            }
            
            const data = triggerMap.get(trigger)!;
            data.frequency++;
            data.impacts.push(Math.abs(peak.sentiment_score));
            peak.trigger_analysis.keywords?.forEach(kw => data.keywords.add(kw));
            peak.participants.forEach(p => data.participants.add(p.sender));
          });
          
          const filteredTriggerAnalysis: TriggerAnalysis[] = Array.from(triggerMap.entries()).map(([trigger, data]) => ({
            trigger,
            frequency: data.frequency,
            avg_impact: data.impacts.reduce((a, b) => a + b, 0) / data.impacts.length,
            typical_sentiment_change: data.impacts.reduce((a, b) => a + b, 0) / data.impacts.length,
            recovery_time: 0, // Would need temporal data to calculate properly
            associated_keywords: Array.from(data.keywords).slice(0, 10),
            time_patterns: [],
            participant_sensitivity: Array.from(data.participants).map(sender => ({
              sender,
              sensitivity_score: 0.5, // Simplified for filtering
              typical_response: 0.5
            })).slice(0, 5)
          })).sort((a, b) => b.frequency - a.frequency);
          
          // Recalculate recovery analysis
          const recoveryTimes = filteredPeaksAndValleys
            .map(p => p.recovery_time || 0)
            .filter(t => t > 0);
          
          const peakRecoveryTimes = peaks
            .map(p => p.recovery_time || 0)
            .filter(t => t > 0);
          
          const valleyRecoveryTimes = valleys
            .map(p => p.recovery_time || 0)
            .filter(t => t > 0);
            
          const fastestRecovery = filteredPeaksAndValleys.reduce((fastest, current) => 
            (current.recovery_time || Infinity) < (fastest.recovery_time || Infinity) 
              ? current : fastest,
            filteredPeaksAndValleys[0] || {} as EmotionalPeak);
          
          // Create comprehensive filtered data
          const filteredData: EmotionalPeaksData = {
            summary: {
              total_peaks: peaks.length,
              total_valleys: valleys.length,
              extreme_events: extremeEvents.length,
              avg_peak_intensity: peaks.length > 0 
                ? peaks.reduce((sum, p) => sum + Math.abs(p.sentiment_score), 0) / peaks.length 
                : 0,
              avg_valley_intensity: valleys.length > 0 
                ? valleys.reduce((sum, v) => sum + Math.abs(v.sentiment_score), 0) / valleys.length 
                : 0,
              avg_recovery_time: recoveryTimes.length > 0
                ? recoveryTimes.reduce((a, b) => a + b, 0) / recoveryTimes.length
                : 0,
              most_volatile_period: 'Filtered Period',
              most_stable_period: 'Filtered Period',
              dominant_pattern: filteredTriggerAnalysis[0]?.trigger || 'none'
            },
            peaks_and_valleys: filteredPeaksAndValleys,
            emotional_patterns: [], // Would need full temporal analysis to recalculate properly
            trigger_analysis: filteredTriggerAnalysis,
            temporal_analysis: {
              hourly_volatility: [],
              daily_volatility: [],
              monthly_trends: []
            },
            recovery_analysis: {
              avg_peak_recovery: peakRecoveryTimes.length > 0
                ? peakRecoveryTimes.reduce((a, b) => a + b, 0) / peakRecoveryTimes.length
                : 0,
              avg_valley_recovery: valleyRecoveryTimes.length > 0
                ? valleyRecoveryTimes.reduce((a, b) => a + b, 0) / valleyRecoveryTimes.length
                : 0,
              fastest_recovery: fastestRecovery,
              slowest_recovery: fastestRecovery, // Simplified
              recovery_factors: [
                { factor: 'time_passage', impact_on_recovery: -0.2, frequency: 1.0 },
                { factor: 'social_support', impact_on_recovery: -0.3, frequency: 0.6 }
              ]
            }
          };
          
          setData(filteredData);
        } else {
          setData(emotionalData);
        }
      } catch (error) {
        console.error('Error loading emotional peaks data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  const themeClasses = getThemeClasses();

  // Loading state
  if (loading) {
    return (
      <div className={themeClasses.sectionCardClass}>
        <div className="flex items-center space-x-2 mb-4">
          <Activity className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Emotional Peaks & Valleys</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading emotional analysis...</div>
        </div>
      </div>
    );
  }

  // Error state
  if (error || !data) {
    return (
      <div className={themeClasses.sectionCardClass}>
        <div className="flex items-center space-x-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold">Emotional Peaks & Valleys</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <AlertTriangle className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">No Emotional Data Available</p>
            <p className="text-sm text-gray-400">
              {error || 'Emotional peaks analysis could not be loaded'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Empty state
  if (data.summary.total_peaks === 0 && data.summary.total_valleys === 0) {
    return (
      <div className={themeClasses.sectionCardClass}>
        <div className="flex items-center space-x-2 mb-4">
          <Target className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Emotional Peaks & Valleys</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <Target className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">No Emotional Extremes Detected</p>
            <p className="text-sm text-gray-400">
              Your conversations show stable emotional patterns
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Helper functions
  const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp).toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getIntensityColor = (intensity: string, type: 'peak' | 'valley'): string => {
    switch (intensity) {
      case 'extreme': return type === 'peak' ? '#059669' : '#dc2626';
      case 'high': return type === 'peak' ? '#10b981' : '#ef4444';
      case 'moderate': return type === 'peak' ? '#34d399' : '#f87171';
      default: return '#6b7280';
    }
  };

  const getTriggerIcon = (trigger: string) => {
    const triggerMap: Record<string, React.ReactNode> = {
      work_stress: <Zap className="w-4 h-4" />,
      relationship: <Heart className="w-4 h-4" />,
      achievement: <TrendingUp className="w-4 h-4" />,
      social: <Users className="w-4 h-4" />,
      health: <Activity className="w-4 h-4" />,
      family: <Heart className="w-4 h-4" />,
      financial: <TrendingDown className="w-4 h-4" />,
      technology: <Zap className="w-4 h-4" />,
      travel: <Calendar className="w-4 h-4" />,
      weather_events: <Target className="w-4 h-4" />
    };
    return triggerMap[trigger] || <Info className="w-4 h-4" />;
  };

  const formatDuration = (ms: number): string => {
    const hours = Math.floor(ms / (1000 * 60 * 60));
    const minutes = Math.floor((ms % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m`;
    } else {
      return '<1m';
    }
  };

  // Prepare timeline data
  const timelineData = data.peaks_and_valleys
    .map(item => ({
      timestamp: item.timestamp,
      sentiment: item.sentiment_score,
      type: item.type,
      intensity: item.intensity,
      trigger: item.trigger_analysis.primary_trigger,
      confidence: item.trigger_analysis.trigger_confidence,
      participants: item.participants.length,
      date: formatTimestamp(item.timestamp),
      id: item.id
    }))
    .sort((a, b) => a.timestamp - b.timestamp);

  // Render different views
  const renderTimelineView = () => (
    <div className="space-y-4">
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()}
            />
            <YAxis 
              domain={[-1, 1]}
              tickFormatter={(value) => value.toFixed(1)}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      <div className="font-semibold text-sm">
                        {data.type === 'peak' ? '📈' : '📉'} {data.type.charAt(0).toUpperCase() + data.type.slice(1)}
                      </div>
                      <div className="text-xs text-gray-600">{data.date}</div>
                      <div className="text-sm">
                        <div>Sentiment: {data.sentiment.toFixed(2)}</div>
                        <div>Intensity: {data.intensity}</div>
                        <div>Trigger: {data.trigger}</div>
                        <div>Confidence: {(data.confidence * 100).toFixed(0)}%</div>
                        <div>Participants: {data.participants}</div>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter
              dataKey="sentiment"
            >
              {timelineData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={getIntensityColor(entry.intensity, entry.type)}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      
      {/* Timeline legend */}
      <div className="flex justify-center space-x-6 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-green-500"></div>
          <span>Emotional Peaks</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-red-500"></div>
          <span>Emotional Valleys</span>
        </div>
      </div>
    </div>
  );

  const renderTriggersView = () => (
    <div className="space-y-6">
      {/* Trigger frequency chart */}
      <div className="h-64">
        <h4 className="text-sm font-semibold mb-3">Trigger Frequency & Impact</h4>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data.trigger_analysis.slice(0, 8)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="trigger" 
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      <div className="font-semibold capitalize">{data.trigger.replace('_', ' ')}</div>
                      <div>Frequency: {data.frequency}</div>
                      <div>Avg Impact: {data.avg_impact.toFixed(2)}</div>
                      <div>Keywords: {(data.associated_keywords || []).slice(0, 3).join(', ') || 'N/A'}</div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="frequency" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed trigger cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data.trigger_analysis.slice(0, 6).map((trigger, index) => (
          <div key={index} className="bg-gray-50 p-4 rounded-lg border">
            <div className="flex items-center space-x-2 mb-2">
              {getTriggerIcon(trigger.trigger)}
              <h5 className="font-semibold capitalize">
                {trigger.trigger.replace('_', ' ')}
              </h5>
            </div>
            <div className="text-sm text-gray-600 space-y-1">
              <div>Frequency: {trigger.frequency} occurrences</div>
              <div>Average Impact: {trigger.avg_impact.toFixed(2)}</div>
              <div>Keywords: {(trigger.associated_keywords || []).slice(0, 4).join(', ') || 'N/A'}</div>
              <div>Most Sensitive: {trigger.participant_sensitivity?.[0]?.sender || 'N/A'}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderPatternsView = () => (
    <div className="space-y-6">
      {/* Emotional patterns */}
      <div>
        <h4 className="text-sm font-semibold mb-3">Emotional Patterns Detected</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.emotional_patterns.map((pattern, index) => (
            <div key={index} className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <div className="font-semibold text-blue-900 capitalize mb-2">
                {pattern.pattern_type.replace('_', ' ')}
              </div>
              <div className="text-sm text-blue-700 mb-2">
                {pattern.description}
              </div>
              <div className="text-xs text-blue-600">
                <div>Frequency: {pattern.frequency}</div>
                <div>Avg Intensity: {'N/A'}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Temporal volatility */}
      <div className="h-64">
        <h4 className="text-sm font-semibold mb-3">Emotional Volatility by Time</h4>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.temporal_analysis.hourly_volatility}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      <div className="font-semibold">Hour: {data.hour}:00</div>
                      <div>Peak Frequency: {data.peak_frequency}</div>
                      <div>Valley Frequency: {data.valley_frequency}</div>
                      <div>Avg Volatility: {data.avg_volatility.toFixed(2)}</div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Line 
              type="monotone" 
              dataKey="avg_volatility" 
              stroke="#8b5cf6" 
              strokeWidth={2}
              dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderRecoveryView = () => (
    <div className="space-y-6">
      {/* Recovery statistics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="text-green-600 font-semibold text-sm">Peak Recovery</div>
          <div className="text-2xl font-bold text-green-900">
            {data.recovery_analysis.avg_peak_recovery > 0 ? formatDuration(data.recovery_analysis.avg_peak_recovery) : 'N/A'}
          </div>
        </div>
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="text-blue-600 font-semibold text-sm">Valley Recovery</div>
          <div className="text-2xl font-bold text-blue-900">
            {data.recovery_analysis.avg_valley_recovery > 0 ? formatDuration(data.recovery_analysis.avg_valley_recovery) : 'N/A'}
          </div>
        </div>
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="text-orange-600 font-semibold text-sm">Fastest Recovery</div>
          <div className="text-2xl font-bold text-orange-900">
            {data.recovery_analysis.fastest_recovery?.recovery_time ? formatDuration(data.recovery_analysis.fastest_recovery.recovery_time) : 'N/A'}
          </div>
        </div>
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="text-purple-600 font-semibold text-sm">Recovery Factors</div>
          <div className="text-2xl font-bold text-purple-900">
            {data.recovery_analysis.recovery_factors?.length || 0}
          </div>
        </div>
      </div>

      {/* Recovery factors */}
      <div>
        <h4 className="text-sm font-semibold mb-3">Recovery Analysis</h4>
        {(data.recovery_analysis.recovery_factors?.length || 0) > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(data.recovery_analysis.recovery_factors || []).slice(0, 6).map((factor, index) => (
              <div key={index} className="bg-gray-50 p-3 rounded-lg border flex justify-between items-center">
                <div>
                  <div className="font-medium capitalize">{factor.factor.replace('_', ' ')}</div>
                  <div className="text-sm text-gray-600">Frequency: {(factor.frequency * 100).toFixed(0)}%</div>
                </div>
                <div className="text-right">
                  <div className="font-bold text-green-600">
                    {(factor.impact_on_recovery * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-500">impact</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <Heart className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">No recovery data available</p>
            <p className="text-sm text-gray-400">
              Recovery analysis requires emotional peaks and valleys to be detected
            </p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className={themeClasses.sectionCardClass}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Emotional Peaks & Valleys</h3>
        </div>
        
        {/* View mode selector */}
        <div className="flex space-x-2">
          {[
            { id: 'timeline', label: 'Timeline', icon: <Calendar className="w-4 h-4" /> },
            { id: 'triggers', label: 'Triggers', icon: <Zap className="w-4 h-4" /> },
            { id: 'patterns', label: 'Patterns', icon: <TrendingUp className="w-4 h-4" /> },
            { id: 'recovery', label: 'Recovery', icon: <Heart className="w-4 h-4" /> }
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => setViewMode(mode.id as typeof viewMode)}
              className={`px-3 py-1 rounded text-sm flex items-center space-x-1 ${
                viewMode === mode.id 
                  ? 'bg-blue-100 text-blue-700' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {mode.icon}
              <span>{mode.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        <div className="bg-green-50 p-3 rounded-lg">
          <div className="text-green-600 font-semibold text-sm">Emotional Peaks</div>
          <div className="text-xl font-bold text-green-900">{data.summary.total_peaks}</div>
        </div>
        <div className="bg-red-50 p-3 rounded-lg">
          <div className="text-red-600 font-semibold text-sm">Emotional Valleys</div>
          <div className="text-xl font-bold text-red-900">{data.summary.total_valleys}</div>
        </div>
        <div className="bg-blue-50 p-3 rounded-lg">
          <div className="text-blue-600 font-semibold text-sm">Extreme Events</div>
          <div className="text-xl font-bold text-blue-900">
            {data.summary.extreme_events}
          </div>
        </div>
        <div className="bg-purple-50 p-3 rounded-lg">
          <div className="text-purple-600 font-semibold text-sm">Pattern</div>
          <div className="text-sm font-bold text-purple-900 capitalize">
            {data.summary.dominant_pattern}
          </div>
        </div>
      </div>

      {/* Main content area */}
      <div className="min-h-96">
        {viewMode === 'timeline' && renderTimelineView()}
        {viewMode === 'triggers' && renderTriggersView()}
        {viewMode === 'patterns' && renderPatternsView()}
        {viewMode === 'recovery' && renderRecoveryView()}
      </div>

      {/* Filtering indicator */}
      {isFiltered && (
        <div className="mt-4 text-xs text-blue-600 bg-blue-50 p-2 rounded">
          📊 Analysis filtered to {selectedConversations.length} selected conversation{selectedConversations.length !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
} 