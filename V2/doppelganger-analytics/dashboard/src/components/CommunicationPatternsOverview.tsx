'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { MessageSquare, Users, Clock, TrendingUp, Zap, Target, Activity } from 'lucide-react';

interface MonthlyMessageData {
  conversation_id: string;
  month: string;
  messageCount: number;
}

interface ReplyLatencyData {
  conversation_id: string;
  bucket: string;
  count: number;
}

interface PatternData {
  pattern: string;
  frequency: number;
  description: string;
  trend: 'increasing' | 'decreasing' | 'stable';
  impact: 'high' | 'medium' | 'low';
}

interface CommunicationMetrics {
  totalMessages: number;
  avgResponseTime: number;
  activeParticipants: number;
  conversationHealth: 'excellent' | 'good' | 'moderate' | 'needs_attention';
  dominantPattern: string;
  communicationStyle: 'formal' | 'casual' | 'mixed';
  engagementLevel: 'high' | 'medium' | 'low';
}

interface TimePattern {
  period: string;
  activity: number;
  engagement: number;
  responseSpeed: number;
}

const PATTERN_COLORS = {
  'Quick Response': '#10b981',
  'Delayed Response': '#f59e0b',
  'Burst Communication': '#3b82f6',
  'Steady Flow': '#8b5cf6',
  'Question-Answer': '#ef4444',
  'Media Sharing': '#06b6d4',
  'Long Messages': '#f97316',
  'Short Messages': '#84cc16'
};

export function CommunicationPatternsOverview() {
  const [patterns, setPatterns] = useState<PatternData[]>([]);
  const [metrics, setMetrics] = useState<CommunicationMetrics | null>(null);
  const [timePatterns, setTimePatterns] = useState<TimePattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'patterns' | 'metrics' | 'timeline'>('patterns');
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Load multiple data sources to create comprehensive patterns
        const [monthlyResponse, latencyResponse, engagementResponse] = await Promise.all([
          fetch('/data/monthly-messages.json'),
          fetch('/data/replyLatencyDistribution.json'),
          fetch('/data/engagementScoring.json').catch(() => null)
        ]);

        const monthlyData: MonthlyMessageData[] = await monthlyResponse.json();
        const latencyData: ReplyLatencyData[] = await latencyResponse.json();
        const engagementData: { summary?: { total_participants?: number } } | null = engagementResponse
          ? await engagementResponse.json()
          : null;

        // Filter data if needed
        let filteredMonthlyData = monthlyData;
        let filteredLatencyData = latencyData;
        
        if (isFiltered && selectedConversations.length > 0) {
          filteredMonthlyData = monthlyData.filter((item) => 
            selectedConversations.includes(item.conversation_id)
          );
          filteredLatencyData = latencyData.filter((item) => 
            selectedConversations.includes(item.conversation_id)
          );
        }

        // Analyze communication patterns from real data only
        const totalMessages = filteredMonthlyData.reduce((sum, item) => sum + item.messageCount, 0);

        const latencyTotals = filteredLatencyData.reduce((acc, item) => {
          acc[item.bucket] = (acc[item.bucket] || 0) + item.count;
          return acc;
        }, {} as Record<string, number>);

        const quickResponses = (latencyTotals['0-10s'] || 0) + (latencyTotals['10-30s'] || 0);
        const delayedResponses = (latencyTotals['15-60m'] || 0) + (latencyTotals['>1h'] || 0);
        const totalResponses = Object.values(latencyTotals).reduce((sum, count) => sum + count, 0);

        // Monthly volume trend for steady vs burst
        const monthlyTotals = filteredMonthlyData.reduce((acc, item) => {
          acc[item.month] = (acc[item.month] || 0) + item.messageCount;
          return acc;
        }, {} as Record<string, number>);
        const monthValues = Object.values(monthlyTotals);
        const monthMean = monthValues.length > 0
          ? monthValues.reduce((s, v) => s + v, 0) / monthValues.length
          : 0;
        const monthVariance = monthValues.length > 0
          ? monthValues.reduce((s, v) => s + Math.pow(v - monthMean, 2), 0) / monthValues.length
          : 0;
        const monthCv = monthMean > 0 ? Math.sqrt(monthVariance) / monthMean : 0;
        const burstShare = Math.min(100, Math.round(monthCv * 100));
        const steadyShare = Math.max(0, 100 - burstShare);

        const communicationPatterns: PatternData[] = [
          {
            pattern: 'Quick Response',
            frequency: totalResponses > 0 ? Math.round((quickResponses / totalResponses) * 100) : 0,
            description: 'Messages responded to within 30 seconds',
            trend: 'stable' as const,
            impact: 'high' as const
          },
          {
            pattern: 'Delayed Response',
            frequency: totalResponses > 0 ? Math.round((delayedResponses / totalResponses) * 100) : 0,
            description: 'Messages with response time over 15 minutes',
            trend: 'stable' as const,
            impact: 'medium' as const
          }
        ];

        if (monthValues.length >= 2) {
          communicationPatterns.push(
            {
              pattern: 'Burst Communication',
              frequency: burstShare,
              description: 'Month-to-month volume variability (higher = more bursty)',
              trend: burstShare > 40 ? 'increasing' as const : 'stable' as const,
              impact: 'high' as const
            },
            {
              pattern: 'Steady Flow',
              frequency: steadyShare,
              description: 'Consistency of monthly message volume',
              trend: steadyShare > 60 ? 'stable' as const : 'decreasing' as const,
              impact: 'medium' as const
            }
          );
        }

        communicationPatterns.sort((a, b) => b.frequency - a.frequency);

        // Calculate overall metrics
        const avgResponseTime = totalResponses > 0 ? 
          Object.entries(latencyTotals).reduce((sum, [bucket, count]) => {
            let bucketTime = 300000; // Default 5 minutes in ms
            if (bucket === '0-10s') bucketTime = 5000;
            else if (bucket === '10-30s') bucketTime = 20000;
            else if (bucket === '30-60s') bucketTime = 45000;
            else if (bucket === '1-5m') bucketTime = 180000;
            else if (bucket === '5-15m') bucketTime = 600000;
            else if (bucket === '15-60m') bucketTime = 2250000;
            else if (bucket === '>1h') bucketTime = 7200000;
            
            return sum + (bucketTime * count);
          }, 0) / totalResponses : 0;

        const activeParticipants = engagementData?.summary?.total_participants || 
          new Set(filteredMonthlyData.map((item) => item.conversation_id)).size;

        // Determine communication health
        let communicationHealth: 'excellent' | 'good' | 'moderate' | 'needs_attention' = 'needs_attention';
        const quickResponseRate = totalResponses > 0 ? quickResponses / totalResponses : 0;
        
        if (quickResponseRate > 0.7 && totalMessages > 1000) communicationHealth = 'excellent';
        else if (quickResponseRate > 0.5 && totalMessages > 500) communicationHealth = 'good';
        else if (quickResponseRate > 0.3 && totalMessages > 100) communicationHealth = 'moderate';

        const dominantPattern = communicationPatterns[0]?.pattern || 'Unknown';
        
        // Determine communication style based on patterns
        let communicationStyle: 'formal' | 'casual' | 'mixed' = 'mixed';
        if (quickResponseRate > 0.6) communicationStyle = 'casual';
        else if (quickResponseRate < 0.3) communicationStyle = 'formal';

        // Determine engagement level
        let engagementLevel: 'high' | 'medium' | 'low' = 'low';
        if (totalMessages > 1000 && quickResponseRate > 0.5) engagementLevel = 'high';
        else if (totalMessages > 300 && quickResponseRate > 0.3) engagementLevel = 'medium';

        const communicationMetrics: CommunicationMetrics = {
          totalMessages,
          avgResponseTime: Math.round(avgResponseTime / 1000), // Convert to seconds
          activeParticipants,
          conversationHealth: communicationHealth,
          dominantPattern,
          communicationStyle,
          engagementLevel
        };

        // Timeline from real monthly volume (scaled to 0–100 relative activity)
        const sortedMonths = Object.entries(monthlyTotals).sort(([a], [b]) => a.localeCompare(b));
        const maxMonth = Math.max(0, ...sortedMonths.map(([, count]) => count));
        const timeBasedPatterns: TimePattern[] = sortedMonths.map(([month, count]) => {
          const date = new Date(`${month}-01`);
          const activity = maxMonth > 0 ? Math.round((count / maxMonth) * 100) : 0;
          return {
            period: date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' }),
            activity,
            engagement: activity,
            responseSpeed: 0
          };
        });

        setPatterns(communicationPatterns);
        setMetrics(communicationMetrics);
        setTimePatterns(timeBasedPatterns);
        
      } catch (error) {
        console.error('Error loading communication patterns:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <Activity className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Communication Patterns Overview</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Loading communication patterns...</div>
        </div>
      </div>
    );
  }

  if (!metrics || patterns.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center space-x-2 mb-4">
          <Activity className="w-5 h-5 text-red-600" />
          <h3 className="text-lg font-semibold">Communication Patterns Overview</h3>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <MessageSquare className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-500">No Pattern Data Available</p>
            <p className="text-sm text-gray-400">Communication patterns could not be analyzed</p>
          </div>
        </div>
      </div>
    );
  }

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'excellent': return 'text-green-600 bg-green-50 border-green-200';
      case 'good': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'moderate': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-red-600 bg-red-50 border-red-200';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing': return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'decreasing': return <TrendingUp className="w-4 h-4 text-red-600 transform rotate-180" />;
      default: return <Target className="w-4 h-4 text-blue-600" />;
    }
  };

  const formatTooltip = (value: number, data: Partial<PatternData> & Partial<TimePattern>) => {
    return [
      <div key="tooltip" className="text-sm">
        <div className="font-semibold">{data.pattern || data.period}</div>
        <div>Frequency: {value}%</div>
        {data.description && <div className="text-xs text-gray-500 mt-1">{data.description}</div>}
      </div>
    ];
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-2">
          <Activity className="w-5 h-5 text-blue-600" />
          <h3 className="text-lg font-semibold">Communication Patterns Overview</h3>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('patterns')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'patterns' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Patterns
          </button>
          <button
            onClick={() => setViewMode('metrics')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'metrics' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Metrics
          </button>
          <button
            onClick={() => setViewMode('timeline')}
            className={`px-3 py-1 rounded text-sm ${
              viewMode === 'timeline' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
            }`}
          >
            Timeline
          </button>
        </div>
      </div>

      {/* Health Status Card */}
      <div className={`p-4 rounded-lg border mb-6 ${getHealthColor(metrics.conversationHealth)}`}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-semibold">Communication Health: {metrics.conversationHealth.toUpperCase()}</h4>
            <p className="text-sm opacity-80">
              {metrics.totalMessages.toLocaleString()} messages • {metrics.activeParticipants} participants • {metrics.communicationStyle} style
            </p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">{metrics.engagementLevel.toUpperCase()}</div>
            <div className="text-sm opacity-80">Engagement</div>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <MessageSquare className="w-4 h-4 text-blue-600" />
            <span className="text-sm text-blue-600 font-medium">Dominant Pattern</span>
          </div>
          <div className="text-lg font-bold text-blue-900 mt-1">
            {metrics.dominantPattern}
          </div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4 text-green-600" />
            <span className="text-sm text-green-600 font-medium">Avg Response</span>
          </div>
          <div className="text-lg font-bold text-green-900 mt-1">
            {metrics.avgResponseTime < 60 ? `${metrics.avgResponseTime}s` : `${Math.round(metrics.avgResponseTime / 60)}m`}
          </div>
        </div>
        
        <div className="bg-purple-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4 text-purple-600" />
            <span className="text-sm text-purple-600 font-medium">Active Users</span>
          </div>
          <div className="text-lg font-bold text-purple-900 mt-1">
            {metrics.activeParticipants}
          </div>
        </div>
        
        <div className="bg-orange-50 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-orange-600" />
            <span className="text-sm text-orange-600 font-medium">Style</span>
          </div>
          <div className="text-lg font-bold text-orange-900 mt-1 capitalize">
            {metrics.communicationStyle}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="h-80">
        {viewMode === 'patterns' && (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={patterns}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="pattern" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      {formatTooltip(Number(payload[0].value), payload[0].payload)}
                    </div>
                  );
                }
                return null;
              }} />
              <Bar dataKey="frequency" radius={[4, 4, 0, 0]}>
                {patterns.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={PATTERN_COLORS[entry.pattern as keyof typeof PATTERN_COLORS] || '#6b7280'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}

        {viewMode === 'metrics' && (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={patterns.slice(0, 6)}
                cx="50%"
                cy="50%"
                outerRadius={120}
                innerRadius={60}
                paddingAngle={2}
                dataKey="frequency"
              >
                {patterns.slice(0, 6).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={PATTERN_COLORS[entry.pattern as keyof typeof PATTERN_COLORS] || '#6b7280'} />
                ))}
              </Pie>
              <Tooltip content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                      {formatTooltip(Number(payload[0].value), payload[0].payload)}
                    </div>
                  );
                }
                return null;
              }} />
            </PieChart>
          </ResponsiveContainer>
        )}

        {viewMode === 'timeline' && (
          timePatterns.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-500 text-sm">
              Monthly volume timeline unavailable for the current selection
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timePatterns}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="period" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="activity" stroke="#3b82f6" strokeWidth={2} name="Relative Volume" />
              </LineChart>
            </ResponsiveContainer>
          )
        )}
      </div>

      {/* Pattern Details */}
      <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        {patterns.slice(0, 4).map((pattern) => (
          <div key={pattern.pattern} className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h5 className="font-semibold text-gray-900">{pattern.pattern}</h5>
              <div className="flex items-center space-x-2">
                {getTrendIcon(pattern.trend)}
                <span className="text-sm font-medium">{pattern.frequency}%</span>
              </div>
            </div>
            <p className="text-sm text-gray-600">{pattern.description}</p>
            <div className="mt-2 flex items-center space-x-4">
              <span className={`text-xs px-2 py-1 rounded ${
                pattern.impact === 'high' ? 'bg-red-100 text-red-700' :
                pattern.impact === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                'bg-green-100 text-green-700'
              }`}>
                {pattern.impact.toUpperCase()} IMPACT
              </span>
              <span className="text-xs text-gray-500 capitalize">{pattern.trend} trend</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 