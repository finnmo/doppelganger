'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface MediaEngagementData {
  summary: {
    totalMediaMessages: number;
    avgEngagementRate: number;
    mostEngagingType: string;
    totalSenders: number;
    analysisWindow: string;
  };
  mediaCorrelations: Array<{
    mediaType: 'photo' | 'video' | 'attachment';
    totalCount: number;
    avgResponseCount: number;
    avgResponseTime: number;
    engagementRate: number;
    responseDistribution: {
      immediate: number;
      quick: number;
      delayed: number;
      late: number;
    };
    topResponders: Array<{
      sender: string;
      responseCount: number;
      avgResponseTime: number;
    }>;
  }>;
  senderEngagement: Array<{
    sender: string;
    mediaShared: {
      photos: number;
      videos: number;
      attachments: number;
      total: number;
    };
    engagementReceived: {
      totalResponses: number;
      avgResponseTime: number;
      engagementScore: number;
    };
    engagementGiven: {
      responsesToMedia: number;
      avgResponseTime: number;
      preferredMediaType: string;
    };
    engagementRatio: number;
  }>;
  timeBasedEngagement: Array<{
    hour: number;
    mediaCount: number;
    avgEngagement: number;
    responseRate: number;
  }>;
}

const MEDIA_TYPE_COLORS = {
  photo: '#3b82f6',
  video: '#ef4444', 
  attachment: '#10b981'
};

const formatTime = (ms: number) => {
  if (ms < 60000) return `${Math.round(ms / 1000)}s`;
  if (ms < 3600000) return `${Math.round(ms / 60000)}m`;
  return `${Math.round(ms / 3600000)}h`;
};

export default function MediaEngagementChart() {
  const [data, setData] = useState<MediaEngagementData | null>(null);
  const [loading, setLoading] = useState(true);
  const [filteredUnavailable, setFilteredUnavailable] = useState(false);
  const [activeView, setActiveView] = useState<'correlations' | 'senders' | 'timing'>('correlations');
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/mediaEngagementMetrics.json');
        const mediaData: MediaEngagementData = await response.json();

        // Media engagement has no conversation_id — cannot accurately filter
        if (isFiltered && selectedConversations.length > 0) {
          setFilteredUnavailable(true);
          setData(null);
          return;
        }

        setFilteredUnavailable(false);
        setData(mediaData);
      } catch (error) {
        console.error('Error loading media engagement data:', error);
        setData(null);
        setFilteredUnavailable(false);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  // Process correlation data for visualization
  const correlationData = useMemo(() => {
    if (!data) return [];
    return data.mediaCorrelations.map(item => ({
      type: item.mediaType,
      count: item.totalCount,
      engagement: item.engagementRate,
      avgResponses: item.avgResponseCount,
      avgTime: item.avgResponseTime,
      immediate: item.responseDistribution.immediate,
      quick: item.responseDistribution.quick,
      delayed: item.responseDistribution.delayed,
      late: item.responseDistribution.late
    }));
  }, [data]);

  // Process response timing distribution
  const timingData = useMemo(() => {
    if (!data) return [];
    const allDistributions = data.mediaCorrelations.flatMap(item => [
      { type: item.mediaType, timing: 'Immediate (<5min)', count: item.responseDistribution.immediate },
      { type: item.mediaType, timing: 'Quick (5-30min)', count: item.responseDistribution.quick },
      { type: item.mediaType, timing: 'Delayed (30min-2hr)', count: item.responseDistribution.delayed },
      { type: item.mediaType, timing: 'Late (>2hr)', count: item.responseDistribution.late }
    ]);
    
    type TimingBucket = {
      timing: string;
      total: number;
    } & Partial<Record<'photo' | 'video' | 'attachment', number>>;

    return allDistributions.reduce((acc, curr) => {
      const existing = acc.find(item => item.timing === curr.timing);
      if (existing) {
        existing[curr.type] = (existing[curr.type] || 0) + curr.count;
        existing.total += curr.count;
      } else {
        const bucket: TimingBucket = {
          timing: curr.timing,
          total: curr.count
        };
        bucket[curr.type] = curr.count;
        acc.push(bucket);
      }
      return acc;
    }, [] as TimingBucket[]);
  }, [data]);

  // Process sender engagement data
  const senderData = useMemo(() => {
    if (!data) return [];
    return data.senderEngagement.slice(0, 10).map(sender => ({
      sender: sender.sender.length > 15 ? sender.sender.substring(0, 15) + '...' : sender.sender,
      fullSender: sender.sender,
      mediaShared: sender.mediaShared.total,
      responsesReceived: sender.engagementReceived.totalResponses,
      engagementScore: sender.engagementReceived.engagementScore,
      avgResponseTime: sender.engagementReceived.avgResponseTime,
      photos: sender.mediaShared.photos,
      videos: sender.mediaShared.videos,
      attachments: sender.mediaShared.attachments,
      engagementRatio: sender.engagementRatio
    }));
  }, [data]);

  // Process hourly engagement data
  const hourlyData = useMemo(() => {
    if (!data) return [];
    return data.timeBasedEngagement.map(item => ({
      hour: `${item.hour.toString().padStart(2, '0')}:00`,
      mediaCount: item.mediaCount,
      avgEngagement: item.avgEngagement,
      responseRate: item.responseRate
    }));
  }, [data]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading media engagement data...</div>;
  }

  if (filteredUnavailable) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        <div className="text-center max-w-md px-4">
          <p className="font-medium text-gray-700 mb-1">Media engagement is global-only</p>
          <p className="text-sm text-gray-500">
            Per-conversation breakdown unavailable. Clear the conversation filter to view aggregate media engagement.
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No media engagement data available
      </div>
    );
  }

  const renderSummaryStats = () => (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
      <div className="bg-blue-50 rounded-lg p-4">
        <div className="text-blue-600 text-sm font-medium">Media Messages</div>
        <div className="text-2xl font-bold text-blue-900">
          {data.summary.totalMediaMessages.toLocaleString()}
        </div>
      </div>
      
      <div className="bg-green-50 rounded-lg p-4">
        <div className="text-green-600 text-sm font-medium">Avg Engagement</div>
        <div className="text-2xl font-bold text-green-900">
          {data.summary.avgEngagementRate.toFixed(1)}%
        </div>
      </div>
      
      <div className="bg-purple-50 rounded-lg p-4">
        <div className="text-purple-600 text-sm font-medium">Active Sharers</div>
        <div className="text-2xl font-bold text-purple-900">
          {data.summary.totalSenders}
        </div>
      </div>
      
      <div className="bg-orange-50 rounded-lg p-4">
        <div className="text-orange-600 text-sm font-medium">Most Engaging</div>
        <div className="text-2xl font-bold text-orange-900">
          {data.summary.mostEngagingType}
        </div>
      </div>
    </div>
  );

  const renderCorrelationsView = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Media Type Engagement Rates */}
        <div className="bg-white rounded-lg border p-4">
          <h4 className="text-lg font-semibold mb-4">📊 Engagement by Media Type</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={correlationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" />
              <YAxis />
              <Tooltip 
                formatter={(value, name) => [
                  name === 'engagement' ? `${Number(value).toFixed(1)}%` : Number(value).toFixed(2),
                  name === 'engagement' ? 'Engagement Rate' : 'Avg Responses'
                ]}
              />
              <Bar dataKey="engagement" fill="#3b82f6" name="engagement" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Response Time Distribution */}
        <div className="bg-white rounded-lg border p-4">
          <h4 className="text-lg font-semibold mb-4">⏱️ Response Timing Distribution</h4>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={timingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timing" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="photo" stackId="a" fill={MEDIA_TYPE_COLORS.photo} />
              <Bar dataKey="video" stackId="a" fill={MEDIA_TYPE_COLORS.video} />
              <Bar dataKey="attachment" stackId="a" fill={MEDIA_TYPE_COLORS.attachment} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Media Type Details */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {data.mediaCorrelations.map((item) => (
          <div key={item.mediaType} className="bg-white rounded-lg border p-4">
            <h5 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <span style={{ color: MEDIA_TYPE_COLORS[item.mediaType] }}>
                {item.mediaType === 'photo' ? '📸' : item.mediaType === 'video' ? '🎥' : '📎'}
              </span>
              {item.mediaType.charAt(0).toUpperCase() + item.mediaType.slice(1)}s
            </h5>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Total Shared:</span>
                <span className="bg-gray-100 px-2 py-1 rounded text-sm font-medium">
                  {item.totalCount.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Avg Responses:</span>
                <span className="bg-blue-100 px-2 py-1 rounded text-sm font-medium">
                  {item.avgResponseCount.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Engagement Rate:</span>
                <span className={`px-2 py-1 rounded text-sm font-medium ${
                  item.engagementRate > 50 ? 'bg-green-100 text-green-800' : 'bg-gray-100'
                }`}>
                  {item.engagementRate.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Avg Response Time:</span>
                <span className="bg-orange-100 px-2 py-1 rounded text-sm font-medium">
                  {formatTime(item.avgResponseTime)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderSendersView = () => (
    <div className="space-y-6">
      {/* Top Senders Chart */}
      <div className="bg-white rounded-lg border p-4">
        <h4 className="text-lg font-semibold mb-4">👥 Top Media Sharers by Engagement</h4>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={senderData} layout="horizontal">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="sender" type="category" width={100} />
            <Tooltip 
              formatter={(value, name) => [
                value,
                name === 'engagementScore' ? 'Engagement Score' : 'Media Shared'
              ]}
            />
            <Bar dataKey="engagementScore" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Sender Details Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {senderData.slice(0, 6).map((sender, index) => (
          <div key={sender.fullSender} className="bg-white rounded-lg border p-4">
            <h5 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-sm font-bold">
                {index + 1}
              </div>
              {sender.sender}
            </h5>
            <div className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-center">
                <div>
                  <p className="text-sm text-gray-600">Photos</p>
                  <p className="font-bold text-blue-600">{sender.photos}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Videos</p>
                  <p className="font-bold text-red-600">{sender.videos}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Files</p>
                  <p className="font-bold text-green-600">{sender.attachments}</p>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Responses Received:</span>
                  <span className="bg-gray-100 px-2 py-1 rounded text-sm font-medium">
                    {sender.responsesReceived}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Engagement Score:</span>
                  <span className="bg-blue-100 px-2 py-1 rounded text-sm font-medium">
                    {sender.engagementScore}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Avg Response Time:</span>
                  <span className="bg-orange-100 px-2 py-1 rounded text-sm font-medium">
                    {formatTime(sender.avgResponseTime)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderTimingView = () => (
    <div className="space-y-6">
      {/* Hourly Media Activity */}
      <div className="bg-white rounded-lg border p-4">
        <h4 className="text-lg font-semibold mb-4">⏰ Media Sharing & Engagement by Hour</h4>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={hourlyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip 
              formatter={(value, name) => [
                name === 'responseRate' ? `${Number(value).toFixed(1)}%` : Number(value).toFixed(1),
                name === 'mediaCount' ? 'Media Shared' : 
                name === 'avgEngagement' ? 'Avg Engagement' : 'Response Rate'
              ]}
            />
            <Bar yAxisId="left" dataKey="mediaCount" fill="#3b82f6" opacity={0.6} />
            <Line yAxisId="right" type="monotone" dataKey="responseRate" stroke="#ef4444" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Peak Activity Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border p-4">
          <h5 className="text-lg font-semibold mb-4">📈 Peak Media Sharing Hours</h5>
          <div className="space-y-3">
            {hourlyData
              .sort((a, b) => b.mediaCount - a.mediaCount)
              .slice(0, 5)
              .map((item, index) => (
                <div key={item.hour} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center text-xs font-bold">
                      {index + 1}
                    </div>
                    <span className="font-medium">{item.hour}</span>
                  </div>
                  <span className="bg-gray-100 px-2 py-1 rounded text-sm font-medium">
                    {item.mediaCount} media
                  </span>
                </div>
              ))}
          </div>
        </div>

        <div className="bg-white rounded-lg border p-4">
          <h5 className="text-lg font-semibold mb-4">🎯 Highest Engagement Hours</h5>
          <div className="space-y-3">
            {hourlyData
              .sort((a, b) => b.responseRate - a.responseRate)
              .slice(0, 5)
              .map((item, index) => (
                <div key={item.hour} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center text-xs font-bold">
                      {index + 1}
                    </div>
                    <span className="font-medium">{item.hour}</span>
                  </div>
                  <span className="bg-green-100 px-2 py-1 rounded text-sm font-medium">
                    {item.responseRate.toFixed(1)}% response
                  </span>
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-xl font-bold text-gray-900 mb-2">📊 Media-Engagement Correlation Analysis</h3>
        <p className="text-gray-600 mb-4">
          Analyzing how different media types correlate with engagement and response patterns
        </p>
      </div>

      {renderSummaryStats()}
      
      <div className="bg-white rounded-lg border">
        <div className="p-4 border-b">
          <div className="flex gap-2">
            <button
              onClick={() => setActiveView('correlations')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeView === 'correlations'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Media Types
            </button>
            <button
              onClick={() => setActiveView('senders')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeView === 'senders'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Top Sharers
            </button>
            <button
              onClick={() => setActiveView('timing')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeView === 'timing'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Timing Analysis
            </button>
          </div>
        </div>
        <div className="p-6">
          {activeView === 'correlations' && renderCorrelationsView()}
          {activeView === 'senders' && renderSendersView()}
          {activeView === 'timing' && renderTimingView()}
        </div>
      </div>
    </div>
  );
} 