'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { Payload, ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { GitBranch, MessageSquare, Users, TrendingUp } from 'lucide-react';

interface ThreadDepthDistribution {
  depth: number;
  count: number;
  percentage: number;
}

interface ThreadData {
  conversation_id: string;
  thread_depth: number;
  thread_count: number;
  avg_thread_length: number;
  max_thread_depth: number;
  participants_in_threads: number;
  depth_distribution?: ThreadDepthDistribution[];
  thread_starters: Array<{
    sender: string;
    threads_started: number;
    avg_thread_depth?: number;
  }>;
}

interface ThreadMetrics {
  summary: {
    total_threads: number;
    avg_depth: number;
    max_depth: number;
    total_conversations: number;
    threaded_conversations: number;
  };
  depth_distribution: ThreadDepthDistribution[];
  conversation_threads: ThreadData[];
  top_thread_starters: Array<{
    sender: string;
    threads_started: number;
    avg_thread_depth: number;
  }>;
}

const DEPTH_COLORS = [
  '#e3f2fd', // Very light blue - depth 1
  '#bbdefb', // Light blue - depth 2  
  '#90caf9', // Medium light blue - depth 3
  '#64b5f6', // Medium blue - depth 4
  '#42a5f5', // Medium dark blue - depth 5
  '#2196f3', // Blue - depth 6
  '#1e88e5', // Dark blue - depth 7
  '#1976d2', // Darker blue - depth 8
  '#1565c0', // Very dark blue - depth 9+
];

export function ThreadVisualization() {
  const [data, setData] = useState<ThreadMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'distribution' | 'starters'>('distribution');
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/threadAnalysis.json');
        const threadData: ThreadMetrics = await response.json();
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          threadData.conversation_threads = threadData.conversation_threads.filter(thread =>
            selectedConversations.includes(thread.conversation_id)
          );

          const filteredThreads = threadData.conversation_threads;
          const totalThreads = filteredThreads.reduce((sum, conv) => sum + conv.thread_count, 0);
          const avgDepth = filteredThreads.length > 0
            ? filteredThreads.reduce((sum, conv) => sum + conv.avg_thread_length, 0) / filteredThreads.length
            : 0;
          const maxDepth = Math.max(...filteredThreads.map(conv => conv.max_thread_depth), 0);

          threadData.summary = {
            total_threads: totalThreads,
            avg_depth: Math.round(avgDepth * 10) / 10,
            max_depth: maxDepth,
            total_conversations: filteredThreads.length,
            threaded_conversations: filteredThreads.filter(conv => conv.thread_count > 0).length
          };

          // Merge real depth_distribution arrays only — never invent histogram bars
          const depthCounts = new Map<number, number>();
          let distributionsFound = 0;
          filteredThreads.forEach(conv => {
            if (!conv.depth_distribution || conv.depth_distribution.length === 0) return;
            distributionsFound++;
            conv.depth_distribution.forEach(entry => {
              depthCounts.set(entry.depth, (depthCounts.get(entry.depth) || 0) + entry.count);
            });
          });

          if (distributionsFound === 0) {
            threadData.depth_distribution = [];
          } else {
            const totalDepthEntries = Array.from(depthCounts.values()).reduce((a, b) => a + b, 0);
            threadData.depth_distribution = Array.from(depthCounts.entries()).map(([depth, count]) => ({
              depth,
              count,
              percentage: totalDepthEntries > 0 ? (count / totalDepthEntries) * 100 : 0
            })).sort((a, b) => a.depth - b.depth);
          }

          // Rebuild top_thread_starters from filtered conversations
          const starterMap = new Map<string, { threads_started: number; depthSum: number; depthCount: number }>();
          filteredThreads.forEach(conv => {
            (conv.thread_starters || []).forEach(starter => {
              if (!starterMap.has(starter.sender)) {
                starterMap.set(starter.sender, { threads_started: 0, depthSum: 0, depthCount: 0 });
              }
              const stats = starterMap.get(starter.sender)!;
              stats.threads_started += starter.threads_started;
              if (typeof starter.avg_thread_depth === 'number') {
                stats.depthSum += starter.avg_thread_depth;
                stats.depthCount++;
              }
            });
          });

          threadData.top_thread_starters = Array.from(starterMap.entries())
            .map(([sender, stats]) => ({
              sender,
              threads_started: stats.threads_started,
              avg_thread_depth: stats.depthCount > 0 ? stats.depthSum / stats.depthCount : 0
            }))
            .sort((a, b) => b.threads_started - a.threads_started);
        }
        
        setData(threadData);
      } catch (error) {
        console.error('Error loading thread data:', error);
        // Create fallback data structure
        setData({
          summary: {
            total_threads: 0,
            avg_depth: 0,
            max_depth: 0,
            total_conversations: 0,
            threaded_conversations: 0
          },
          depth_distribution: [],
          conversation_threads: [],
          top_thread_starters: []
        });
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading thread analysis...</p>
        </div>
      </div>
    );
  }

  if (!data || data.summary.total_threads === 0) {
    return (
      <div className="h-64 flex items-center justify-center bg-gray-50 rounded border-2 border-dashed border-gray-300">
        <div className="text-center">
          <GitBranch className="w-12 h-12 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-500">No Thread Data Available</p>
          <p className="text-sm text-gray-400">
            {isFiltered ? 'No threads found in selected conversations' : 'Thread analysis data not available'}
          </p>
        </div>
      </div>
    );
  }

  const getDepthColor = (depth: number): string => {
    return DEPTH_COLORS[Math.min(depth - 1, DEPTH_COLORS.length - 1)];
  };

  const formatTooltip = (value: ValueType | undefined, entry: Payload<ValueType, NameType>) => {
    if (viewMode === 'distribution') {
      return [
        <div key="tooltip" className="text-sm">
          <div className="font-semibold">Depth {entry.payload.depth}</div>
          <div>Threads: {Number(value).toLocaleString()}</div>
          <div>Percentage: {entry.payload.percentage.toFixed(1)}%</div>
        </div>
      ];
    } else {
      return [
        <div key="tooltip" className="text-sm">
          <div className="font-semibold">{entry.payload.sender}</div>
          <div>Threads Started: {Number(value).toLocaleString()}</div>
          <div>Avg Depth: {entry.payload.avg_thread_depth?.toFixed(1) || 'N/A'}</div>
        </div>
      ];
    }
  };

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="flex items-center">
            <GitBranch className="w-4 h-4 text-blue-600 mr-2" />
            <div className="text-blue-600 text-sm font-medium">Total Threads</div>
          </div>
          <div className="text-xl font-bold text-blue-900">{data.summary.total_threads.toLocaleString()}</div>
        </div>
        
        <div className="bg-green-50 rounded-lg p-3">
          <div className="flex items-center">
            <TrendingUp className="w-4 h-4 text-green-600 mr-2" />
            <div className="text-green-600 text-sm font-medium">Avg Depth</div>
          </div>
          <div className="text-xl font-bold text-green-900">{data.summary.avg_depth}</div>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-3">
          <div className="flex items-center">
            <MessageSquare className="w-4 h-4 text-purple-600 mr-2" />
            <div className="text-purple-600 text-sm font-medium">Max Depth</div>
          </div>
          <div className="text-xl font-bold text-purple-900">{data.summary.max_depth}</div>
        </div>
        
        <div className="bg-orange-50 rounded-lg p-3">
          <div className="flex items-center">
            <Users className="w-4 h-4 text-orange-600 mr-2" />
            <div className="text-orange-600 text-sm font-medium">Threaded Convs</div>
          </div>
          <div className="text-xl font-bold text-orange-900">{data.summary.threaded_conversations}</div>
        </div>
      </div>

      {/* View Mode Toggle */}
      <div className="flex justify-center">
        <div className="bg-gray-100 rounded-lg p-1 flex">
          <button
            onClick={() => setViewMode('distribution')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              viewMode === 'distribution'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            📊 Depth Distribution
          </button>
          <button
            onClick={() => setViewMode('starters')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              viewMode === 'starters'
                ? 'bg-white text-blue-600 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            🚀 Thread Starters
          </button>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === 'distribution' ? (
            <BarChart data={data.depth_distribution} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="depth" 
                stroke="#6b7280"
                fontSize={12}
                label={{ value: 'Thread Depth', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value) => value.toLocaleString()}
                label={{ value: 'Number of Threads', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        {formatTooltip(payload[0].value, payload[0])}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {data.depth_distribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getDepthColor(entry.depth)} />
                ))}
              </Bar>
            </BarChart>
          ) : (
            <BarChart data={data.top_thread_starters.slice(0, 10)} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="sender" 
                stroke="#6b7280"
                fontSize={10}
                angle={-45}
                textAnchor="end"
                height={60}
                interval={0}
              />
              <YAxis 
                stroke="#6b7280"
                fontSize={12}
                tickFormatter={(value) => value.toLocaleString()}
              />
              <Tooltip 
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
                        {formatTooltip(payload[0].value, payload[0])}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="threads_started" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Thread Insights */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
        <h4 className="font-semibold mb-3 flex items-center">
          <GitBranch className="w-4 h-4 mr-2 text-blue-600" />
          Thread Analysis Insights
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white p-3 rounded border border-blue-100">
            <div className="text-sm font-medium text-gray-700">Thread Complexity</div>
            <div className="text-lg font-bold text-blue-600">
              {data.summary.avg_depth < 2 ? 'Simple' : 
               data.summary.avg_depth < 4 ? 'Moderate' : 'Complex'}
            </div>
            <div className="text-xs text-gray-500">
              Based on average depth of {data.summary.avg_depth}
            </div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-100">
            <div className="text-sm font-medium text-gray-700">Threading Rate</div>
            <div className="text-lg font-bold text-green-600">
              {data.summary.total_conversations > 0 
                ? Math.round((data.summary.threaded_conversations / data.summary.total_conversations) * 100)
                : 0}%
            </div>
            <div className="text-xs text-gray-500">
              Conversations with threads
            </div>
          </div>
          <div className="bg-white p-3 rounded border border-blue-100">
            <div className="text-sm font-medium text-gray-700">Top Thread Starter</div>
            <div className="text-lg font-bold text-purple-600">
              {data.top_thread_starters[0]?.sender.split(' ')[0] || 'N/A'}
            </div>
            <div className="text-xs text-gray-500">
              {data.top_thread_starters[0]?.threads_started || 0} threads started
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 