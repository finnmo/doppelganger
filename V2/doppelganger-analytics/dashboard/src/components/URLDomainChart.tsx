'use client';

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface URLMetric {
  domain: string;
  count: number;
  sender: string;
  conversation_id: string;
  first_seen: number;
  last_seen: number;
}

interface AggregatedURLData {
  domain: string;
  total_count: number;
  unique_senders: number;
  conversations: number;
  first_seen: number;
  last_seen: number;
}

export function URLDomainChart() {
  const [data, setData] = useState<AggregatedURLData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch('/data/urlMetrics.json');
        if (!response.ok) {
          throw new Error(`Failed to load URL data: ${response.status}`);
        }
        
        let urlData: URLMetric[] = await response.json();
        
        if (!Array.isArray(urlData)) {
          throw new Error('Invalid URL data format');
        }
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          urlData = urlData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
        }
        
        // Aggregate URL data by domain
        const domainMap = new Map<string, {
          totalCount: number;
          senders: Set<string>;
          conversations: Set<string>;
          firstSeen: number;
          lastSeen: number;
        }>();

        urlData.forEach(item => {
          if (!domainMap.has(item.domain)) {
            domainMap.set(item.domain, {
              totalCount: 0,
              senders: new Set(),
              conversations: new Set(),
              firstSeen: item.first_seen,
              lastSeen: item.last_seen
            });
          }
          
          const domainStats = domainMap.get(item.domain)!;
          domainStats.totalCount += item.count;
          domainStats.senders.add(item.sender);
          domainStats.conversations.add(item.conversation_id);
          domainStats.firstSeen = Math.min(domainStats.firstSeen, item.first_seen);
          domainStats.lastSeen = Math.max(domainStats.lastSeen, item.last_seen);
        });

        // Convert to array and sort by total count
        const aggregatedData = Array.from(domainMap.entries())
          .map(([domain, stats]) => ({
            domain: domain.replace(/^(https?:\/\/)?(www\.)?/, ''), // Clean up domain names
            total_count: stats.totalCount,
            unique_senders: stats.senders.size,
            conversations: stats.conversations.size,
            first_seen: stats.firstSeen,
            last_seen: stats.lastSeen
          }))
          .sort((a, b) => b.total_count - a.total_count)
          .slice(0, 12); // Top 12 domains

        console.log('URL Domain Chart Data:', aggregatedData); // Debug log
        setData(aggregatedData);
      } catch (error) {
        console.error('Error loading URL data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load URL data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading URL data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-500 text-center">
          <div>Error loading URL data</div>
          <div className="text-xs mt-1">{error}</div>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center text-gray-500">
          <div>No URL data available</div>
          {isFiltered && (
            <div className="text-xs mt-1">
              for selected conversations
            </div>
          )}
        </div>
      </div>
    );
  }

  // Format domain names for better display on X-axis
  const formatDomain = (domain: string) => {
    // Clean up domain and make it shorter for X-axis
    let formatted = domain.replace(/^(https?:\/\/)?(www\.)?/, '');
    
    // Special cases for common domains
    if (formatted.includes('tiktok')) return 'TikTok';
    if (formatted.includes('youtube') || formatted.includes('youtu.be')) return 'YouTube';
    if (formatted.includes('instagram')) return 'Instagram';
    if (formatted.includes('facebook')) return 'Facebook';
    if (formatted.includes('google')) return 'Google';
    if (formatted.includes('apple')) return 'Apple Music';
    if (formatted.includes('spotify')) return 'Spotify';
    if (formatted.includes('netflix')) return 'Netflix';
    if (formatted.includes('amazon')) return 'Amazon';
    if (formatted.includes('airbnb')) return 'Airbnb';
    if (formatted.includes('ikea')) return 'IKEA';
    
    // For other domains, truncate at first dot or 10 chars
    const firstDot = formatted.indexOf('.');
    if (firstDot > 0 && firstDot < 12) {
      formatted = formatted.substring(0, firstDot);
    } else if (formatted.length > 12) {
      formatted = formatted.substring(0, 10) + '...';
    }
    
    return formatted;
  };

  const chartData = data.map(item => ({
    ...item,
    displayDomain: formatDomain(item.domain),
    fullDomain: item.domain
  }));

  const getBarColor = (domain: string) => {
    // Color coding based on domain type
    if (domain.includes('youtube') || domain.includes('youtu.be')) return '#ff0000';
    if (domain.includes('instagram') || domain.includes('ig')) return '#e4405f';
    if (domain.includes('twitter') || domain.includes('x.com')) return '#1da1f2';
    if (domain.includes('tiktok')) return '#000000';
    if (domain.includes('facebook')) return '#1877f2';
    if (domain.includes('spotify')) return '#1db954';
    if (domain.includes('netflix')) return '#e50914';
    if (domain.includes('amazon')) return '#ff9900';
    if (domain.includes('google')) return '#4285f4';
    if (domain.includes('apple')) return '#007aff';
    return '#6366f1'; // Default indigo
  };

  return (
    <div className="flex h-full flex-col">
      {isFiltered && (
        <div className="mb-1 shrink-0 text-center text-xs text-blue-600">
          Filtered across {selectedConversations.length} selected conversation{selectedConversations.length !== 1 ? 's' : ''}
        </div>
      )}

      <div className="min-h-0 flex-1">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={chartData} 
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis 
            dataKey="displayDomain"
            tick={{ fontSize: 10 }}
            height={60}
            interval={0}
            axisLine={{ stroke: '#6b7280' }}
            tickLine={{ stroke: '#6b7280' }}
            angle={-45}
          />
          <YAxis 
            tickFormatter={(value) => value.toLocaleString()}
            tick={{ fontSize: 11 }}
            axisLine={{ stroke: '#6b7280' }}
            tickLine={{ stroke: '#6b7280' }}
            label={{ value: 'Shares', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload && payload.length > 0) {
                const data = payload[0].payload;
                const firstSeenDate = new Date(data.first_seen).toLocaleDateString();
                const lastSeenDate = new Date(data.last_seen).toLocaleDateString();
                
                return (
                  <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg z-50">
                    <div className="text-sm">
                      <div className="font-semibold text-gray-900">{data.fullDomain}</div>
                      <div className="text-gray-700">Shared: {data.total_count.toLocaleString()} time{data.total_count !== 1 ? 's' : ''}</div>
                      <div className="text-gray-700">By: {data.unique_senders} sender{data.unique_senders !== 1 ? 's' : ''}</div>
                      <div className="text-gray-700">In: {data.conversations} conversation{data.conversations !== 1 ? 's' : ''}</div>
                      <div className="text-xs text-gray-500 mt-1 border-t pt-1">
                        First: {firstSeenDate} | Last: {lastSeenDate}
                      </div>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <Bar 
            dataKey="total_count" 
            radius={[4, 4, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.domain)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
} 