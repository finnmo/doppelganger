'use client';

import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { FileText, Image as ImageIcon, Link, Smile, Hash, MessageSquare, Phone, Settings } from 'lucide-react';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface ContentType {
  conversation_id: string;
  type: string;
  count: number;
  percentage: number;
  examples: string[];
  avgLength: number;
  uniqueSenders: number;
}

interface AggregatedContentType {
  type: string;
  count: number;
  percentage: number;
  examples: string[];
  avgLength: number;
  uniqueSenders: number;
}

interface ContentTypeData {
  summary: {
    totalMessages: number;
    totalTypes: number;
    avgMessageLength: number;
    totalConversations: number;
  };
  contentTypes: ContentType[];
}

const ContentTypeChart: React.FC = () => {
  const [data, setData] = useState<ContentTypeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredType, setHoveredType] = useState<string | null>(null);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/contentTypeMetrics.json');
        const contentData: ContentTypeData = await response.json();
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          contentData.contentTypes = contentData.contentTypes.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
          
          // Recalculate summary for filtered data
          const filteredMessages = contentData.contentTypes.reduce((sum, item) => sum + item.count, 0);
          const filteredTypes = new Set(contentData.contentTypes.map(item => item.type)).size;
          const avgLength = contentData.contentTypes.length > 0 
            ? contentData.contentTypes.reduce((sum, item) => sum + (item.avgLength * item.count), 0) / filteredMessages
            : 0;
          
          contentData.summary = {
            totalMessages: filteredMessages,
            totalTypes: filteredTypes,
            avgMessageLength: Math.round(avgLength),
            totalConversations: new Set(contentData.contentTypes.map(item => item.conversation_id)).size
          };
        }
        
        setData(contentData);
      } catch (error) {
        console.error('Error loading content type data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  // Aggregate content types across conversations for display
  const aggregatedData = React.useMemo(() => {
    if (!data) return [];
    
    const typeMap = new Map<string, {
      count: number;
      totalLength: number;
      examples: Set<string>;
      senders: Set<number>;
    }>();

    data.contentTypes.forEach(item => {
      if (!typeMap.has(item.type)) {
        typeMap.set(item.type, {
          count: 0,
          totalLength: 0,
          examples: new Set(),
          senders: new Set()
        });
      }
      
      const typeData = typeMap.get(item.type)!;
      typeData.count += item.count;
      typeData.totalLength += item.avgLength * item.count;
      item.examples.forEach(ex => typeData.examples.add(ex));
      typeData.senders.add(item.uniqueSenders);
    });

    const totalMessages = data.summary.totalMessages;
    
    return Array.from(typeMap.entries()).map(([type, typeData]) => ({
      type,
      count: typeData.count,
      percentage: (typeData.count / totalMessages) * 100,
      examples: Array.from(typeData.examples).slice(0, 3),
      avgLength: Math.round(typeData.totalLength / typeData.count),
      uniqueSenders: Math.max(...Array.from(typeData.senders))
    })).sort((a, b) => b.count - a.count);
  }, [data]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'short_text':
      case 'medium_text':
      case 'long_text':
        return <FileText className="w-4 h-4" />;
      case 'emoji_only':
        return <Smile className="w-4 h-4" />;
      case 'single_word':
        return <MessageSquare className="w-4 h-4" />;
      case 'link_share':
        return <Link className="w-4 h-4" />;
      case 'media_notification':
        return <ImageIcon className="w-4 h-4" />;
      case 'call_event':
        return <Phone className="w-4 h-4" />;
      case 'system_event':
        return <Settings className="w-4 h-4" />;
      case 'symbols_only':
        return <Hash className="w-4 h-4" />;
      default:
        return <MessageSquare className="w-4 h-4" />;
    }
  };

  const getTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      'short_text': 'Short Text',
      'medium_text': 'Medium Text',
      'long_text': 'Long Text',
      'single_word': 'Single Word',
      'emoji_only': 'Emojis Only',
      'link_share': 'Link Shares',
      'media_notification': 'Media Messages',
      'call_event': 'Call Events',  
      'system_event': 'System Events',
      'symbols_only': 'Symbols Only',
      'reaction': 'Reactions'
    };
    return labels[type] || type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getTypeColor = (type: string, index: number) => {
    const colors = [
      '#3B82F6', // blue
      '#10B981', // emerald
      '#F59E0B', // amber
      '#EF4444', // red
      '#8B5CF6', // violet
      '#06B6D4', // cyan
      '#F97316', // orange
      '#84CC16', // lime
      '#EC4899', // pink
      '#6B7280'  // gray
    ];
    return colors[index % colors.length];
  };

  const CustomTooltip = ({ active, payload }: {
    active?: boolean;
    payload?: Array<{ payload: AggregatedContentType }>;
  }) => {
    if (active && payload && payload.length > 0) {
      const data = payload[0].payload;
      
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg max-w-xs z-50">
          <div className="flex items-center gap-2 mb-2">
            {getTypeIcon(data.type)}
            <span className="font-semibold">{getTypeLabel(data.type)}</span>
          </div>
          <div className="space-y-1 text-sm">
            <p><strong>Count:</strong> {data.count.toLocaleString()}</p>
            <p><strong>Percentage:</strong> {data.percentage.toFixed(1)}%</p>
            <p><strong>Avg Length:</strong> {data.avgLength} chars</p>
            <p><strong>Unique Senders:</strong> {data.uniqueSenders}</p>
            {data.examples && data.examples.length > 0 && (
              <div>
                <p className="font-medium mt-2">Examples:</p>
                <div className="max-w-xs">
                  {data.examples.slice(0, 2).map((example: string, idx: number) => (
                    <p key={idx} className="text-xs text-gray-600 truncate bg-gray-50 px-2 py-1 rounded mt-1">
                      &ldquo;{example}&rdquo;
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-500">No data available</p>
      </div>
    );
  }

  const chartData = aggregatedData.map((type, index) => ({
    ...type,
    color: getTypeColor(type.type, index),
    label: getTypeLabel(type.type)
  }));

  return (
    <div className="h-full flex flex-col">
      <div className="mb-3 flex-shrink-0">
        <div className="grid grid-cols-3 gap-4 text-sm text-gray-600">
          <div>
            <span className="font-medium text-gray-900">{data.summary.totalMessages.toLocaleString()}</span>
            <br />Total Messages
          </div>
          <div>
            <span className="font-medium text-gray-900">{data.summary.totalTypes}</span>
            <br />Content Types
          </div>
          <div>
            <span className="font-medium text-gray-900">{data.summary.avgMessageLength}</span>
            <br />Avg Length (chars)
          </div>
        </div>
        {isFiltered && (
          <div className="mt-2 text-xs text-blue-600">
            Showing data for {data.summary.totalConversations} selected conversation{data.summary.totalConversations !== 1 ? 's' : ''}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
        {/* Chart */}
        <div className="relative min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={85}
                paddingAngle={2}
                dataKey="count"
              >
                {chartData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={entry.color}
                    stroke={hoveredType === entry.type ? '#374151' : 'none'}
                    strokeWidth={hoveredType === entry.type ? 2 : 0}
                  />
                ))}
              </Pie>
              <Tooltip 
                content={<CustomTooltip />}
                offset={10}
                allowEscapeViewBox={{ x: true, y: true }}
                wrapperStyle={{ zIndex: 1000 }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Legend & Details */}
        <div className="space-y-1.5 overflow-y-auto min-h-0">
          {chartData.slice(0, 8).map((type) => (
            <div
              key={type.type}
              className="flex items-center justify-between p-1.5 rounded hover:bg-gray-50 transition-colors cursor-pointer"
              onMouseEnter={() => setHoveredType(type.type)}  
              onMouseLeave={() => setHoveredType(null)}
            >
              <div className="flex items-center gap-2.5">
                <div
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: type.color }}
                />
                <div className="flex items-center gap-1.5">
                  {React.cloneElement(getTypeIcon(type.type), { className: "w-3.5 h-3.5" })}
                  <span className="text-sm font-medium">{type.label}</span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium">{type.percentage.toFixed(1)}%</div>
                <div className="text-xs text-gray-500">{type.count.toLocaleString()}</div>
              </div>
            </div>
          ))}
          
          {chartData.length > 8 && (
            <div className="text-xs text-gray-500 pt-1.5 border-t">
              +{chartData.length - 8} more types
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContentTypeChart; 