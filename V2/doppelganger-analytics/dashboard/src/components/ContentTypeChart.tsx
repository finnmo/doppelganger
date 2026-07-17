'use client';

import React, { useState, useEffect } from 'react';
import { ChartTooltip } from '@/components/ui/ChartTooltip';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { FileText, Image as ImageIcon, Link, Smile, Hash, MessageSquare, Phone, Settings } from 'lucide-react';
import { useParticipantScope } from '@/hooks/useParticipantScope';

interface ContentType {
  conversation_id: string;
  type: string;
  count: number;
  percentage: number;
  examples: string[];
  avgLength: number;
  uniqueSenders: number;
  representativeExample?: {
    text: string;
    sender: string;
    conversation_id: string;
  };
}

interface AggregatedContentType {
  type: string;
  count: number;
  percentage: number;
  examples: string[];
  avgLength: number;
  uniqueSenders: number;
  representativeExample?: {
    text: string;
    sender: string;
    conversation_id: string;
  };
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

interface ContentTypeChartProps {
  /** Pie + legend in card; full breakdown table for fullscreen. */
  variant?: 'pie' | 'table';
}

const ContentTypeChart: React.FC<ContentTypeChartProps> = ({ variant = 'pie' }) => {
  const [data, setData] = useState<ContentTypeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [hoveredType, setHoveredType] = useState<string | null>(null);
  const { filterScopedRows, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/contentTypeMetrics.json');
        let contentData: ContentTypeData = await response.json();
        contentData = {
          ...contentData,
          contentTypes: filterScopedRows(contentData.contentTypes),
        };
        const filteredMessages = contentData.contentTypes.reduce((sum, item) => sum + item.count, 0);
        contentData.summary = {
          ...contentData.summary,
          totalMessages: filteredMessages,
          totalConversations: new Set(contentData.contentTypes.map(i => i.conversation_id)).size,
        };
        
        setData(contentData);
      } catch (error) {
        console.error('Error loading content type data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds]);

  // Aggregate content types across conversations for display
  const aggregatedData = React.useMemo(() => {
    if (!data) return [];
    
    const typeMap = new Map<string, {
      count: number;
      totalLength: number;
      examples: Set<string>;
      senders: Set<number>;
      representativeExample?: { text: string; sender: string; conversation_id: string };
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
      if (!typeData.representativeExample && item.representativeExample) {
        typeData.representativeExample = item.representativeExample;
      }
    });

    const totalMessages = data.summary.totalMessages;
    
    return Array.from(typeMap.entries()).map(([type, typeData]) => ({
      type,
      count: typeData.count,
      percentage: (typeData.count / totalMessages) * 100,
      examples: Array.from(typeData.examples).slice(0, 3),
      avgLength: Math.round(typeData.totalLength / typeData.count),
      uniqueSenders: Math.max(...Array.from(typeData.senders)),
      representativeExample: typeData.representativeExample,
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

  const Custom = ({ active, payload }: {
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
            {data.representativeExample ? (
              <p className="text-gray-600 italic">
                e.g. &ldquo;{data.representativeExample.text}&rdquo; — {data.representativeExample.sender}
              </p>
            ) : data.examples && data.examples.length > 0 ? (
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
            ) : null}
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

  if (variant === 'table') {
    return (
      <div className="flex h-full min-h-0 flex-col gap-3">
        <div className="grid shrink-0 grid-cols-1 gap-3 text-sm text-gray-600 sm:grid-cols-3 sm:gap-4">
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
        <div className="min-h-0 flex-1 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-white text-left text-xs text-gray-500">
              <tr>
                <th className="pb-2 pr-2">Type</th>
                <th className="pb-2 pr-2 text-right">Count</th>
                <th className="pb-2 pr-2 text-right">%</th>
                <th className="pb-2 pr-2 text-right">Avg Len</th>
                <th className="pb-2 text-right">Senders</th>
              </tr>
            </thead>
            <tbody>
              {chartData.map((type) => (
                <tr key={type.type} className="border-t border-gray-100 hover:bg-gray-50">
                  <td className="py-2 pr-2">
                    <div className="flex items-center gap-2">
                      <div className="h-3 w-3 shrink-0 rounded-full" style={{ backgroundColor: type.color }} />
                      {getTypeIcon(type.type)}
                      <span className="font-medium">{type.label}</span>
                    </div>
                  </td>
                  <td className="py-2 pr-2 text-right">{type.count.toLocaleString()}</td>
                  <td className="py-2 pr-2 text-right">{type.percentage.toFixed(1)}%</td>
                  <td className="py-2 pr-2 text-right">{type.avgLength}</td>
                  <td className="py-2 text-right">{type.uniqueSenders}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="mb-3 flex-shrink-0">
        <div className="grid grid-cols-1 gap-3 text-sm text-gray-600 sm:grid-cols-3 sm:gap-4">
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
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Chart */}
        <div className="relative min-h-0 h-full">
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
              <ChartTooltip 
                content={<Custom />}
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

export function ContentTypeFullscreen() {
  return <ContentTypeChart variant="table" />;
}

export default ContentTypeChart;