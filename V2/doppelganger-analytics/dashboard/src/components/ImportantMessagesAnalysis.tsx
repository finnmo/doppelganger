'use client';

import React, { useState, useEffect } from 'react';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import { Star, MessageCircle, Heart, Eye, TrendingUp, Clock, ChevronDown, ChevronUp } from 'lucide-react';

interface ImportantMessage {
  message_id: number;
  content: string;
  sender: string;
  importance_score: number;
  factors: string[];
  timestamp_ms: number;
  conversation_id: string;
}

interface ImportantMessagesAnalysisProps {
  /** Max messages to show (card: 20, fullscreen: 50). */
  limit?: number;
}

export function ImportantMessagesAnalysis({ limit = 20 }: ImportantMessagesAnalysisProps) {
  const [messages, setMessages] = useState<ImportantMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'top' | 'recent'>('top');
  const [expandedMessages, setExpandedMessages] = useState<Set<number>>(new Set());
  const { filterScopedRows, scopeConversationIds } = useParticipantScope();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/messageImportance.json');
        let importanceData: ImportantMessage[] = await response.json();
        importanceData = filterScopedRows(importanceData, { senderKey: 'sender' });
        
        const sortedData = importanceData
          .sort((a, b) => b.importance_score - a.importance_score)
          .slice(0, limit);
        
        setMessages(sortedData);
      } catch (error) {
        console.error('Error loading important messages:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds, limit]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading important messages...</div>;
  }

  const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getImportanceColor = (score: number): string => {
    if (score >= 0.8) return 'text-red-600 bg-red-50 border-red-200';
    if (score >= 0.6) return 'text-orange-600 bg-orange-50 border-orange-200';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-blue-600 bg-blue-50 border-blue-200';
  };

  const getFactorIcon = (factor: string) => {
    const lowerFactor = factor.toLowerCase();
    if (lowerFactor.includes('length')) return <MessageCircle className="w-3 h-3" />;
    if (lowerFactor.includes('emoji')) return <Heart className="w-3 h-3" />;
    if (lowerFactor.includes('caps')) return <TrendingUp className="w-3 h-3" />;
    if (lowerFactor.includes('question')) return <Eye className="w-3 h-3" />;
    if (lowerFactor.includes('url')) return <TrendingUp className="w-3 h-3" />;
    return <Star className="w-3 h-3" />;
  };

  const shouldTruncate = (content: string, maxLength: number = 150): boolean => {
    return content.length > maxLength;
  };

  const truncateContent = (content: string, maxLength: number = 150): string => {
    return content.length > maxLength ? content.substring(0, maxLength) + '...' : content;
  };

  const toggleExpanded = (messageId: number) => {
    setExpandedMessages(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  const displayedMessages = viewMode === 'top' 
    ? messages 
    : [...messages].sort((a, b) => b.timestamp_ms - a.timestamp_ms);

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      {/* Header Controls */}
      <div className="flex justify-between items-center">
        <div>
          <h4 className="text-sm font-medium text-gray-700">
            Top {messages.length} Important Messages
          </h4>
          <p className="text-xs text-gray-500">
            Ranked by importance factors including length, engagement, and content type
          </p>
        </div>
        
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('top')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              viewMode === 'top' 
                ? 'bg-yellow-500 text-white' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            By Importance
          </button>
          <button
            onClick={() => setViewMode('recent')}
            className={`px-3 py-1 text-xs rounded transition-colors ${
              viewMode === 'recent' 
                ? 'bg-yellow-500 text-white' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            By Date
          </button>
        </div>
      </div>

      {/* Messages List */}
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto">
        {displayedMessages.map((message, index) => {
          const isExpanded = expandedMessages.has(message.message_id);
          const needsTruncation = shouldTruncate(message.content);
          
          return (
            <div 
              key={message.message_id} 
              className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow"
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-gray-500">#{index + 1}</span>
                  <span 
                    className={`px-2 py-1 rounded text-xs font-medium border ${getImportanceColor(message.importance_score)}`}
                  >
                    {(message.importance_score * 100).toFixed(0)}% Important
                  </span>
                </div>
                <div className="text-xs text-gray-500 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {formatTimestamp(message.timestamp_ms)}
                </div>
              </div>
              
              <div className="mb-3">
                <div className="text-sm font-medium text-gray-800 mb-1">
                  From: {message.sender}
                </div>
                <div className="text-sm text-gray-700 leading-relaxed">
                  &ldquo;{isExpanded || !needsTruncation ? message.content : truncateContent(message.content)}&rdquo;
                  
                  {needsTruncation && (
                    <button
                      onClick={() => toggleExpanded(message.message_id)}
                      className="ml-2 inline-flex items-center text-xs text-blue-600 hover:text-blue-800 transition-colors"
                    >
                      {isExpanded ? (
                        <>
                          Show less <ChevronUp className="w-3 h-3 ml-1" />
                        </>
                      ) : (
                        <>
                          Show more <ChevronDown className="w-3 h-3 ml-1" />
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
              
              <div className="flex flex-wrap gap-1">
                {message.factors.slice(0, 4).map((factor, idx) => (
                  <span 
                    key={idx}
                    className="inline-flex items-center gap-1 px-2 py-1 bg-gray-100 text-xs text-gray-600 rounded"
                  >
                    {getFactorIcon(factor)}
                    {factor}
                  </span>
                ))}
                {message.factors.length > 4 && (
                  <span className="text-xs text-gray-400">
                    +{message.factors.length - 4} more
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {messages.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <MessageCircle className="w-12 h-12 text-gray-300 mx-auto mb-2" />
          <p>No important messages found</p>
          <p className="text-xs">Try adjusting your conversation filter</p>
        </div>
      )}
    </div>
  );
}

export function ImportantMessagesFullscreen() {
  return <ImportantMessagesAnalysis limit={50} />;
}