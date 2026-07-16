'use client';

import React, { useState, useEffect } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface WordData {
  word: string;
  count: number;
  sender: string;
  conversation_id: string;
}

interface AggregatedWordData {
  word: string;
  total_count: number;
  unique_senders: number;
  conversations: number;
}

export function TopWordsChart() {
  const [data, setData] = useState<AggregatedWordData[]>([]);
  const [loading, setLoading] = useState(true);
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);
  const { selectedConversations, isFiltered } = useConversationFilter();

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/data/wordFrequencies.json');
        let wordData: WordData[] = await response.json();
        
        // Filter by selected conversations if filtering is active
        if (isFiltered && selectedConversations.length > 0) {
          wordData = wordData.filter(item => 
            selectedConversations.includes(item.conversation_id)
          );
        }
        
        // Aggregate word counts across conversations and senders
        const wordMap = new Map<string, {
          totalCount: number;
          senders: Set<string>;
          conversations: Set<string>;
        }>();

        wordData.forEach(item => {
          const word = item.word.toLowerCase();
          
          // Skip very common words and short words
          if (word.length < 3 || ['the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 'this', 'have', 'from', 'not', 'had', 'but', 'what', 'can', 'said', 'all', 'were', 'when', 'your', 'how', 'each', 'she', 'which', 'their', 'time', 'will', 'about', 'out', 'many', 'then', 'them', 'these', 'may', 'way', 'use', 'her', 'than', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'part', 'just', 'like', 'good', 'know', 'think', 'want', 'see', 'back', 'also', 'well', 'work', 'make', 'look', 'feel', 'going', 'really', 'yeah', 'okay', 'haha', 'lol', 'yes', 'sure', 'thanks', 'thank', 'nice', 'cool'].includes(word)) {
            return;
          }

          if (!wordMap.has(word)) {
            wordMap.set(word, {
              totalCount: 0,
              senders: new Set(),
              conversations: new Set()
            });
          }
          
          const wordStats = wordMap.get(word)!;
          wordStats.totalCount += item.count;
          wordStats.senders.add(item.sender);
          wordStats.conversations.add(item.conversation_id);
        });

        // Convert to array and sort by total count
        const aggregatedData = Array.from(wordMap.entries())
          .map(([word, stats]) => ({
            word,
            total_count: stats.totalCount,
            unique_senders: stats.senders.size,
            conversations: stats.conversations.size
          }))
          .sort((a, b) => b.total_count - a.total_count)
          .slice(0, 50); // Top 50 words for word cloud

        setData(aggregatedData);
      } catch (error) {
        console.error('Error loading word frequency data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedConversations, isFiltered]);

  if (loading) {
    return <div className="flex items-center justify-center h-64">Loading word cloud...</div>;
  }

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-gray-500">
          <div>No word frequency data available</div>
          {isFiltered && (
            <div className="text-xs mt-1">
              for selected conversations
            </div>
          )}
        </div>
      </div>
    );
  }

  // Calculate font sizes based on frequency
  const maxCount = data[0]?.total_count || 1;
  const minCount = data[data.length - 1]?.total_count || 1;
  
  const getFontSize = (count: number): number => {
    const ratio = (count - minCount) / (maxCount - minCount);
    return Math.max(12, Math.min(48, 12 + ratio * 36));
  };

  const getColor = (index: number): string => {
    const colors = [
      '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
      '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#6b7280',
      '#14b8a6', '#f472b6', '#a855f7', '#22c55e', '#eab308'
    ];
    return colors[index % colors.length];
  };

  const getOpacity = (count: number): number => {
    const ratio = (count - minCount) / (maxCount - minCount);
    return 0.6 + ratio * 0.4; // Range from 0.6 to 1.0
  };

  return (
    <div className="h-full">
      {isFiltered && (
        <div className="mb-4 text-center text-xs text-blue-600">
          Filtered across {selectedConversations.length} selected conversation{selectedConversations.length !== 1 ? 's' : ''}
        </div>
      )}
      
      <div className="relative h-full w-full overflow-hidden">
        {/* Word Cloud Container */}
        <div className="flex flex-wrap justify-center items-center h-full gap-2 leading-none">
          {data.map((word, index) => (
            <span
              key={word.word}
              className="cursor-pointer transition-all duration-200 hover:scale-110 select-none font-medium"
              style={{
                fontSize: `${getFontSize(word.total_count)}px`,
                color: getColor(index),
                opacity: getOpacity(word.total_count),
                fontWeight: hoveredWord === word.word ? 'bold' : 'normal',
                textShadow: hoveredWord === word.word ? '1px 1px 2px rgba(0,0,0,0.3)' : 'none'
              }}
              onMouseEnter={() => setHoveredWord(word.word)}
              onMouseLeave={() => setHoveredWord(null)}
              title={`"${word.word}" - Used ${word.total_count.toLocaleString()} times by ${word.unique_senders} sender${word.unique_senders !== 1 ? 's' : ''} across ${word.conversations} conversation${word.conversations !== 1 ? 's' : ''}`}
            >
              {word.word}
            </span>
          ))}
        </div>

        {/* Hover Details */}
        {hoveredWord && (
          <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white p-3 rounded-t-lg">
            {(() => {
              const wordData = data.find(w => w.word === hoveredWord);
              return wordData ? (
                <div className="text-sm">
                  <div className="font-semibold capitalize mb-1">&ldquo;{wordData.word}&rdquo;</div>
                  <div className="flex justify-between text-xs">
                    <span>Used {wordData.total_count.toLocaleString()} times</span>
                    <span>{wordData.unique_senders} sender{wordData.unique_senders !== 1 ? 's' : ''}</span>
                    <span>{wordData.conversations} conversation{wordData.conversations !== 1 ? 's' : ''}</span>
                  </div>
                </div>
              ) : null;
            })()}
          </div>
        )}
      </div>
    </div>
  );
} 