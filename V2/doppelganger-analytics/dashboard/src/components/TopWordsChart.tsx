'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useParticipantScope } from '@/hooks/useParticipantScope';
import { AnchoredPopover } from '@/components/ui/AnchoredPopover';

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

interface WordExampleMessage {
  text: string;
  sender: string;
  conversation_id: string;
}

interface WordExampleEntry {
  word: string;
  examples: WordExampleMessage[];
}

const EXTRA_STOP_WORDS = new Set([
  'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they', 'this', 'have', 'from',
  'not', 'had', 'but', 'what', 'can', 'said', 'all', 'were', 'when', 'your', 'how', 'each', 'she',
  'which', 'their', 'time', 'will', 'about', 'out', 'many', 'then', 'them', 'these', 'may', 'way',
  'use', 'her', 'than', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get',
  'come', 'made', 'part', 'just', 'like', 'good', 'know', 'think', 'want', 'see', 'back', 'also',
  'well', 'work', 'make', 'look', 'feel', 'going', 'really', 'yeah', 'okay', 'haha', 'lol', 'yes',
  'sure', 'thanks', 'thank', 'nice', 'cool',
  // Platform notification vocabulary (Messenger / Instagram system messages)
  'sent', 'photo', 'photos', 'video', 'videos', 'reacted', 'reaction', 'reactions',
  'message', 'messages', 'attachment', 'attachments', 'liked', 'shared', 'share',
  'voice', 'removed', 'unsent', 'sticker', 'stickers', 'gamepigeon', 'move',
  'named', 'changed', 'added', 'missed', 'started', 'ended',
]);

function aggregateWords(
  wordData: WordData[],
  limit: number
): AggregatedWordData[] {
  const wordMap = new Map<string, {
    totalCount: number;
    senders: Set<string>;
    conversations: Set<string>;
  }>();

  for (const item of wordData) {
    const word = item.word.toLowerCase();
    if (word.length < 3 || EXTRA_STOP_WORDS.has(word)) continue;

    let stats = wordMap.get(word);
    if (!stats) {
      stats = { totalCount: 0, senders: new Set(), conversations: new Set() };
      wordMap.set(word, stats);
    }
    stats.totalCount += item.count;
    stats.senders.add(item.sender);
    stats.conversations.add(item.conversation_id);
  }

  return Array.from(wordMap.entries())
    .map(([word, stats]) => ({
      word,
      total_count: stats.totalCount,
      unique_senders: stats.senders.size,
      conversations: stats.conversations.size,
    }))
    .sort((a, b) => b.total_count - a.total_count)
    .slice(0, limit);
}

function pickRandomExample(
  examples: WordExampleMessage[],
  scopeIds: string[],
  participantUnion: Set<string>
): WordExampleMessage | null {
  const allowed = new Set(scopeIds);
  const pool = examples.filter(
    (e) => allowed.has(e.conversation_id) && participantUnion.has(e.sender)
  );
  if (pool.length === 0) return null;
  return pool[Math.floor(Math.random() * pool.length)];
}

interface TopWordsChartProps {
  /** Cloud in card; list shows ranked top-200 for fullscreen. */
  variant?: 'cloud' | 'list';
}

export function TopWordsChart({ variant = 'cloud' }: TopWordsChartProps) {
  const [data, setData] = useState<AggregatedWordData[]>([]);
  const [examplesByWord, setExamplesByWord] = useState<Map<string, WordExampleMessage[]>>(new Map());
  const [loading, setLoading] = useState(true);
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);
  const [hoverExample, setHoverExample] = useState<WordExampleMessage | null>(null);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const { filterScopedRows, scopeConversationIds, participantUnion } = useParticipantScope();

  const limit = variant === 'list' ? 200 : 50;

  useEffect(() => {
    const loadData = async () => {
      try {
        const [freqRes, exRes] = await Promise.all([
          fetch('/data/wordFrequencies.json'),
          fetch('/data/wordExamples.json'),
        ]);
        let wordData: WordData[] = await freqRes.json();
        wordData = filterScopedRows(wordData, { senderKey: 'sender' });

        setData(aggregateWords(wordData, limit));

        if (exRes.ok) {
          const examples: WordExampleEntry[] = await exRes.json();
          const map = new Map<string, WordExampleMessage[]>();
          for (const entry of examples) {
            map.set(entry.word.toLowerCase(), filterScopedRows(entry.examples, { senderKey: 'sender' }));
          }
          setExamplesByWord(map);
        }
      } catch (error) {
        console.error('Error loading word frequency data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [filterScopedRows, scopeConversationIds, limit]);

  const onWordEnter = useCallback(
    (word: string) => {
      setHoveredWord(word);
      const examples = examplesByWord.get(word.toLowerCase()) ?? [];
      setHoverExample(pickRandomExample(examples, scopeConversationIds, participantUnion));
    },
    [examplesByWord, scopeConversationIds, participantUnion]
  );

  const onWordLeave = useCallback(() => {
    setHoveredWord(null);
    setHoverExample(null);
  }, []);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        Loading {variant === 'list' ? 'top words' : 'word cloud'}...
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center text-gray-500">
          <div>No word frequency data available</div>
        </div>
      </div>
    );
  }

  const maxCount = data[0]?.total_count || 1;
  const minCount = data[data.length - 1]?.total_count || 1;

  const getFontSize = (count: number): number => {
    const ratio = maxCount === minCount ? 1 : (count - minCount) / (maxCount - minCount);
    return Math.max(12, Math.min(48, 12 + ratio * 36));
  };

  const colors = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#6b7280',
  ];

  if (variant === 'list') {
    return (
      <div className="flex h-full flex-col gap-2">
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="space-y-1 pr-1">
            {data.map((word, index) => {
              const barPct = (word.total_count / maxCount) * 100;
              return (
                <div
                  key={word.word}
                  className="group flex items-center gap-2 rounded-md px-1 py-0.5 hover:bg-gray-50"
                  onMouseEnter={() => onWordEnter(word.word)}
                  onMouseLeave={onWordLeave}
                >
                  <span className="w-6 shrink-0 text-right text-xs text-gray-400">{index + 1}</span>
                  <span
                    ref={hoveredWord === word.word ? anchorRef : undefined}
                    className="w-28 shrink-0 truncate text-sm font-medium capitalize text-gray-900"
                  >
                    {word.word}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div
                      className="h-2 rounded-full bg-purple-200"
                      style={{ width: `${Math.max(barPct, 4)}%` }}
                    />
                  </div>
                  <span className="w-14 shrink-0 text-right text-xs text-gray-500">
                    {word.total_count.toLocaleString()}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <AnchoredPopover
          open={!!hoveredWord && !!hoverExample}
          onOpenChange={(open) => {
            if (!open) onWordLeave();
          }}
          anchorRef={anchorRef}
          className="max-w-sm p-3"
        >
          {hoverExample && hoveredWord && (
            <div className="text-sm">
              <div className="mb-1 font-semibold capitalize text-gray-900">&ldquo;{hoveredWord}&rdquo;</div>
              <p className="text-gray-700">&ldquo;{hoverExample.text}&rdquo;</p>
              <p className="mt-1 text-xs text-gray-500">— {hoverExample.sender}</p>
            </div>
          )}
        </AnchoredPopover>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <div className="flex h-full w-full flex-wrap items-center justify-center gap-2 overflow-hidden leading-none">
        {data.map((word, index) => (
          <span
            key={word.word}
            ref={hoveredWord === word.word ? anchorRef : undefined}
            className="cursor-pointer select-none font-medium transition-all duration-200 hover:scale-110"
            style={{
              fontSize: `${getFontSize(word.total_count)}px`,
              color: colors[index % colors.length],
              opacity: 0.6 + ((word.total_count - minCount) / (maxCount - minCount || 1)) * 0.4,
              fontWeight: hoveredWord === word.word ? 'bold' : 'normal',
            }}
            onMouseEnter={() => onWordEnter(word.word)}
            onMouseLeave={onWordLeave}
          >
            {word.word}
          </span>
        ))}
      </div>

      <AnchoredPopover
        open={!!hoveredWord && !!hoverExample}
        onOpenChange={(open) => {
          if (!open) onWordLeave();
        }}
        anchorRef={anchorRef}
        className="max-w-sm p-3"
      >
        {hoverExample && hoveredWord && (
          <div className="text-sm">
            <div className="mb-1 font-semibold capitalize text-gray-900">&ldquo;{hoveredWord}&rdquo;</div>
            <p className="text-gray-700">&ldquo;{hoverExample.text}&rdquo;</p>
            <p className="mt-1 text-xs text-gray-500">— {hoverExample.sender}</p>
          </div>
        )}
      </AnchoredPopover>
    </div>
  );
}

export function TopWordsFullscreen() {
  return <TopWordsChart variant="list" />;
}
