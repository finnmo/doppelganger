'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode, useMemo } from 'react';
import { resolveSource, sourceLabel } from '@/lib/platforms';

export interface ConversationInfo {
  conversation_id: string;
  source: string;
  source_label: string;
  participants: string[];
  first_message_ms: number;
  last_message_ms: number;
  total_messages: number;
  duration_ms: number;
  turns: number;
  avg_response_time: number;
}

interface RawConversationInfo {
  conversation_id: string;
  source?: string;
  source_label?: string;
  participants?: string[];
  first_message_ms?: number;
  last_message_ms?: number;
  total_messages?: number;
  duration_ms?: number;
  turns?: number;
  avg_response_time?: number;
}

export interface PlatformSummary {
  source: string;
  label: string;
  conversations: number;
  messages: number;
}

interface ConversationContextType {
  conversations: ConversationInfo[];
  selectedConversations: string[];
  setSelectedConversations: (conversations: string[]) => void;
  isLoading: boolean;
  isFiltered: boolean;
  platforms: PlatformSummary[];
}

const ConversationContext = createContext<ConversationContextType | undefined>(undefined);

export function ConversationProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<ConversationInfo[]>([]);
  const [platforms, setPlatforms] = useState<PlatformSummary[]>([]);
  const [selectedConversations, setSelectedConversations] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadConversations = async () => {
      try {
        const response = await fetch('/data/conversationMetrics.json');
        const data: {
          conversations: RawConversationInfo[];
          summary?: { platforms?: PlatformSummary[] };
        } = await response.json();

        const conversationList: ConversationInfo[] = data.conversations.map((conv) => {
          const source = resolveSource(conv.conversation_id, conv.source);
          return {
            conversation_id: conv.conversation_id,
            source,
            source_label: conv.source_label || sourceLabel(source),
            participants: conv.participants || [],
            first_message_ms: conv.first_message_ms || 0,
            last_message_ms: conv.last_message_ms || 0,
            total_messages: conv.total_messages || 0,
            duration_ms: conv.duration_ms || 0,
            turns: conv.turns || 0,
            avg_response_time: conv.avg_response_time || 0,
          };
        });

        setConversations(conversationList);
        setSelectedConversations(conversationList.map((c) => c.conversation_id));

        if (data.summary?.platforms?.length) {
          setPlatforms(data.summary.platforms);
        } else {
          // Derive from conversations when regenerating older metrics files.
          const map = new Map<string, PlatformSummary>();
          for (const c of conversationList) {
            const entry = map.get(c.source) ?? {
              source: c.source,
              label: c.source_label,
              conversations: 0,
              messages: 0,
            };
            entry.conversations += 1;
            entry.messages += c.total_messages;
            map.set(c.source, entry);
          }
          setPlatforms([...map.values()].sort((a, b) => b.messages - a.messages));
        }
      } catch (error) {
        console.error('Error loading conversations:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadConversations();
  }, []);

  const isFiltered = selectedConversations.length < conversations.length;

  const value = useMemo(
    () => ({
      conversations,
      selectedConversations,
      setSelectedConversations,
      isLoading,
      isFiltered,
      platforms,
    }),
    [conversations, selectedConversations, isLoading, isFiltered, platforms]
  );

  return (
    <ConversationContext.Provider value={value}>
      {children}
    </ConversationContext.Provider>
  );
}

export function useConversationFilter() {
  const context = useContext(ConversationContext);
  if (context === undefined) {
    throw new Error('useConversationFilter must be used within a ConversationProvider');
  }
  return context;
}
