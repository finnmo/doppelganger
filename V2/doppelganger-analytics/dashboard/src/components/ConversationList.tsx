'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import { Users, MessageCircle, Clock, ArrowRight, EyeOff, Eye } from 'lucide-react';
import { ThemeSelector } from '@/components/ThemeSelector';
import { ApiKeySettingsButton } from '@/components/ApiKeySettings';
import { PrivacySettingsButton } from '@/components/PrivacySettings';
import { PersonaChatButton } from '@/components/PersonaChatButton';
import { DataFreshnessBanner } from '@/components/DataFreshnessBanner';
import { PlatformBadge } from '@/components/PlatformBadge';
import { parseConversationId, platformStyles, sourceLabel } from '@/lib/platforms';
import { TOOLBAR_ROW } from '@/lib/layout';

const HIDDEN_CONVERSATIONS_KEY = 'hiddenConversationIds';

function readHiddenConversationIds(): string[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(HIDDEN_CONVERSATIONS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((value): value is string => typeof value === 'string');
  } catch {
    return [];
  }
}

function writeHiddenConversationIds(ids: string[]) {
  window.localStorage.setItem(HIDDEN_CONVERSATIONS_KEY, JSON.stringify(ids));
}

type Conversation = ReturnType<typeof useConversationFilter>['conversations'][number];

interface ConversationCardProps {
  conversation: Conversation;
  onClick: (conversationId: string) => void;
  hidden: boolean;
  onToggleHidden: (conversationId: string) => void;
}

function ConversationCard({ conversation, onClick, hidden, onToggleHidden }: ConversationCardProps) {
  const getDisplayName = (conv: Conversation) => {
    if (conv.participants && conv.participants.length > 0) {
      if (conv.participants.length === 1) {
        return conv.participants[0];
      } else if (conv.participants.length === 2) {
        return conv.participants.join(' & ');
      } else {
        return `${conv.participants[0]} + ${conv.participants.length - 1} others`;
      }
    }
    return parseConversationId(conv.conversation_id).rawId;
  };

  const getConversationType = (conv: Conversation) => {
    if (!conv.participants || conv.participants.length === 0) return 'Unknown';
    if (conv.participants.length === 1) return 'Self';
    if (conv.participants.length === 2) return 'Direct Message';
    return 'Group Chat';
  };

  const formatDuration = (durationMs: number) => {
    const days = Math.floor(durationMs / (1000 * 60 * 60 * 24));
    if (days > 365) {
      const years = Math.floor(days / 365);
      return `${years} year${years > 1 ? 's' : ''}`;
    }
    if (days > 30) {
      const months = Math.floor(days / 30);
      return `${months} month${months > 1 ? 's' : ''}`;
    }
    if (days > 0) {
      return `${days} day${days > 1 ? 's' : ''}`;
    }
    const hours = Math.floor(durationMs / (1000 * 60 * 60));
    return `${hours} hour${hours > 1 ? 's' : ''}`;
  };

  const accent = platformStyles(conversation.source);

  return (
    <div
      className={`bg-white rounded-lg shadow-sm border p-6 cursor-pointer hover:shadow-md transition-all duration-200 relative overflow-hidden ${
        hidden ? 'border-dashed border-gray-300 opacity-80' : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={() => onClick(conversation.conversation_id)}
    >
      <div className={`absolute left-0 top-0 bottom-0 w-1 ${accent.dot}`} aria-hidden />

      <div className="flex items-start justify-between pl-1">
        <div className="flex-1 min-w-0">
          <div className="flex items-center flex-wrap gap-2 mb-2">
            <PlatformBadge source={conversation.source} />
            <span className="inline-flex items-center gap-1 text-xs text-gray-500 uppercase tracking-wide">
              <Users className="w-3.5 h-3.5" />
              {getConversationType(conversation)}
            </span>
          </div>

          <h3 className="text-lg font-semibold text-gray-900 truncate mb-2">
            {getDisplayName(conversation)}
          </h3>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-gray-600">
            <div className="flex items-center space-x-1">
              <MessageCircle className="w-4 h-4 shrink-0" />
              <span>{conversation.total_messages?.toLocaleString() || 0} messages</span>
            </div>

            <div className="flex items-center space-x-1">
              <Clock className="w-4 h-4 shrink-0" />
              <span>{formatDuration(conversation.duration_ms || 0)}</span>
            </div>
          </div>

          {conversation.participants && conversation.participants.length > 2 && (
            <div className="mt-3">
              <div className="text-xs text-gray-500">Participants:</div>
              <div className="text-sm text-gray-700 mt-1">
                {conversation.participants.slice(0, 3).join(', ')}
                {conversation.participants.length > 3 &&
                  ` and ${conversation.participants.length - 3} more`}
              </div>
            </div>
          )}
        </div>

        <ArrowRight className="w-5 h-5 text-gray-400 shrink-0 ml-2" />
      </div>
      <button
        type="button"
        className="mt-4 inline-flex items-center gap-1 rounded-md border border-gray-200 bg-white px-2 py-1 text-xs text-gray-600 hover:bg-gray-50"
        onClick={(event) => {
          event.stopPropagation();
          onToggleHidden(conversation.conversation_id);
        }}
      >
        {hidden ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
        {hidden ? 'Unhide' : 'Hide'}
      </button>
    </div>
  );
}

interface ConversationListProps {
  onConversationSelect: (conversationId: string) => void;
}

export function ConversationList({ onConversationSelect }: ConversationListProps) {
  const { conversations, isLoading, platforms } = useConversationFilter();
  const [filteredConversations, setFilteredConversations] = useState<Conversation[]>([]);
  const [hiddenConversations, setHiddenConversations] = useState<string[]>(readHiddenConversationIds);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<'messages' | 'recent' | 'duration'>('messages');
  const [platformFilter, setPlatformFilter] = useState<string | 'all'>('all');
  const [showHidden, setShowHidden] = useState(false);
  const hiddenConversationSet = useMemo(() => new Set(hiddenConversations), [hiddenConversations]);

  const availablePlatforms = useMemo(() => {
    if (platforms.length > 0) return platforms;
    const map = new Map<string, { source: string; label: string; conversations: number; messages: number }>();
    for (const c of conversations) {
      const entry = map.get(c.source) ?? {
        source: c.source,
        label: c.source_label || sourceLabel(c.source),
        conversations: 0,
        messages: 0,
      };
      entry.conversations += 1;
      entry.messages += c.total_messages || 0;
      map.set(c.source, entry);
    }
    return [...map.values()].sort((a, b) => b.messages - a.messages);
  }, [platforms, conversations]);

  const showPlatformFilters = availablePlatforms.length > 1;

  useEffect(() => {
    if (!conversations) return;

    let filtered = conversations.filter((conv) => {
      if ((conv.total_messages || 0) <= 5) return false;
      if (!showHidden && hiddenConversationSet.has(conv.conversation_id)) return false;
      if (platformFilter !== 'all' && conv.source !== platformFilter) return false;

      if (!searchTerm) return true;
      const searchLower = searchTerm.toLowerCase();
      if (conv.participants?.some((p) => p.toLowerCase().includes(searchLower))) {
        return true;
      }
      if (conv.source_label?.toLowerCase().includes(searchLower)) return true;
      return conv.conversation_id.toLowerCase().includes(searchLower);
    });

    filtered = filtered.sort((a, b) => {
      switch (sortBy) {
        case 'messages':
          return (b.total_messages || 0) - (a.total_messages || 0);
        case 'duration':
          return (b.duration_ms || 0) - (a.duration_ms || 0);
        case 'recent':
          return (b.last_message_ms || 0) - (a.last_message_ms || 0);
        default:
          return 0;
      }
    });

    setFilteredConversations(filtered);
  }, [conversations, searchTerm, sortBy, platformFilter, hiddenConversationSet, showHidden]);

  const toggleConversationHidden = (conversationId: string) => {
    setHiddenConversations((current) => {
      const next = current.includes(conversationId)
        ? current.filter((id) => id !== conversationId)
        : [...current, conversationId];
      writeHiddenConversationIds(next);
      return next;
    });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto" />
          <p className="mt-2 text-gray-600">Loading conversations...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      <div className="mb-6 flex flex-col gap-4 sm:mb-8 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h1 className="mb-2 text-2xl font-bold text-gray-900 sm:text-3xl">Your Conversations</h1>
          <p className="text-sm text-gray-600 sm:text-base">
            {showPlatformFilters
              ? 'Conversations from multiple messaging platforms — filter by source or open one for analytics.'
              : 'Select a conversation to view detailed analytics and insights'}
          </p>
        </div>
        <div className={`${TOOLBAR_ROW} shrink-0`}>
          <PersonaChatButton />
          <PrivacySettingsButton />
          <ApiKeySettingsButton />
          <ThemeSelector />
        </div>
      </div>

      <DataFreshnessBanner />

      {showPlatformFilters && (
        <div className="mb-6 flex flex-wrap gap-2 items-center">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide mr-1">
            Source
          </span>
          <button
            type="button"
            onClick={() => setPlatformFilter('all')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              platformFilter === 'all'
                ? 'bg-gray-900 text-white'
                : 'bg-white text-gray-700 border border-gray-200 hover:border-gray-300'
            }`}
          >
            All platforms
          </button>
          {availablePlatforms.map((p) => {
            const styles = platformStyles(p.source);
            const active = platformFilter === p.source;
            return (
              <button
                key={p.source}
                type="button"
                onClick={() => setPlatformFilter(p.source)}
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ring-1 ring-inset ${
                  active
                    ? `${styles.bg} ${styles.text} ${styles.ring}`
                    : 'bg-white text-gray-700 ring-gray-200 hover:ring-gray-300'
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${styles.dot}`} aria-hidden />
                {p.label}
                <span className="text-xs opacity-70 tabular-nums">
                  {p.conversations}
                </span>
              </button>
            );
          })}
        </div>
      )}

      <div className="mb-6 flex flex-col sm:flex-row gap-4">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search by participant or platform..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="flex space-x-2">
          {hiddenConversations.length > 0 && (
            <button
              type="button"
              onClick={() => setShowHidden((current) => !current)}
              className={`px-4 py-2 rounded-lg border transition-colors ${
                showHidden
                  ? 'bg-gray-900 text-white border-gray-900'
                  : 'bg-white text-gray-700 border-gray-300 hover:border-gray-400'
              }`}
            >
              {showHidden ? 'Hide hidden rows' : `Show hidden (${hiddenConversations.length})`}
            </button>
          )}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'messages' | 'recent' | 'duration')}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="messages">Sort by Messages</option>
            <option value="duration">Sort by Duration</option>
            <option value="recent">Sort by Recent</option>
          </select>
        </div>
      </div>

      <div
        className={`mb-6 grid grid-cols-2 gap-3 sm:mb-8 sm:gap-4 ${
          showPlatformFilters
            ? 'md:grid-cols-4'
            : 'md:grid-cols-3'
        }`}
      >
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="text-blue-600 text-sm font-medium">Conversations</div>
          <div className="text-2xl font-bold text-blue-900">{filteredConversations.length}</div>
        </div>
        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-green-600 text-sm font-medium">Messages</div>
          <div className="text-2xl font-bold text-green-900">
            {filteredConversations
              .reduce((sum, conv) => sum + (conv.total_messages || 0), 0)
              .toLocaleString()}
          </div>
        </div>
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="text-purple-600 text-sm font-medium">Avg per conversation</div>
          <div className="text-2xl font-bold text-purple-900">
            {Math.round(
              filteredConversations.reduce((sum, conv) => sum + (conv.total_messages || 0), 0) /
                filteredConversations.length || 0
            ).toLocaleString()}
          </div>
        </div>
        {showPlatformFilters && (
          <div className="bg-slate-50 rounded-lg p-4">
            <div className="text-slate-600 text-sm font-medium">Platforms</div>
            <div className="text-2xl font-bold text-slate-900">{availablePlatforms.length}</div>
            <div className="mt-1 text-xs text-slate-500 truncate">
              {availablePlatforms.map((p) => p.label).join(' · ')}
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 sm:gap-6 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
        {filteredConversations.map((conversation) => (
          <ConversationCard
            key={conversation.conversation_id}
            conversation={conversation}
            onClick={onConversationSelect}
            hidden={hiddenConversationSet.has(conversation.conversation_id)}
            onToggleHidden={toggleConversationHidden}
          />
        ))}
      </div>

      {filteredConversations.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-500">
            {searchTerm || platformFilter !== 'all'
              ? 'No conversations match your filters.'
              : hiddenConversations.length > 0 && !showHidden
                ? 'All visible conversations are hidden. Use "Show hidden" to review and unhide them.'
                : 'No conversations found.'}
          </div>
        </div>
      )}
    </div>
  );
}
