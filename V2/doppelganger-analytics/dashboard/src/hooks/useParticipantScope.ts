'use client';

import { useCallback, useMemo } from 'react';
import { useConversationFilter } from '@/contexts/ConversationContext';
import {
  buildParticipantIndex,
  filterRowsBySelection,
  getParticipantUnion,
  type FilterRowsOptions,
} from '@/lib/participantFilter';

/**
 * Conversation + participant scoping for dashboard charts.
 * scopeConversationIds is the active selection (all conversations when unfiltered).
 */
export function useParticipantScope() {
  const { conversations, selectedConversations, isFiltered } = useConversationFilter();

  const participantIndex = useMemo(
    () => buildParticipantIndex(conversations),
    [conversations]
  );

  const scopeConversationIds = useMemo(() => {
    if (isFiltered && selectedConversations.length > 0) return selectedConversations;
    return conversations.map((c) => c.conversation_id);
  }, [conversations, selectedConversations, isFiltered]);

  const participantUnion = useMemo(
    () => getParticipantUnion(conversations, scopeConversationIds),
    [conversations, scopeConversationIds]
  );

  const filterScopedRows = useCallback(
    <T extends { conversation_id: string }>(
      rows: T[],
      options?: FilterRowsOptions<T>
    ) => filterRowsBySelection(rows, scopeConversationIds, participantIndex, options),
    [scopeConversationIds, participantIndex]
  );

  return {
    conversations,
    isFiltered,
    participantIndex,
    participantUnion,
    scopeConversationIds,
    filterScopedRows,
  };
}
