/**
 * Scopes dashboard metrics to known conversation participants.
 * Filtering by conversation_id alone is insufficient — sender-level rows must
 * also match conversationMetrics.json participants[] for that conversation.
 */

export interface ConversationParticipantRecord {
  conversation_id: string;
  participants: string[];
}

/** conversation_id → Set of participant display names. */
export function buildParticipantIndex(
  conversations: ConversationParticipantRecord[]
): Map<string, Set<string>> {
  const index = new Map<string, Set<string>>();
  for (const conv of conversations) {
    index.set(conv.conversation_id, new Set(conv.participants ?? []));
  }
  return index;
}

export function isKnownParticipant(
  index: Map<string, Set<string>>,
  conversationId: string,
  sender: string
): boolean {
  const participants = index.get(conversationId);
  if (!participants || participants.size === 0) return true;
  return participants.has(sender);
}

/** Union of participant names across the selected conversations. */
export function getParticipantUnion(
  conversations: ConversationParticipantRecord[],
  selectedConversationIds: string[]
): Set<string> {
  const selected = new Set(selectedConversationIds);
  const union = new Set<string>();
  for (const conv of conversations) {
    if (!selected.has(conv.conversation_id)) continue;
    for (const p of conv.participants ?? []) union.add(p);
  }
  return union;
}

export interface FilterRowsOptions<T> {
  /** Property holding the sender/participant name. Omit to skip sender check. */
  senderKey?: keyof T;
}

/**
 * Keep rows whose conversation is selected and (when senderKey set) whose sender
 * is a registered participant of that conversation.
 */
export function filterRowsBySelection<T extends { conversation_id: string }>(
  rows: T[],
  selectedConversationIds: string[],
  participantIndex: Map<string, Set<string>>,
  options?: FilterRowsOptions<T>
): T[] {
  const selected = new Set(selectedConversationIds);
  const senderKey = options?.senderKey;

  return rows.filter((row) => {
    if (!selected.has(row.conversation_id)) return false;
    if (!senderKey) return true;
    const sender = row[senderKey];
    if (typeof sender !== 'string' || !sender) return true;
    return isKnownParticipant(participantIndex, row.conversation_id, sender);
  });
}

/** Restrict messages_by_sender to registered participants only. */
export function filterMessagesBySender(
  messagesBySender: Record<string, number> | undefined,
  participants: string[]
): Record<string, number> {
  if (!messagesBySender) return {};
  const allowed = new Set(participants);
  const out: Record<string, number> = {};
  for (const [sender, count] of Object.entries(messagesBySender)) {
    if (allowed.has(sender)) out[sender] = count;
  }
  return out;
}
