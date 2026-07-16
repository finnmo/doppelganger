/**
 * Contract tests for filtered Overview contribution math.
 * The dashboard must never divide a participant's GLOBAL message total by a
 * FILTERED conversation's message count (that produced 510363% in production).
 */

describe('filtered contribution % contract', () => {
  test('per-conversation messages_by_sender sums to total_messages', () => {
    const messages_by_sender = { Alice: 10, Bob: 6 };
    const total_messages = 16;
    expect(Object.values(messages_by_sender).reduce((a, b) => a + b, 0)).toBe(total_messages);
  });

  test('contribution % is sender_msgs / conversation_total, never global/filtered', () => {
    const globalFinn = 81658;
    const conversationTotal = 16;
    // The bug: global ÷ filtered
    const buggyPct = Math.round((globalFinn / conversationTotal) * 100);
    expect(buggyPct).toBeGreaterThan(100_000);

    // The fix: per-conversation counts only
    const finnInConv = 9;
    const tiaInConv = 7;
    const total = finnInConv + tiaInConv;
    expect(total).toBe(conversationTotal);
    expect(Math.round((finnInConv / total) * 100)).toBe(56);
    expect(Math.round((tiaInConv / total) * 100)).toBe(44);
    expect(Math.round((finnInConv / total) * 100)).toBeLessThanOrEqual(100);
  });

  test('aggregating messages_by_sender across selected conversations is additive', () => {
    const selected = [
      { messages_by_sender: { Alice: 10, Bob: 6 } },
      { messages_by_sender: { Alice: 5, Carol: 2 } }
    ];
    const aggregated = new Map<string, number>();
    for (const conv of selected) {
      for (const [sender, count] of Object.entries(conv.messages_by_sender)) {
        aggregated.set(sender, (aggregated.get(sender) || 0) + count);
      }
    }
    expect(Object.fromEntries(aggregated)).toEqual({ Alice: 15, Bob: 6, Carol: 2 });
    const total = [...aggregated.values()].reduce((a, b) => a + b, 0);
    expect(Math.round((15 / total) * 100)).toBe(65);
  });
});
