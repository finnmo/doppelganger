import type { PersonaFewShotExample, PersonaProfile } from './personaProfiles';
import type { MemorySnippet } from './personaRetrieval';
import { tokenizeForRetrieval } from './personaRetrieval';

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

const MAX_CONTEXT_CHARS = 320;
const MAX_REPLY_CHARS = 450;
const DEFAULT_MAX_EXAMPLES = 12;
const MAX_MEMORIES = 8;

const ULTRA_SHORT_ACK =
  /^(ok|k|kk|okay|lol|lmao|haha|hahah|yeah|yep|ye|nah|no|yes|sure|cool|nice|same|true|bet|omg| rip|ahaha|aha)\.?\!?$/i;

function truncate(text: string, max: number): string {
  const t = text.trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

/** Prefer engaged replies over one-word dead-ends when the live message has substance. */
function exampleSubstanceBonus(query: string, example: PersonaFewShotExample): number {
  const qWords = wordCount(query);
  const rWords = wordCount(example.reply);
  if (qWords <= 2) return 0;
  if (ULTRA_SHORT_ACK.test(example.reply.trim())) return -0.8;
  if (rWords >= 8) return 0.45;
  if (rWords >= 4) return 0.2;
  return -0.15;
}

function engagementGuidance(profile: PersonaProfile): string {
  const withYouAvg = profile.relationshipCard?.avgWordsWithYou;
  const avg = withYouAvg ?? profile.vocabulary?.avgWordsPerMessage ?? 6;
  const low = Math.max(5, Math.round(avg));
  const high = Math.max(low + 8, Math.round(avg * 4));
  const withPerson = profile.relationshipCard?.withPerson;
  return [
    withPerson
      ? `Length target with ${withPerson}: when there is something to say, write about ${low}–${high} words (or 1–3 short text bubbles) — not a clipped “ok / lol / idk” shutdown.`
      : `Length target: when there is something to say, write about ${low}–${high} words (or 1–3 short text bubbles) — not a clipped “ok / lol / idk” shutdown.`,
    `They do send short acks sometimes, but only when the other person sent something that warrants a quick ack. If the other person is chatting, sharing, asking, or joking — engage: react, add a thought, ask something back, or continue the thread.`,
  ].join(' ');
}

function examplePool(profile: PersonaProfile): PersonaFewShotExample[] {
  const withYou = (profile.withYouFewShotExamples ?? []).map((e) => ({ ...e, withYou: true as const }));
  const general = profile.fewShotExamples ?? [];
  // Dedupe by context+reply; prefer withYou copies
  const seen = new Set<string>();
  const out: PersonaFewShotExample[] = [];
  for (const e of [...withYou, ...general]) {
    const key = `${e.context}::${e.reply}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(e);
  }
  return out;
}

/** Token overlap score between query and an example's context (+ light reply signal). */
export function scoreExampleSimilarity(query: string, example: PersonaFewShotExample): number {
  const qTokens = new Set(tokenizeForRetrieval(query));
  if (qTokens.size === 0) return 0;

  const contextTokens = tokenizeForRetrieval(example.context);
  const replyTokens = tokenizeForRetrieval(example.reply);
  if (contextTokens.length === 0 && replyTokens.length === 0) return 0;

  let contextHits = 0;
  const seenCtx = new Set<string>();
  for (const t of contextTokens) {
    if (qTokens.has(t) && !seenCtx.has(t)) {
      contextHits += 1;
      seenCtx.add(t);
    }
  }

  let replyHits = 0;
  const seenReply = new Set<string>();
  for (const t of replyTokens) {
    if (qTokens.has(t) && !seenReply.has(t)) {
      replyHits += 1;
      seenReply.add(t);
    }
  }

  const contextScore =
    contextHits === 0 ? 0 : contextHits + contextHits / Math.sqrt(Math.max(contextTokens.length, 1));
  const replyScore = replyHits * 0.25;
  return contextScore + replyScore;
}

/**
 * Pick few-shot pairs most similar to the current user message.
 * Strongly prefers with-you / open-conversation register.
 */
export function selectExamples(
  profile: PersonaProfile,
  options: {
    conversationId?: string;
    maxExamples: number;
    query?: string;
  }
): PersonaFewShotExample[] {
  const all = examplePool(profile);
  if (all.length === 0) return [];

  const maxExamples = options.maxExamples;
  const query = options.query?.trim() ?? '';
  const conversationId = options.conversationId;
  const withYouIds = new Set(profile.relationshipCard?.sharedConversationIds ?? []);

  const withYouFirst = () => {
    const withYou = all.filter((e) => e.withYou || withYouIds.has(e.conversationId));
    const rest = all.filter((e) => !(e.withYou || withYouIds.has(e.conversationId)));
    if (conversationId) {
      const scoped = all.filter((e) => e.conversationId === conversationId);
      return [...scoped, ...withYou, ...rest].slice(0, maxExamples);
    }
    return [...withYou, ...rest].slice(0, maxExamples);
  };

  if (!query) {
    return withYouFirst();
  }

  const scored = all.map((example, index) => {
    let score = scoreExampleSimilarity(query, example);
    if (conversationId && example.conversationId === conversationId) {
      score += 0.55;
    }
    if (example.withYou || withYouIds.has(example.conversationId)) {
      score += 0.7;
    }
    score += exampleSubstanceBonus(query, example);
    return { example, score, index };
  });

  scored.sort((a, b) => b.score - a.score || a.index - b.index);

  const best = scored[0]?.score ?? 0;
  if (best < 0.15) {
    return withYouFirst();
  }

  const picked: PersonaFewShotExample[] = [];
  const perConv = new Map<string, number>();
  const MAX_PER_CONV = Math.max(3, Math.ceil(maxExamples / 3));

  for (const row of scored) {
    if (picked.length >= maxExamples) break;
    const cid = row.example.conversationId;
    const n = perConv.get(cid) ?? 0;
    if (n >= MAX_PER_CONV && picked.length < maxExamples - 2) {
      continue;
    }
    picked.push(row.example);
    perConv.set(cid, n + 1);
  }

  if (picked.length < maxExamples) {
    const used = new Set(picked.map((p) => `${p.conversationId}::${p.context}::${p.reply}`));
    for (const row of scored) {
      if (picked.length >= maxExamples) break;
      const key = `${row.example.conversationId}::${row.example.context}::${row.example.reply}`;
      if (used.has(key)) continue;
      picked.push(row.example);
      used.add(key);
    }
  }

  return picked;
}

/**
 * Build Anthropic Messages API payload pieces for persona simulation.
 * System is separate; messages are user/assistant only (few-shot + live/thread chat).
 */
export function buildAnthropicPersonaRequest(
  profile: PersonaProfile,
  conversation: ChatTurn[],
  options?: {
    conversationId?: string;
    maxExamples?: number;
    memories?: MemorySnippet[];
    /** Override query used for few-shot ranking (defaults to last user turn). */
    query?: string;
  }
): { system: string; messages: ChatTurn[]; memoryCount: number; exampleCount: number } {
  const maxExamples = options?.maxExamples ?? DEFAULT_MAX_EXAMPLES;
  const lastUser = [...conversation].reverse().find((t) => t.role === 'user');
  const query = options?.query ?? lastUser?.content ?? '';
  const examples = selectExamples(profile, {
    conversationId: options?.conversationId,
    maxExamples,
    query,
  });
  const memories = (options?.memories ?? []).slice(0, MAX_MEMORIES);

  const systemParts = [
    `You are ${profile.sender} texting in a private chat. Stay fully in character.`,
    `Sound like them: their vocabulary, rhythm, punctuation, slang, humor, and how warm/blunt they are with this person.`,
    profile.styleSummary,
    engagementGuidance(profile),
  ];

  if (profile.relationshipCard?.registerSummary) {
    systemParts.push(
      '',
      'Register with this person (highest priority for tone/openers/questions):',
      profile.relationshipCard.registerSummary
    );
  } else if (profile.relationshipCard?.summary) {
    systemParts.push('', 'Relationship facts (stay consistent with these):', profile.relationshipCard.summary);
  }

  if (
    profile.relationshipCard?.summary &&
    profile.relationshipCard?.registerSummary &&
    profile.relationshipCard.summary !== profile.relationshipCard.registerSummary
  ) {
    systemParts.push('', 'Relationship facts:', profile.relationshipCard.summary);
  }

  const activeVoice = options?.conversationId
    ? profile.conversationVoices?.find((v) => v.conversationId === options.conversationId)
    : undefined;
  if (activeVoice?.styleSummary) {
    systemParts.push(
      '',
      'Voice for this specific chat (override global averages when they conflict):',
      activeVoice.styleSummary
    );
  }

  if (profile.bubbleHabits?.styleSummary) {
    systemParts.push('', 'Multi-bubble habit:', profile.bubbleHabits.styleSummary);
    if (profile.bubbleHabits.sampleTurns?.length) {
      systemParts.push(
        'Examples of their real multi-bubble turns (each line was a separate text):',
        ...profile.bubbleHabits.sampleTurns.slice(0, 3).map((turn, i) => {
          return `${i + 1}. ${turn.map((b) => `“${b}”`).join(' <<<BUBBLE>>> ')}`;
        })
      );
    }
  }

  if (profile.turnTakingHabits?.styleSummary) {
    systemParts.push('', 'Turn-taking:', profile.turnTakingHabits.styleSummary);
  }

  if (profile.sharedTimeline?.summary) {
    systemParts.push('', profile.sharedTimeline.summary);
  }

  systemParts.push(
    '',
    'Rules:',
    '- Keep the conversation alive the way they would — react, riff, ask, share. Do not sound avoidant or eager to end the chat.',
    '- Match the energy of the latest message. A joke gets a joke/reaction; a question gets a real answer; a story gets a real response.',
    '- Prefer a natural texting reply over a one-liner closer. One-word replies only when that is clearly how they would respond to this exact message.',
    '- When sending multiple short texts, separate each bubble with exactly <<<BUBBLE>>> (no other labels).',
    '- Use remembered facts when relevant. If something is unknown, answer in a normal human way without inventing detailed false memories — do not dodge with “idk” every time.',
    '- Never mention that you are an AI or that you are role-playing.',
    '- Reply with only the message text they would send — no quotes, labels, or stage directions (except <<<BUBBLE>>> between bubbles).',
  );

  if (memories.length > 0) {
    systemParts.push(
      '',
      'Relevant memories from their real past messages (use for facts/context):',
      ...memories.map((m, i) => `${i + 1}. ${m.text}`)
    );
  }

  const system = systemParts.join('\n');
  const messages: ChatTurn[] = [];

  for (const pair of examples) {
    messages.push({ role: 'user', content: truncate(pair.context, MAX_CONTEXT_CHARS) });
    messages.push({ role: 'assistant', content: truncate(pair.reply, MAX_REPLY_CHARS) });
  }

  for (const turn of conversation) {
    const content = turn.content.trim();
    if (!content) continue;
    messages.push({
      role: turn.role,
      content: truncate(content, turn.role === 'user' ? 2000 : MAX_REPLY_CHARS),
    });
  }

  if (messages.length === 0 || messages[0].role !== 'user') {
    throw new Error('Conversation must include at least one user message.');
  }

  const normalized: ChatTurn[] = [];
  for (const turn of messages) {
    const last = normalized[normalized.length - 1];
    if (last && last.role === turn.role) {
      last.content = `${last.content}\n${turn.content}`;
    } else {
      normalized.push({ ...turn });
    }
  }

  if (normalized[normalized.length - 1].role !== 'user') {
    throw new Error('Last message must be from the user (awaiting a persona reply).');
  }

  return {
    system,
    messages: normalized,
    memoryCount: memories.length,
    exampleCount: examples.length,
  };
}
