// Builds LLM prompts for persona simulation (~85% target: style profile + few-shot pairs).
// Does not call any model — consumers pass the result to their LLM provider.

import type { PersonaStyleProfile } from '../processors/personaMetrics.js';

export interface FewShotPair {
  /** Prior message(s) the person was responding to. */
  context: string;
  /** Their actual reply. */
  reply: string;
  conversationId: string;
  source: string;
}

export interface PersonaPromptBundle {
  sender: string;
  system: string;
  fewShot: FewShotPair[];
  /** Ready-to-send chat messages for OpenAI-style APIs. */
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
}

const MAX_CONTEXT_CHARS = 400;
const MAX_REPLY_CHARS = 500;

function truncate(text: string, max: number): string {
  const t = text.trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max - 1)}…`;
}

/**
 * Build a system prompt + few-shot message list for high-similarity reply generation.
 * Target tier: ~70–85% with 20–40 diverse pairs and a capable chat model.
 */
export function buildPersonaPrompt(
  profile: PersonaStyleProfile,
  fewShot: FewShotPair[],
  options?: { maxExamples?: number }
): PersonaPromptBundle {
  const maxExamples = options?.maxExamples ?? 25;
  const examples = fewShot.slice(0, maxExamples);

  const system = [
    `You are ${profile.sender} texting in a private chat. Stay fully in character.`,
    `Sound like them: vocabulary, rhythm, punctuation, slang, humor, and warmth.`,
    profile.styleSummary,
    `When there is something to say, write a natural engaged reply — not a clipped shutdown. Short acks only when the message warrants it.`,
    '',
    'Rules:',
    '- Keep the conversation going the way they would (react, riff, ask, share).',
    '- Match the energy of the latest message.',
    '- Never mention that you are an AI or that you are role-playing.',
    '- Stay in character; do not moralize or add disclaimers.',
    '- Reply with only the message text they would send.'
  ].join('\n');

  const messages: PersonaPromptBundle['messages'] = [{ role: 'system', content: system }];

  for (const pair of examples) {
    messages.push({
      role: 'user',
      content: truncate(pair.context, MAX_CONTEXT_CHARS)
    });
    messages.push({
      role: 'assistant',
      content: truncate(pair.reply, MAX_REPLY_CHARS)
    });
  }

  return {
    sender: profile.sender,
    system,
    fewShot: examples,
    messages
  };
}
