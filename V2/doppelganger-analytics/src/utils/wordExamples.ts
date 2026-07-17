import { trimMessageSnippet } from './messageFilters.js';

export interface WordExampleMessage {
  text: string;
  sender: string;
  conversation_id: string;
}

export interface WordExampleEntry {
  word: string;
  examples: WordExampleMessage[];
}

const MAX_EXAMPLES_PER_WORD = 3;
const TOP_WORDS_FOR_EXAMPLES = 300;

/**
 * Reservoir-sample up to k real message snippets per word for the global top-N words.
 */
export function reservoirSampleWordExamples(
  messages: Array<{ content: string; sender: string; conversation_id: string; words: string[] }>,
  topWords: string[]
): WordExampleEntry[] {
  const target = new Set(topWords.slice(0, TOP_WORDS_FOR_EXAMPLES));
  const reservoirs = new Map<string, WordExampleMessage[]>();
  const seen = new Map<string, number>();

  for (const msg of messages) {
    const uniqueInMsg = new Set(msg.words.filter(w => target.has(w)));
    for (const word of uniqueInMsg) {
      const n = (seen.get(word) ?? 0) + 1;
      seen.set(word, n);
      let bucket = reservoirs.get(word);
      if (!bucket) {
        bucket = [];
        reservoirs.set(word, bucket);
      }
      const example: WordExampleMessage = {
        text: trimMessageSnippet(msg.content),
        sender: msg.sender,
        conversation_id: msg.conversation_id
      };
      if (bucket.length < MAX_EXAMPLES_PER_WORD) {
        bucket.push(example);
      } else {
        const j = Math.floor(Math.random() * n);
        if (j < MAX_EXAMPLES_PER_WORD) bucket[j] = example;
      }
    }
  }

  return [...target]
    .filter(word => (reservoirs.get(word)?.length ?? 0) > 0)
    .map(word => ({ word, examples: reservoirs.get(word)! }));
}
