import { decodeInstagramUnicode, extractEmojis } from '../utils/unicodeDecoder.js';

const URL_REGEX = /(?:https?:\/\/|www\.|[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.)[^\s]+/g;

export interface TextMetrics {
  wordCount: number;
  emojiCount: number;
  urlCount: number;
}

export function processText(text: string | null): TextMetrics {
  if (!text) {
    return {
      wordCount: 0,
      emojiCount: 0,
      urlCount: 0
    };
  }

  // Extract emojis from the original text (before decoding) to catch malformed Unicode emojis
  const originalEmojis = extractEmojis(text);
  
  // Decode Instagram's malformed Unicode encoding
  const decodedText = decodeInstagramUnicode(text);
  
  // Extract emojis from decoded text as well to catch any that were properly encoded
  const decodedEmojis = extractEmojis(decodedText);
  
  // Count words more accurately - split by whitespace and filter out empty strings
  const words = decodedText.trim().length > 0
    ? decodedText.trim().split(/\s+/).filter(word => word.length > 0).length
    : 0;
  
  // Count URLs using the improved regex
  const urlMatches = decodedText.match(URL_REGEX) || [];
  
  // Total emoji count from both original and decoded text (avoiding duplicates)
  const totalEmojiCount = Math.max(originalEmojis.count, decodedEmojis.count);

  return {
    wordCount: words,
    emojiCount: totalEmojiCount,
    urlCount: urlMatches.length
  };
} 