/**
 * Instagram/Facebook Unicode Decoder
 * 
 * Instagram exports JSON with malformed Unicode escape sequences.
 * Instead of proper Unicode code points like \u1F600 for 😀,
 * they encode UTF-8 bytes as individual \u00XX sequences.
 * 
 * Example:
 * - Correct: 😀 should be \uD83D\uDE00 (UTF-16 surrogate pairs)  
 * - Instagram: 😀 becomes \u00f0\u009f\u0098\u0080 (UTF-8 bytes as escapes)
 * 
 * This utility fixes that encoding issue.
 */

/**
 * Decode Facebook/Instagram's double-encoded Latin-1 data to proper UTF-8
 * This fixes the issue where emojis and special characters are double-encoded
 * @param text - Text that may be double-encoded
 * @returns Properly decoded Unicode text
 */
export function decodeFacebookLatinToUtf8(text: string): string {
  if (!text || typeof text !== 'string') {
    return text;
  }

  try {
    // Convert the string to Latin-1 bytes and then decode as UTF-8
    const decoder = new TextDecoder('utf-8');
    
    // First, encode as Latin-1 (ISO-8859-1)
    const latin1Bytes: number[] = [];
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      // Only process if it's within Latin-1 range
      if (code <= 255) {
        latin1Bytes.push(code);
      } else {
        // If it's already proper Unicode, return original
        return text;
      }
    }
    
    // Now decode as UTF-8
    const utf8Array = new Uint8Array(latin1Bytes);
    return decoder.decode(utf8Array);
  } catch (_error) {
    // If decoding fails, return original
    return text;
  }
}

/**
 * Decode Instagram's malformed Unicode escape sequences
 * @param text - Text containing malformed \u00XX sequences
 * @returns Properly decoded Unicode text
 */
export function decodeInstagramUnicode(text: string): string {
  if (!text || typeof text !== 'string') {
    return text;
  }

  // First try the Latin-1 to UTF-8 fix for double-encoded data
  const result = decodeFacebookLatinToUtf8(text);
  
  // Then handle escape sequences
  const pattern = /(?:\\u00[0-9a-fA-F]{2})+/g;
  
  return result.replace(pattern, (match) => {
    try {
      // Extract all the hex bytes from the escape sequences
      const hexMatches = match.match(/\\u00([0-9a-fA-F]{2})/g);
      if (!hexMatches) {
        return match; // Return original if no matches
      }

      // Convert escape sequences to actual bytes
      const bytes: number[] = [];
      for (const hexMatch of hexMatches) {
        const hexValue = hexMatch.substring(4); // Remove \u00 prefix
        const byteValue = parseInt(hexValue, 16);
        bytes.push(byteValue);
      }

      // Convert bytes to Uint8Array and decode as UTF-8
      const byteArray = new Uint8Array(bytes);
      const decoder = new TextDecoder('utf-8');
      return decoder.decode(byteArray);
      
    } catch (error) {
      console.warn('Failed to decode Unicode sequence:', match, error);
      return match; // Return original on error
    }
  });
}

/**
 * Process an entire object, decoding all string values
 * @param obj - Object to process
 * @returns Object with decoded strings
 */
export function decodeObjectUnicode<T>(obj: T): T {
  if (obj === null || obj === undefined) {
    return obj;
  }

  if (typeof obj === 'string') {
    return decodeInstagramUnicode(obj) as T;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => decodeObjectUnicode(item)) as T;
  }

  if (typeof obj === 'object') {
    const result: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(obj)) {
      result[key] = decodeObjectUnicode(value);
    }
    return result as T;
  }

  return obj;
}

/**
 * Check if a string contains Instagram's malformed Unicode sequences
 * @param text - Text to check
 * @returns True if malformed sequences are detected
 */
export function hasInstagramEncoding(text: string): boolean {
  if (!text || typeof text !== 'string') {
    return false;
  }
  
  // Look for patterns like \u00XX that could be UTF-8 bytes
  return /\\u00[8-9a-fA-F][0-9a-fA-F]/i.test(text);
}

/**
 * Extract and count emojis from text
 * @param text - Text to analyze
 * @returns Object with emoji count and list
 */
export function extractEmojis(text: string): { count: number; emojis: string[] } {
  if (!text || typeof text !== 'string') {
    return { count: 0, emojis: [] };
  }

  // Unicode ranges for emojis
  const emojiRegex = /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu;
  
  const matches = text.match(emojiRegex) || [];
  const uniqueEmojis = [...new Set(matches)];
  
  return {
    count: matches.length,
    emojis: uniqueEmojis
  };
}

/**
 * Detect reaction patterns in Instagram messages
 * @param text - Message text to analyze
 * @returns Reaction information
 */
export function detectReaction(text: string): { 
  isReaction: boolean; 
  reactionType?: string; 
  emoji?: string;
  target?: string;
} {
  if (!text || typeof text !== 'string') {
    return { isReaction: false };
  }

  // Common Instagram reaction patterns
  const reactionPatterns = [
    /reacted (.+) to your message/i,
    /reacted (.+) to a message/i,
    /liked your message/i,
    /loved your message/i,
    /laughed at your message/i,
    /disliked your message/i,
    /emphasized your message/i,
    /questioned your message/i,
  ];

  for (const pattern of reactionPatterns) {
    const match = text.match(pattern);
    if (match) {
      const emojiInfo = extractEmojis(text);
      return {
        isReaction: true,
        reactionType: match[1] || 'like',
        emoji: emojiInfo.emojis[0],
        target: 'message'
      };
    }
  }

  // Check for standalone emoji (potential reaction)
  const emojiInfo = extractEmojis(text);
  if (emojiInfo.count === 1 && text.trim().length <= 10) {
    return {
      isReaction: true,
      reactionType: 'emoji',
      emoji: emojiInfo.emojis[0]
    };
  }

  return { isReaction: false };
}

/**
 * Test function to validate the decoder with known examples
 */
export function testDecoder(): void {
  const tests = [
    {
      input: '\\u00f0\\u009f\\u0098\\u0080', // 😀 encoded incorrectly
      expected: '😀'
    },
    {
      input: '\\u00f0\\u009f\\u0098\\u0082', // 😂 encoded incorrectly  
      expected: '😂'
    },
    {
      input: 'Hello \\u00f0\\u009f\\u0098\\u0080 world!',
      expected: 'Hello 😀 world!'
    }
  ];

  console.log('Testing Instagram Unicode decoder:');
  for (const test of tests) {
    const result = decodeInstagramUnicode(test.input);
    const passed = result === test.expected;
    console.log(`Input: ${test.input}`);
    console.log(`Expected: ${test.expected}`);
    console.log(`Got: ${result}`);
    console.log(`✅ ${passed ? 'PASS' : 'FAIL'}\n`);
  }
} 