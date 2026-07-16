/** Split a model reply into text bubbles (dashboard copy of processor helper). */

export function splitReplyBubbles(text: string): string[] {
  const raw = text.trim();
  if (!raw) return [];

  if (raw.includes('<<<BUBBLE>>>')) {
    return raw
      .split('<<<BUBBLE>>>')
      .map((b) => b.trim())
      .filter(Boolean)
      .slice(0, 5);
  }

  const parts = raw
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter(Boolean);
  if (
    parts.length >= 2 &&
    parts.length <= 4 &&
    parts.every((p) => p.length <= 220 && !p.includes('\n\n'))
  ) {
    return parts;
  }

  return [raw];
}
