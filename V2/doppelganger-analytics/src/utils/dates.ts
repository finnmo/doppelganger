/**
 * Format a timestamp as a local-timezone YYYY-MM-DD string.
 * (toISOString() would shift dates to UTC, misbucketing late-night messages.)
 */
export function localDateString(timestampMs: number): string {
  const date = new Date(timestampMs);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}
