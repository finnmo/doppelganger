/** Shared Recharts tooltip props (positioning handled by portal wrapper). */
export const CHART_TOOLTIP_PROPS = {
  allowEscapeViewBox: { x: true, y: true },
  reverseDirection: { x: true, y: true },
  wrapperStyle: { visibility: 'hidden' as const, pointerEvents: 'none' as const },
} as const;
