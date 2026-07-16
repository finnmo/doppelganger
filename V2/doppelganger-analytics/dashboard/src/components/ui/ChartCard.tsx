'use client';

import React, { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { InfoTooltip } from '@/components/InfoTooltip';
import { useTheme } from '@/contexts/ThemeContext';

export type CardAccent =
  | 'blue'
  | 'green'
  | 'purple'
  | 'orange'
  | 'red'
  | 'yellow'
  | 'indigo'
  | 'pink'
  | 'teal'
  | 'slate';

// Static class maps so Tailwind keeps the styles (no dynamic class names).
const ACCENT_STRIP: Record<CardAccent, string> = {
  blue: 'border-t-blue-400',
  green: 'border-t-green-400',
  purple: 'border-t-purple-400',
  orange: 'border-t-orange-400',
  red: 'border-t-red-400',
  yellow: 'border-t-yellow-400',
  indigo: 'border-t-indigo-400',
  pink: 'border-t-pink-400',
  teal: 'border-t-teal-400',
  slate: 'border-t-slate-400',
};

const ACCENT_ICON: Record<CardAccent, string> = {
  blue: 'text-blue-600 bg-blue-50',
  green: 'text-green-600 bg-green-50',
  purple: 'text-purple-600 bg-purple-50',
  orange: 'text-orange-600 bg-orange-50',
  red: 'text-red-600 bg-red-50',
  yellow: 'text-yellow-600 bg-yellow-50',
  indigo: 'text-indigo-600 bg-indigo-50',
  pink: 'text-pink-600 bg-pink-50',
  teal: 'text-teal-600 bg-teal-50',
  slate: 'text-slate-600 bg-slate-50',
};

export interface ChartCardTooltip {
  description: string;
  calculation?: string;
  example?: string;
}

interface ChartCardProps {
  title: string;
  icon: LucideIcon;
  accent?: CardAccent;
  /** Full explanation lives in the tooltip so the card header stays one slim row. */
  tooltip?: ChartCardTooltip;
  /** Optional controls rendered at the right edge of the header (toggles, etc). */
  actions?: ReactNode;
  /** Grid placement / sizing from the parent (col-span, row-span, min-h…). */
  className?: string;
  /** Body sizing, e.g. a fixed chart height or internal scroll. */
  bodyClassName?: string;
  children: ReactNode;
}

/**
 * Dense dashboard card: one slim header row (icon chip, title, info tooltip,
 * optional actions) over a chart/list body. The colored top strip preserves
 * the section color-coding the old banner headers provided, at 3px instead of
 * ~110px, so a full tab fits on one screen.
 */
export function ChartCard({
  title,
  icon: Icon,
  accent = 'blue',
  tooltip,
  actions,
  className = '',
  bodyClassName = '',
  children,
}: ChartCardProps) {
  const { themeStyle } = useTheme();
  const radius = themeStyle === 'modern' ? 'rounded-xl' : 'rounded-lg';

  return (
    <section
      className={`flex min-w-0 flex-col overflow-hidden bg-white ${radius} border border-gray-200 border-t-[3px] ${ACCENT_STRIP[accent]} shadow-sm ${className}`}
    >
      <header className="flex shrink-0 items-center gap-2 px-3 pb-1.5 pt-2.5 sm:px-4">
        <span className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${ACCENT_ICON[accent]}`}>
          <Icon className="h-3.5 w-3.5" />
        </span>
        <h3 className="min-w-0 truncate text-sm font-semibold text-gray-900" title={title}>
          {title}
        </h3>
        {tooltip && (
          <InfoTooltip
            title={title}
            description={tooltip.description}
            calculation={tooltip.calculation}
            example={tooltip.example}
            iconColor="default"
          />
        )}
        {actions && <div className="ml-auto flex shrink-0 items-center gap-1.5">{actions}</div>}
      </header>
      {/* `grow` (not flex-1): flex-basis stays `auto`, so explicit body
          heights like h-64 are respected instead of content-sizing the card. */}
      <div className={`min-h-0 grow px-3 pb-3 sm:px-4 ${bodyClassName}`}>{children}</div>
    </section>
  );
}
