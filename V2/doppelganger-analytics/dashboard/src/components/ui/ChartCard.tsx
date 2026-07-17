'use client';

import React, { ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { LucideIcon } from 'lucide-react';
import { Maximize2, X } from 'lucide-react';
import { InfoTooltip } from '@/components/InfoTooltip';
import { ChartPlotContext } from '@/components/ui/chartPlotContext';
import { FULLSCREEN_BODY } from '@/lib/layout';
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
  tooltip?: ChartCardTooltip;
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
  /** Optional enhanced view when fullscreen (defaults to children). */
  fullscreenChildren?: ReactNode;
  /** When false, hides the expand button (e.g. compact insight panels). */
  enableFullscreen?: boolean;
}

function CardShell({
  title,
  Icon,
  accent,
  tooltip,
  actions,
  bodyClassName,
  children,
  onFullscreen,
  onClose,
  radius,
  shellClassName = '',
}: {
  title: string;
  Icon: LucideIcon;
  accent: CardAccent;
  tooltip?: ChartCardTooltip;
  actions?: ReactNode;
  bodyClassName: string;
  children: ReactNode;
  onFullscreen?: () => void;
  onClose?: () => void;
  radius: string;
  shellClassName?: string;
}) {
  const plotRef = useRef<HTMLDivElement>(null);
  return (
    <section
      className={`flex min-h-0 min-w-0 flex-col bg-white ${radius} border border-gray-200 border-t-[3px] ${ACCENT_STRIP[accent]} shadow-sm ${shellClassName}`}
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
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {onFullscreen && (
            <button
              type="button"
              onClick={onFullscreen}
              className="rounded-md p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700"
              aria-label={`Expand ${title} to fullscreen`}
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          )}
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="rounded-md p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700"
              aria-label="Close fullscreen"
            >
              <X className="h-4 w-4" />
            </button>
          )}
          {actions}
        </div>
      </header>
      <ChartPlotContext.Provider value={plotRef}>
        <div ref={plotRef} data-chart-plot className={`min-h-0 grow overflow-hidden px-3 pb-3 sm:px-4 ${bodyClassName}`}>
          {children}
        </div>
      </ChartPlotContext.Provider>
    </section>
  );
}

export function ChartCard({
  title,
  icon: Icon,
  accent = 'blue',
  tooltip,
  actions,
  className = '',
  bodyClassName = '',
  children,
  fullscreenChildren,
  enableFullscreen = true,
}: ChartCardProps) {
  const { themeStyle } = useTheme();
  const radius = themeStyle === 'modern' ? 'rounded-xl' : 'rounded-lg';
  const [expanded, setExpanded] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const closeFullscreen = useCallback(() => setExpanded(false), []);

  useEffect(() => {
    if (!expanded) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeFullscreen();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener('keydown', onKey);
    };
  }, [expanded, closeFullscreen]);

  // Never mount the same chart tree in card + fullscreen at once — React will
  // only attach one instance, and hover/tooltips end up bound to the hidden card.
  const cardChildren = expanded ? null : children;
  const fullscreenBody = (
    <div className="flex h-full min-h-0 flex-col overflow-hidden [&>div]:min-h-0 [&>div]:flex-1">
      {fullscreenChildren ?? children}
    </div>
  );

  return (
    <>
      <CardShell
        title={title}
        Icon={Icon}
        accent={accent}
        tooltip={tooltip}
        actions={actions}
        bodyClassName={bodyClassName}
        onFullscreen={enableFullscreen ? () => setExpanded(true) : undefined}
        radius={radius}
        shellClassName={className}
      >
        {cardChildren}
      </CardShell>

      {mounted &&
        expanded &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            data-chart-fullscreen-root
            className="fixed inset-0 z-[100] flex flex-col bg-black/50 p-2 sm:p-3 md:p-4"
            role="dialog"
            aria-modal="true"
            aria-label={`${title} fullscreen`}
            onClick={closeFullscreen}
          >
            <div
              className="mx-auto flex min-h-0 w-full max-w-[min(100%,1600px)] flex-1 flex-col"
              onClick={(e) => e.stopPropagation()}
            >
              <CardShell
                title={title}
                Icon={Icon}
                accent={accent}
                tooltip={tooltip}
                actions={actions}
                bodyClassName={FULLSCREEN_BODY}
                onClose={closeFullscreen}
                radius={radius}
                shellClassName="h-full min-h-0 flex-1"
              >
                {fullscreenBody}
              </CardShell>
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
