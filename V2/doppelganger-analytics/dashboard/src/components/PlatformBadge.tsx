'use client';

import React from 'react';
import { platformStyles, sourceLabel } from '@/lib/platforms';

interface PlatformBadgeProps {
  source: string | null | undefined;
  size?: 'sm' | 'md';
  className?: string;
}

/** Compact platform label used on conversation cards and headers. */
export function PlatformBadge({ source, size = 'sm', className = '' }: PlatformBadgeProps) {
  const styles = platformStyles(source);
  const label = sourceLabel(source);
  const sizeClass =
    size === 'md'
      ? 'text-xs px-2.5 py-1 gap-1.5'
      : 'text-[10px] px-2 py-0.5 gap-1';

  return (
    <span
      className={`inline-flex items-center font-semibold tracking-wide uppercase rounded-md ring-1 ring-inset ${styles.bg} ${styles.text} ${styles.ring} ${sizeClass} ${className}`}
      title={`Imported from ${label}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${styles.dot}`} aria-hidden />
      {label}
    </span>
  );
}
