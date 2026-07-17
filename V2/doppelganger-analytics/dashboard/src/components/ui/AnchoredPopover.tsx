'use client';

import React, { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { getChartTooltipHost } from '@/components/ui/tooltipHost';

export interface AnchoredPopoverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Element that anchors the popover (usually the trigger button). */
  anchorRef: React.RefObject<HTMLElement | null>;
  children: React.ReactNode;
  className?: string;
  /** Gap in px between anchor and popover. */
  gap?: number;
  /** Preferred placement relative to anchor. */
  placement?: 'bottom' | 'top';
}

interface Position {
  top: number;
  left: number;
  placement: 'bottom' | 'top';
}

function computePosition(
  anchor: DOMRect,
  popover: DOMRect,
  gap: number,
  preferred: 'bottom' | 'top'
): Position {
  const vw = typeof window !== 'undefined' ? window.innerWidth : 0;
  const vh = typeof window !== 'undefined' ? window.innerHeight : 0;
  const margin = 8;

  const spaceBelow = vh - anchor.bottom;
  const spaceAbove = anchor.top;
  let placement: 'bottom' | 'top' = preferred;

  if (preferred === 'bottom' && spaceBelow < popover.height + gap + margin && spaceAbove > spaceBelow) {
    placement = 'top';
  } else if (preferred === 'top' && spaceAbove < popover.height + gap + margin && spaceBelow > spaceAbove) {
    placement = 'bottom';
  }

  let top =
    placement === 'bottom'
      ? anchor.bottom + gap
      : anchor.top - popover.height - gap;

  let left = anchor.left + anchor.width / 2 - popover.width / 2;

  // Prefer opening inward when the anchor sits near viewport edges
  if (anchor.right > vw - margin - 24) {
    left = anchor.right - popover.width;
  } else if (anchor.left < margin + 24) {
    left = anchor.left;
  }

  left = Math.max(margin, Math.min(left, vw - popover.width - margin));
  top = Math.max(margin, Math.min(top, vh - popover.height - margin));

  return { top, left, placement };
}

/**
 * Renders popover content in a portal so parent overflow-hidden/scroll
 * containers cannot clip hover tooltips.
 */
export function AnchoredPopover({
  open,
  onOpenChange,
  anchorRef,
  children,
  className = '',
  gap = 8,
  placement: preferredPlacement = 'bottom',
}: AnchoredPopoverProps) {
  const popoverRef = useRef<HTMLDivElement>(null);
  const [mounted, setMounted] = useState(false);
  const [position, setPosition] = useState<Position | null>(null);
  const id = useId();

  useEffect(() => {
    setMounted(true);
  }, []);

  const updatePosition = useCallback(() => {
    const anchor = anchorRef.current;
    const popover = popoverRef.current;
    if (!anchor || !popover) return;
    setPosition(computePosition(anchor.getBoundingClientRect(), popover.getBoundingClientRect(), gap, preferredPlacement));
  }, [anchorRef, gap, preferredPlacement]);

  useLayoutEffect(() => {
    if (!open || !mounted) return;
    updatePosition();
    const raf = requestAnimationFrame(updatePosition);
    return () => cancelAnimationFrame(raf);
  }, [open, mounted, updatePosition, children]);

  useEffect(() => {
    if (!open) return;

    const onScrollOrResize = () => updatePosition();
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onOpenChange(false);
    };
    const onPointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (popoverRef.current?.contains(target)) return;
      if (anchorRef.current?.contains(target)) return;
      onOpenChange(false);
    };

    window.addEventListener('scroll', onScrollOrResize, true);
    window.addEventListener('resize', onScrollOrResize);
    window.addEventListener('keydown', onKeyDown);
    document.addEventListener('mousedown', onPointerDown);

    return () => {
      window.removeEventListener('scroll', onScrollOrResize, true);
      window.removeEventListener('resize', onScrollOrResize);
      window.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('mousedown', onPointerDown);
    };
  }, [open, onOpenChange, updatePosition, anchorRef]);

  if (!mounted || !open || typeof document === 'undefined') return null;

  return createPortal(
    <div
      ref={popoverRef}
      id={id}
      role="tooltip"
      className={`fixed z-[1100] rounded-lg border border-gray-200 bg-white shadow-lg ${className}`}
      style={
        position
          ? { top: position.top, left: position.left }
          : { top: -9999, left: -9999, visibility: 'hidden' as const }
      }
    >
      {children}
    </div>,
    getChartTooltipHost()
  );
}
