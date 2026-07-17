'use client';

import React, { useContext, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import type { TooltipProps } from 'recharts';
import type { ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { ChartPlotContext } from '@/components/ui/chartPlotContext';
import { getChartTooltipHost } from '@/components/ui/tooltipHost';

function clampPosition(
  anchorX: number,
  anchorY: number,
  width: number,
  height: number,
  offset = 12
): { left: number; top: number } {
  const margin = 8;
  const vw = typeof window !== 'undefined' ? window.innerWidth : 0;
  const vh = typeof window !== 'undefined' ? window.innerHeight : 0;

  let left = anchorX + offset;
  let top = anchorY - height - offset;

  if (left + width > vw - margin) {
    left = anchorX - width - offset;
  }
  if (left < margin) {
    left = margin;
  }
  if (top < margin) {
    top = anchorY + offset;
  }
  if (top + height > vh - margin) {
    top = Math.max(margin, vh - height - margin);
  }

  return { left, top };
}

interface PortalRechartsTooltipProps extends TooltipProps<ValueType, NameType> {
  render: () => React.ReactNode;
}

/**
 * Renders Recharts tooltip content in a body portal so ChartCard
 * overflow-hidden cannot clip hover labels.
 */
export function PortalRechartsTooltip({
  active,
  coordinate,
  render,
}: PortalRechartsTooltipProps) {
  const plotRef = useContext(ChartPlotContext);
  const tipRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  useLayoutEffect(() => {
    if (!active || !coordinate || !tipRef.current) {
      setPosition(null);
      return;
    }

    const plotRoot = plotRef?.current;
    const plot =
      (plotRoot?.querySelector('.recharts-wrapper') as HTMLElement | null) ??
      plotRoot ??
      tipRef.current.closest('.recharts-responsive-container') ??
      tipRef.current.closest('[data-chart-plot]');

    if (!plot) {
      setPosition(null);
      return;
    }

    const updatePosition = () => {
      if (!tipRef.current || !coordinate) return;
      const { x, y } = coordinate;
      if (x == null || y == null) return;
      const plotRect = plot.getBoundingClientRect();
      const tipRect = tipRef.current.getBoundingClientRect();
      const anchorX = plotRect.left + x;
      const anchorY = plotRect.top + y;
      setPosition(clampPosition(anchorX, anchorY, tipRect.width, tipRect.height));
    };

    updatePosition();
    const frame = requestAnimationFrame(updatePosition);
    return () => cancelAnimationFrame(frame);
  }, [active, coordinate, plotRef]);

  if (!active || typeof document === 'undefined') {
    return null;
  }

  const content = (
    <div
      ref={tipRef}
      data-chart-portal-tooltip
      className="pointer-events-none fixed z-[1100] max-w-[min(20rem,calc(100vw-1rem))] rounded-lg border border-gray-200 bg-white shadow-lg"
      style={
        position
          ? { left: position.left, top: position.top }
          : { left: -9999, top: -9999, visibility: 'hidden' }
      }
    >
      {render()}
    </div>
  );

  return createPortal(content, getChartTooltipHost());
}

export function makePortalTooltipContent(
  render: (props: TooltipProps<ValueType, NameType>) => React.ReactNode
) {
  return function PortalTooltipContent(props: TooltipProps<ValueType, NameType>) {
    return <PortalRechartsTooltip {...props} render={() => render(props)} />;
  };
}
