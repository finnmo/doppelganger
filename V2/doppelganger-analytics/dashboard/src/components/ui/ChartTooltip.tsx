'use client';

import React from 'react';
import { Tooltip } from 'recharts';
import type { TooltipProps } from 'recharts';
import type { ValueType, NameType } from 'recharts/types/component/DefaultTooltipContent';
import { CHART_TOOLTIP_PROPS } from '@/lib/rechartsTooltip';
import { makePortalTooltipContent } from '@/components/ui/PortalRechartsTooltip';

type ChartTooltipProps = TooltipProps<ValueType, NameType>;

/**
 * Drop-in Recharts Tooltip that portals content to document.body so card
 * overflow-hidden never clips hover labels.
 */
export function ChartTooltip({ content, ...props }: ChartTooltipProps) {
  const wrappedContent = React.useMemo(() => {
    if (!content) return makePortalTooltipContent((p) => <DefaultPortalTooltip {...p} />);
    if (typeof content === 'function') return makePortalTooltipContent(content);
    return makePortalTooltipContent(() => content);
  }, [content]);

  return <Tooltip {...CHART_TOOLTIP_PROPS} {...props} content={wrappedContent} />;
}

/** Recharts finds tooltip children by displayName — must stay "Tooltip". */
ChartTooltip.displayName = 'Tooltip';

function DefaultPortalTooltip({
  active,
  payload,
  label,
}: TooltipProps<ValueType, NameType>) {
  if (!active || !payload?.length) return null;
  return (
    <div className="p-3 text-sm">
      {label != null && label !== '' && (
        <div className="mb-1 font-semibold text-gray-900">{String(label)}</div>
      )}
      {payload.map((entry, i) => (
        <div key={i} className="text-gray-700">
          {entry.name != null && entry.name !== '' ? `${entry.name}: ` : ''}
          {entry.value != null ? String(entry.value) : ''}
        </div>
      ))}
    </div>
  );
}
