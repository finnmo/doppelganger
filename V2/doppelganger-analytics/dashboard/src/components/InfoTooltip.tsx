'use client';

import React, { useRef, useState } from 'react';
import { HelpCircle } from 'lucide-react';
import { AnchoredPopover } from '@/components/ui/AnchoredPopover';

interface InfoTooltipProps {
  title: string;
  description: string;
  calculation?: string;
  example?: string;
  className?: string;
  iconColor?: 'default' | 'white';
}

export function InfoTooltip({
  title,
  description,
  calculation,
  example,
  className = '',
  iconColor = 'default',
}: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);

  return (
    <div className={`inline-block ${className}`}>
      <button
        ref={triggerRef}
        type="button"
        className="ml-1 rounded-full p-0.5 transition-colors duration-200 hover:bg-gray-100"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label={`Information about ${title}`}
        aria-describedby={isVisible ? `info-${title}` : undefined}
      >
        <HelpCircle
          className={`h-3.5 w-3.5 ${
            iconColor === 'white'
              ? 'text-white text-opacity-70 hover:text-opacity-100'
              : 'text-gray-400 hover:text-gray-600'
          }`}
        />
      </button>

      <AnchoredPopover open={isVisible} onOpenChange={setIsVisible} anchorRef={triggerRef}>
        <div id={`info-${title}`} className="w-[min(20rem,calc(100vw-2rem))] p-4">
          <div className="space-y-3">
            <div>
              <h4 className="text-sm font-semibold text-gray-900">{title}</h4>
              <p className="mt-1 text-xs text-gray-600">{description}</p>
            </div>

            {calculation && (
              <div>
                <h5 className="text-xs font-medium text-gray-800">How it&apos;s calculated:</h5>
                <p className="mt-1 break-words rounded bg-gray-50 p-2 font-mono text-xs text-gray-600">
                  {calculation}
                </p>
              </div>
            )}

            {example && (
              <div>
                <h5 className="text-xs font-medium text-gray-800">Example:</h5>
                <p className="mt-1 text-xs text-gray-600">{example}</p>
              </div>
            )}
          </div>
        </div>
      </AnchoredPopover>
    </div>
  );
}
