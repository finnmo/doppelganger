'use client';

import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

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

  return (
    <div className={`relative inline-block ${className}`}>
      <button
        type="button"
        className="ml-1 rounded-full p-0.5 transition-colors duration-200 hover:bg-gray-100"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label={`Information about ${title}`}
      >
        <HelpCircle
          className={`h-3.5 w-3.5 ${
            iconColor === 'white'
              ? 'text-white text-opacity-70 hover:text-opacity-100'
              : 'text-gray-400 hover:text-gray-600'
          }`}
        />
      </button>

      {isVisible && (
        <div className="absolute left-1/2 top-full z-50 mt-2 w-[min(20rem,calc(100vw-2rem))] -translate-x-1/2 rounded-lg border border-gray-200 bg-white p-4 shadow-lg sm:left-6 sm:top-0 sm:mt-0 sm:translate-x-0">
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
      )}
    </div>
  );
}
