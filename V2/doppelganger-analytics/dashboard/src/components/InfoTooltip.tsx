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
  className = "",
  iconColor = 'default'
}: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className={`relative inline-block ${className}`}>
      <button
        className="ml-1 p-0.5 rounded-full hover:bg-gray-100 transition-colors duration-200"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        aria-label={`Information about ${title}`}
      >
        <HelpCircle className={`w-3.5 h-3.5 ${
          iconColor === 'white' 
            ? 'text-white text-opacity-70 hover:text-opacity-100' 
            : 'text-gray-400 hover:text-gray-600'
        }`} />
      </button>
      
      {isVisible && (
        <div className="absolute z-50 w-80 p-4 bg-white border border-gray-200 rounded-lg shadow-lg -top-2 left-6">
          <div className="space-y-3">
            <div>
              <h4 className="font-semibold text-gray-900 text-sm">{title}</h4>
              <p className="text-xs text-gray-600 mt-1">{description}</p>
            </div>
            
            {calculation && (
              <div>
                <h5 className="font-medium text-gray-800 text-xs">How it&apos;s calculated:</h5>
                <p className="text-xs text-gray-600 mt-1 font-mono bg-gray-50 p-2 rounded">
                  {calculation}
                </p>
              </div>
            )}
            
            {example && (
              <div>
                <h5 className="font-medium text-gray-800 text-xs">Example:</h5>
                <p className="text-xs text-gray-600 mt-1">{example}</p>
              </div>
            )}
          </div>
          
          {/* Arrow pointing to the icon */}
          <div className="absolute top-3 -left-1 w-2 h-2 bg-white border-l border-t border-gray-200 transform rotate-45"></div>
        </div>
      )}
    </div>
  );
} 