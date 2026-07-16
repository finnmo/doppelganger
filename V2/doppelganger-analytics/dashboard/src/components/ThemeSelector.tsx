'use client';

import React, { useState } from 'react';
import { useTheme, ThemeStyle } from '@/contexts/ThemeContext';
import { Palette, Monitor, Sparkles, ChevronDown, Check } from 'lucide-react';

export function ThemeSelector() {
  const { themeStyle, setThemeStyle } = useTheme();
  const [isOpen, setIsOpen] = useState(false);

  const themes: Array<{
    id: ThemeStyle;
    name: string;
    description: string;
    icon: React.ReactNode;
    preview: {
      primary: string;
      secondary: string;
      accent: string;
    };
  }> = [
    {
      id: 'modern',
      name: 'Modern',
      description: 'Contemporary design with gradients and rounded corners',
      icon: <Sparkles className="w-4 h-4" />,
      preview: {
        primary: 'bg-gradient-to-r from-blue-400 to-blue-500',
        secondary: 'bg-gradient-to-r from-emerald-400 to-emerald-500',
        accent: 'bg-gradient-to-r from-violet-400 to-violet-500',
      },
    },
    {
      id: 'classic',
      name: 'Classic',
      description: 'Traditional design with clean lines and subtle shadows',
      icon: <Monitor className="w-4 h-4" />,
      preview: {
        primary: 'bg-blue-600',
        secondary: 'bg-emerald-600',
        accent: 'bg-violet-600',
      },
    },
  ];

  const selectedTheme = themes.find(theme => theme.id === themeStyle);

  const handleThemeChange = (newTheme: ThemeStyle) => {
    setThemeStyle(newTheme);
    setIsOpen(false);
  };

  return (
    <div className="relative">
      {/* Theme Selector Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 
          bg-white border border-gray-300 rounded-lg hover:bg-gray-50 
          focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
          transition-colors duration-200
        `}
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        <Palette className="w-4 h-4" />
        <span>{selectedTheme?.name} Theme</span>
        <ChevronDown 
          className={`w-4 h-4 transition-transform duration-200 ${
            isOpen ? 'transform rotate-180' : ''
          }`} 
        />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setIsOpen(false)}
          />
          
          {/* Menu */}
          <div className="absolute right-0 mt-2 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-20">
            <div className="p-4 border-b border-gray-200">
              <h3 className="text-sm font-semibold text-gray-900 flex items-center">
                <Palette className="w-4 h-4 mr-2" />
                Choose Theme Style
              </h3>
              <p className="text-xs text-gray-500 mt-1">
                Select your preferred visual style for the dashboard
              </p>
            </div>
            
            <div className="p-2">
              {themes.map((theme) => (
                <button
                  key={theme.id}
                  onClick={() => handleThemeChange(theme.id)}
                  className={`
                    w-full flex items-start space-x-3 p-3 rounded-lg text-left
                    transition-colors duration-200
                    ${themeStyle === theme.id 
                      ? 'bg-blue-50 border border-blue-200' 
                      : 'hover:bg-gray-50 border border-transparent'
                    }
                  `}
                >
                  {/* Theme Icon & Check */}
                  <div className="flex-shrink-0 flex items-center space-x-2">
                    <div className={`
                      p-2 rounded-lg
                      ${themeStyle === theme.id ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-600'}
                    `}>
                      {theme.icon}
                    </div>
                    {themeStyle === theme.id && (
                      <Check className="w-4 h-4 text-blue-600" />
                    )}
                  </div>
                  
                  {/* Theme Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h4 className={`
                        text-sm font-medium
                        ${themeStyle === theme.id ? 'text-blue-900' : 'text-gray-900'}
                      `}>
                        {theme.name}
                      </h4>
                    </div>
                    <p className={`
                      text-xs mt-1
                      ${themeStyle === theme.id ? 'text-blue-700' : 'text-gray-500'}
                    `}>
                      {theme.description}
                    </p>
                    
                    {/* Color Preview */}
                    <div className="flex space-x-1 mt-2">
                      <div className={`w-4 h-4 rounded-sm ${theme.preview.primary}`} />
                      <div className={`w-4 h-4 rounded-sm ${theme.preview.secondary}`} />
                      <div className={`w-4 h-4 rounded-sm ${theme.preview.accent}`} />
                    </div>
                  </div>
                </button>
              ))}
            </div>
            
            {/* Footer */}
            <div className="p-3 border-t border-gray-200 bg-gray-50 rounded-b-lg">
              <p className="text-xs text-gray-500">
                Your theme preference is saved automatically and will persist across sessions.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// Compact version for mobile or space-constrained areas
export function CompactThemeSelector() {
  const { themeStyle, setThemeStyle } = useTheme();

  return (
    <div className="flex items-center space-x-1 bg-gray-100 rounded-lg p-1">
      <button
        onClick={() => setThemeStyle('modern')}
        className={`
          flex items-center space-x-1 px-3 py-1 text-xs font-medium rounded-md transition-colors
          ${themeStyle === 'modern' 
            ? 'bg-white text-blue-600 shadow-sm' 
            : 'text-gray-600 hover:text-gray-900'
          }
        `}
      >
        <Sparkles className="w-3 h-3" />
        <span>Modern</span>
      </button>
      <button
        onClick={() => setThemeStyle('classic')}
        className={`
          flex items-center space-x-1 px-3 py-1 text-xs font-medium rounded-md transition-colors
          ${themeStyle === 'classic' 
            ? 'bg-white text-blue-600 shadow-sm' 
            : 'text-gray-600 hover:text-gray-900'
          }
        `}
      >
        <Monitor className="w-3 h-3" />
        <span>Classic</span>
      </button>
    </div>
  );
} 