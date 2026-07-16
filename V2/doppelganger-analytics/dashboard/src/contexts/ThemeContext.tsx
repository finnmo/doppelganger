'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export type ThemeStyle = 'modern' | 'classic';

/**
 * DOPPELGANGER ANALYTICS THEME DESIGN RULES
 * ==========================================
 * 
 * MODERN THEME:
 * - Background: All cards use solid white backgrounds (bg-white)
 * - Borders: Subtle gray borders (border-gray-200) 
 * - Shadows: Enhanced shadows (shadow-lg hover:shadow-xl)
 * - Border Radius: Rounded corners (rounded-2xl for cards, rounded-t-2xl for headers)
 * - Padding: Generous padding (p-8 for cards)
 * - Spacing: Larger gaps between elements (gap-10, space-y-10)
 * - Headers: Gradient backgrounds with white text for section headers only
 * - Colors: Vibrant accent colors for icons and interactive elements
 * - Typography: Bold, modern font weights
 * 
 * CLASSIC THEME:
 * - Background: All cards use solid white backgrounds (bg-white)
 * - Borders: Standard gray borders (border-gray-300)
 * - Shadows: Subtle shadows (shadow-sm)
 * - Border Radius: Conservative rounding (rounded-lg for cards)
 * - Padding: Standard padding (p-6 for cards)
 * - Spacing: Compact gaps (gap-8, space-y-8)
 * - Headers: Simple titles with colored icons, no gradient backgrounds
 * - Colors: Professional, muted accent colors
 * - Typography: Standard font weights, clean and readable
 * 
 * CONSISTENCY RULES:
 * - Hero cards should match other cards (white background, not gradient)
 * - Only section headers in modern theme use gradient backgrounds
 * - All cards maintain consistent padding within each theme
 * - Icon colors should be vibrant in modern, professional in classic
 * - Tooltips use white icons on gradients, default icons on white backgrounds
 */

interface ThemeConfig {
  style: ThemeStyle;
  colors: {
    primary: string;
    secondary: string;
    accent: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
    success: string;
    warning: string;
    error: string;
    info: string;
  };
  gradients: {
    primary: string;
    secondary: string;
    accent: string;
    surface: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  spacing: {
    card: string;
    section: string;
  };
}

const themes: Record<ThemeStyle, ThemeConfig> = {
  modern: {
    style: 'modern',
    colors: {
      primary: '#3b82f6', // blue-500
      secondary: '#10b981', // emerald-500
      accent: '#8b5cf6', // violet-500
      background: '#f9fafb', // gray-50
      surface: '#ffffff',
      text: '#111827', // gray-900
      textSecondary: '#6b7280', // gray-500
      border: '#e5e7eb', // gray-200
      success: '#10b981', // emerald-500
      warning: '#f59e0b', // amber-500
      error: '#ef4444', // red-500
      info: '#06b6d4', // cyan-500
    },
    gradients: {
      primary: 'from-blue-400 to-blue-500',
      secondary: 'from-emerald-400 to-emerald-500',
      accent: 'from-violet-400 to-violet-500',
      surface: 'from-blue-50 via-indigo-50 to-purple-50',
    },
    shadows: {
      sm: 'shadow-sm',
      md: 'shadow-md',
      lg: 'shadow-lg hover:shadow-xl',
      xl: 'shadow-xl hover:shadow-2xl',
    },
    borderRadius: {
      sm: 'rounded-lg',
      md: 'rounded-xl',
      lg: 'rounded-2xl',
      xl: 'rounded-3xl',
    },
    spacing: {
      card: 'p-8',
      section: 'space-y-10',
    },
  },
  classic: {
    style: 'classic',
    colors: {
      primary: '#2563eb', // blue-600
      secondary: '#059669', // emerald-600
      accent: '#7c3aed', // violet-600
      background: '#ffffff',
      surface: '#f8fafc', // slate-50
      text: '#1e293b', // slate-800
      textSecondary: '#64748b', // slate-500
      border: '#cbd5e1', // slate-300
      success: '#059669', // emerald-600
      warning: '#d97706', // amber-600
      error: '#dc2626', // red-600
      info: '#0284c7', // sky-600
    },
    gradients: {
      primary: 'from-blue-400 to-blue-500',
      secondary: 'from-emerald-400 to-emerald-500',
      accent: 'from-violet-400 to-violet-500',
      surface: 'from-blue-50 via-indigo-50 to-purple-50',
    },
    shadows: {
      sm: 'shadow-sm',
      md: 'shadow-sm',
      lg: 'shadow-sm',
      xl: 'shadow-md',
    },
    borderRadius: {
      sm: 'rounded',
      md: 'rounded-md',
      lg: 'rounded-lg',
      xl: 'rounded-xl',
    },
    spacing: {
      card: 'p-6',
      section: 'space-y-8',
    },
  },
};

interface ThemeContextType {
  currentTheme: ThemeConfig;
  themeStyle: ThemeStyle;
  setThemeStyle: (style: ThemeStyle) => void;
  getThemeClasses: () => {
    cardClass: string;
    headerGradientClass: string;
    buttonClass: string;
    heroCardClass: (color: string) => string;
    sectionCardClass: string;
    sectionHeaderClass: (color: string) => string;
    sectionContentClass: string;
    sectionTitleClass: string;
    textClass: string;
    borderClass: string;
    shadowClass: string;
    spacingClass: string;
  };
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const [themeStyle, setThemeStyleState] = useState<ThemeStyle>('modern');

  // Load theme from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('doppelganger-theme') as ThemeStyle;
    if (savedTheme && (savedTheme === 'modern' || savedTheme === 'classic')) {
      setThemeStyleState(savedTheme);
    }
  }, []);

  // Update CSS variables when theme changes
  useEffect(() => {
    const theme = themes[themeStyle];
    const root = document.documentElement;
    
    // Update CSS custom properties
    root.style.setProperty('--theme-primary', theme.colors.primary);
    root.style.setProperty('--theme-secondary', theme.colors.secondary);
    root.style.setProperty('--theme-accent', theme.colors.accent);
    root.style.setProperty('--theme-background', theme.colors.background);
    root.style.setProperty('--theme-surface', theme.colors.surface);
    root.style.setProperty('--theme-text', theme.colors.text);
    root.style.setProperty('--theme-text-secondary', theme.colors.textSecondary);
    root.style.setProperty('--theme-border', theme.colors.border);
    root.style.setProperty('--theme-success', theme.colors.success);
    root.style.setProperty('--theme-warning', theme.colors.warning);
    root.style.setProperty('--theme-error', theme.colors.error);
    root.style.setProperty('--theme-info', theme.colors.info);
    
    // Add theme class to body for global styling
    document.body.className = document.body.className.replace(/theme-\w+/g, '');
    document.body.classList.add(`theme-${themeStyle}`);
  }, [themeStyle]);

  // Save theme to localStorage when changed
  const setThemeStyle = (style: ThemeStyle) => {
    setThemeStyleState(style);
    localStorage.setItem('doppelganger-theme', style);
  };

  const currentTheme = themes[themeStyle];

  const getThemeClasses = () => {
    const isModern = themeStyle === 'modern';
    
    // Predefined gradient classes for section headers to avoid Tailwind purging
    const sectionHeaderGradients = {
      blue: 'bg-gradient-to-r from-blue-400 to-blue-500 text-white p-4 sm:p-6 rounded-t-2xl',
      green: 'bg-gradient-to-r from-green-400 to-green-500 text-white p-4 sm:p-6 rounded-t-2xl',
      purple: 'bg-gradient-to-r from-purple-400 to-purple-500 text-white p-4 sm:p-6 rounded-t-2xl',
      orange: 'bg-gradient-to-r from-orange-400 to-orange-500 text-white p-4 sm:p-6 rounded-t-2xl',
      red: 'bg-gradient-to-r from-red-400 to-red-500 text-white p-4 sm:p-6 rounded-t-2xl',
      yellow: 'bg-gradient-to-r from-yellow-400 to-yellow-500 text-white p-4 sm:p-6 rounded-t-2xl',
      indigo: 'bg-gradient-to-r from-indigo-400 to-indigo-500 text-white p-4 sm:p-6 rounded-t-2xl',
      pink: 'bg-gradient-to-r from-pink-400 to-pink-500 text-white p-4 sm:p-6 rounded-t-2xl',
    };

    // Hero cards should match other cards - white background with colored left border
    const heroCardClasses = {
      blue: isModern 
        ? 'bg-white border-l-4 border-blue-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-blue-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
      orange: isModern 
        ? 'bg-white border-l-4 border-orange-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-orange-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
      green: isModern 
        ? 'bg-white border-l-4 border-green-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-green-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
      purple: isModern 
        ? 'bg-white border-l-4 border-purple-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-purple-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
      red: isModern 
        ? 'bg-white border-l-4 border-red-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-red-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
      yellow: isModern 
        ? 'bg-white border-l-4 border-yellow-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-yellow-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
      indigo: isModern 
        ? 'bg-white border-l-4 border-indigo-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-indigo-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
      pink: isModern 
        ? 'bg-white border-l-4 border-pink-500 rounded-2xl p-4 sm:p-6 lg:p-8 shadow-lg hover:shadow-xl transition-shadow border border-gray-200'
        : 'bg-white border-l-4 border-pink-600 rounded-lg p-4 sm:p-6 shadow-sm border border-gray-300',
    };
    
    return {
      cardClass: isModern 
        ? `bg-white ${currentTheme.borderRadius.xl} border border-gray-200 ${currentTheme.shadows.lg} transition-shadow hover:shadow-xl p-4 sm:p-6 lg:p-8`
        : `bg-white p-4 sm:p-6 rounded-lg shadow-sm border`,
      
      headerGradientClass: isModern
        ? `bg-gradient-to-r ${currentTheme.gradients.surface} ${currentTheme.borderRadius.xl} opacity-40`
        : `bg-gradient-to-r from-orange-50 via-red-50 to-pink-50 rounded-lg opacity-40`,
      
      buttonClass: isModern
        ? `${currentTheme.borderRadius.md} transition-all duration-200 hover:scale-105`
        : `${currentTheme.borderRadius.sm} transition-colors`,
      
      // For hero cards - both themes use same structure and styling
      heroCardClass: (color: string) => 
        heroCardClasses[color as keyof typeof heroCardClasses] || heroCardClasses.blue,
      
      // Modern section cards use flush header + content padding (no double p-8)
      sectionCardClass: isModern
        ? `bg-white ${currentTheme.borderRadius.xl} border border-gray-200 ${currentTheme.shadows.lg} transition-shadow hover:shadow-xl overflow-hidden`
        : `bg-white p-4 sm:p-6 rounded-lg shadow-sm border`,
      
      // For section headers within cards - only used in modern theme
      sectionHeaderClass: (color: string) => 
        sectionHeaderGradients[color as keyof typeof sectionHeaderGradients] || sectionHeaderGradients.blue,
      
      // For section content - only used in modern theme  
      sectionContentClass: 'p-4 sm:p-6',
      
      // For simple section titles in classic theme
      sectionTitleClass: 'text-lg font-semibold text-gray-900 mb-4 sm:mb-6 flex flex-wrap items-center gap-2',
      
      textClass: `text-${currentTheme.colors.text}`,
      borderClass: `border-${currentTheme.colors.border}`,
      shadowClass: currentTheme.shadows.lg,
      spacingClass: isModern ? 'space-y-6 sm:space-y-10' : 'space-y-6 sm:space-y-8',
    };
  };

  const value: ThemeContextType = {
    currentTheme,
    themeStyle,
    setThemeStyle,
    getThemeClasses,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// Theme-aware component utilities
export const themeClasses = {
  modern: {
    hero: {
      card: 'bg-gradient-to-br rounded-2xl p-8 text-white shadow-lg hover:shadow-xl transition-shadow',
      colors: {
        blue: 'from-blue-400 to-blue-500',
        orange: 'from-orange-400 to-orange-500',
        green: 'from-green-400 to-green-500',
        purple: 'from-purple-400 to-purple-500',
        red: 'from-red-400 to-red-500',
        yellow: 'from-yellow-400 to-yellow-500',
        indigo: 'from-indigo-400 to-indigo-500',
        pink: 'from-pink-400 to-pink-500',
      },
    },
    section: {
      header: 'bg-gradient-to-r from-blue-50 via-indigo-50 to-purple-50 rounded-2xl opacity-40',
      card: 'bg-white rounded-2xl border shadow-lg hover:shadow-xl transition-shadow p-8',
      spacing: 'space-y-10',
    },
    participant: {
      card: 'bg-white rounded-2xl border shadow-lg hover:shadow-xl transition-shadow p-8',
      header: 'bg-gradient-to-r text-white p-8 rounded-t-2xl',
      content: 'p-8 space-y-4',
    },
  },
  classic: {
    hero: {
      card: 'bg-gradient-to-br rounded-2xl p-6 text-white shadow-lg hover:shadow-xl transition-shadow',
      colors: {
        blue: 'from-blue-400 to-blue-500',
        orange: 'from-orange-400 to-orange-500',
        green: 'from-green-400 to-green-500',
        purple: 'from-purple-400 to-purple-500',
        red: 'from-red-400 to-red-500',
        yellow: 'from-yellow-400 to-yellow-500',
        indigo: 'from-indigo-400 to-indigo-500',
        pink: 'from-pink-400 to-pink-500',
      },
    },
    section: {
      header: 'bg-gradient-to-r from-orange-50 via-red-50 to-pink-50 rounded-2xl opacity-40',
      card: 'bg-white p-6 rounded-lg shadow-sm border',
      spacing: 'space-y-8',
    },
    participant: {
      card: 'bg-white p-6 rounded-lg shadow-sm border',
      header: 'bg-gradient-to-r text-white p-4 rounded-t-lg',
      content: 'p-4 space-y-3',
    },
  },
}; 