'use client';

import { createContext } from 'react';

export const ChartPlotContext = createContext<React.RefObject<HTMLDivElement | null> | null>(null);
