/** Shared responsive layout tokens for the dashboard shell. */

/** Wide desktop content rail — uses more horizontal space than max-w-7xl. */
export const PAGE_SHELL =
  'mx-auto w-full max-w-[1600px] px-3 sm:px-6 lg:px-8 xl:px-10';

/** Horizontal gap for toolbars that must wrap on small screens. */
export const TOOLBAR_ROW =
  'flex flex-wrap items-center gap-2 sm:gap-3';

/** Page / tab titles that stay readable on phones. */
export const PAGE_TITLE =
  'text-2xl sm:text-3xl font-bold text-gray-900 flex flex-wrap items-center gap-2 mb-1';

/* ------------------------------------------------------------------ */
/* Dense dashboard tokens — every tab lays out as a single-screen grid */
/* ------------------------------------------------------------------ */

/** Vertical rhythm between grid rows/sections inside a tab. */
export const TAB_STACK = 'flex flex-col gap-3 sm:gap-4';

/** Standard dense grid gap. */
export const GRID_GAP = 'gap-3 sm:gap-4';

/** Fixed chart body heights (ResponsiveContainer fills them). */
export const CHART_SM = 'h-48';
export const CHART_MD = 'h-64';
export const CHART_LG = 'h-72';

/** Scrollable list body inside a card, so long lists never grow the page. */
export const SCROLL_SM = 'max-h-48 overflow-y-auto';
export const SCROLL_MD = 'max-h-60 overflow-y-auto';
export const SCROLL_LG = 'max-h-72 overflow-y-auto';
