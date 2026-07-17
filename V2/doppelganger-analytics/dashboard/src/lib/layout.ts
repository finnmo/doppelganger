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

/** Tab fills viewport below header + tab bar on large screens. */
export const TAB_VIEWPORT =
  'flex flex-col gap-3 sm:gap-4 lg:min-h-0 lg:flex-1 lg:h-[calc(100dvh-11rem)]';

/** Grid row that splits remaining tab height evenly on large screens. */
export const ROW_FILL = 'min-h-[16rem] lg:min-h-0 lg:flex-1';

/** Chart card fills its grid cell on viewport-fit tabs. */
export const CARD_FILL = 'h-full min-h-0 overflow-hidden';

/** Chart body grows inside a fill card; scrolls when content is taller than the cell. */
export const BODY_FILL = 'flex min-h-0 flex-1 flex-col overflow-y-auto overflow-x-hidden';

/** Flexible chart plot area inside a fill card (replaces fixed h-64/h-80 in grid cells). */
export const CHART_AREA = 'chart-area w-full flex-1 basis-0 min-h-[8rem]';

/** ChartCard fullscreen body — fills the viewport and expands nested chart areas. */
export const FULLSCREEN_BODY =
  'flex min-h-0 flex-1 flex-col overflow-y-auto h-full ' +
  '[&>div]:min-h-0 [&>div.h-full]:flex-1 ' +
  '[&_.chart-area]:!min-h-[calc(100dvh-14rem)] [&_.chart-area]:!flex-1 ' +
  '[&_.h-48]:!h-[calc(100dvh-14rem)] [&_.h-64]:!h-[calc(100dvh-14rem)] ' +
  '[&_.h-72]:!h-[calc(100dvh-14rem)] [&_.h-80]:!h-[calc(100dvh-14rem)] ' +
  '[&_.h-48]:!min-h-[16rem] [&_.h-64]:!min-h-[16rem] ' +
  '[&_.h-72]:!min-h-[16rem] [&_.h-80]:!min-h-[16rem]';

/** Standard viewport-fit grid row for chart cards. */
export const CARD_GRID_ROW = (cols: string) =>
  `grid ${cols} ${GRID_GAP} ${ROW_FILL} [&>*]:min-h-0`;

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
