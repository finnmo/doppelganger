/**
 * Prefer the open ChartCard fullscreen dialog as the portal host so hover
 * tooltips stack above the modal backdrop. Falls back to document.body.
 */
export function getChartTooltipHost(): HTMLElement {
  if (typeof document === 'undefined') {
    throw new Error('getChartTooltipHost requires a browser document');
  }
  return (
    document.querySelector<HTMLElement>('[data-chart-fullscreen-root]') ??
    document.body
  );
}
