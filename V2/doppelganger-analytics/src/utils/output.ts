import fs from 'fs';
import path from 'path';

// Output directory for generated dashboard data. Tests redirect this via
// DOPPELGANGER_DASH_DIR so they don't clobber real output.
function outDir(): string {
  return process.env.DOPPELGANGER_DASH_DIR
    ? path.resolve(process.env.DOPPELGANGER_DASH_DIR)
    : path.resolve('dash-data');
}

export function getDashDir(): string {
  return outDir();
}

/**
 * Writes a dashboard data file into dash-data/, creating the directory if
 * needed. Output is compact JSON by default — these files are machine-read by
 * the dashboard, so whitespace is pure overhead. Pass { pretty: true } only for
 * files a human is expected to open.
 */
export function writeDashData(filename: string, data: unknown, opts: { pretty?: boolean } = {}): void {
  const dir = outDir();
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const json = opts.pretty ? JSON.stringify(data, null, 2) : JSON.stringify(data);
  fs.writeFileSync(path.join(dir, filename), json);
}
