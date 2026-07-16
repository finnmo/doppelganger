import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const moduleDir = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(moduleDir, '../src');

function listTsFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listTsFiles(full));
    else if (entry.name.endsWith('.ts')) out.push(full);
  }
  return out;
}

/**
 * Every dashboard data file should be produced by exactly one source file. A
 * second writer silently overwriting the first is the class of bug that this
 * guards against.
 */
describe('dash-data output writers', () => {
  test('each output filename is written by exactly one source file', () => {
    const writers = new Map<string, Set<string>>();
    const patterns = [
      /writeDashData\(\s*['"`]([\w.-]+\.json)['"`]/g,
      /writeFileSync\([^)]*['"`]([\w./-]+\.json)['"`]/g
    ];

    for (const file of listTsFiles(SRC)) {
      const content = fs.readFileSync(file, 'utf-8');
      for (const pattern of patterns) {
        for (const match of content.matchAll(pattern)) {
          const name = path.basename(match[1]);
          if (!writers.has(name)) writers.set(name, new Set());
          writers.get(name)!.add(path.relative(SRC, file));
        }
      }
    }

    const duplicates = [...writers.entries()].filter(([, files]) => files.size > 1);
    const message = duplicates.map(([name, files]) => `${name}: ${[...files].join(', ')}`).join('\n');
    expect(message).toBe('');
  });
});
