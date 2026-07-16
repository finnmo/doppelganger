// Locates a messaging export for the one-command `start` flow.
// Priority: explicit --data → scan project root for known export shapes.

import fs from 'fs';
import path from 'path';

const EXPORT_NAMES = new Set([
  '_chat.txt',
  'chat.db',
  'Chat.db',
  'messages.db'
]);

/** True when `dir` looks like an extracted Meta/Instagram/Messenger tree. */
function isMetaMessagesTree(dir: string): boolean {
  const inbox = path.join(dir, 'messages', 'inbox');
  if (!fs.existsSync(inbox)) return false;
  try {
    const entries = fs.readdirSync(inbox, { withFileTypes: true });
    return entries.some(e => e.isDirectory());
  } catch {
    return false;
  }
}

/** True when `dir` contains a WhatsApp plain-text export. */
function isWhatsAppTree(dir: string): boolean {
  if (fs.existsSync(path.join(dir, '_chat.txt'))) return true;
  try {
    return fs.readdirSync(dir).some(f => f.endsWith('.txt') && f.toLowerCase().includes('chat'));
  } catch {
    return false;
  }
}

const IMESSAGE_DB_NAMES = ['chat.db', 'Chat.db', 'messages.db'];

function isIMessageTree(dir: string): boolean {
  return IMESSAGE_DB_NAMES.some((name) => fs.existsSync(path.join(dir, name)));
}

function scoreCandidate(filePath: string): number {
  const name = path.basename(filePath).toLowerCase();
  let score = fs.statSync(filePath).mtimeMs;
  if (name.endsWith('.zip')) score += 1_000_000;
  if (name === 'chat.db' || name === '_chat.txt') score += 500_000;
  if (name.includes('instagram') || name.includes('facebook') || name.includes('whatsapp')) {
    score += 100_000;
  }
  return score;
}

/**
 * Resolve one or more export paths. Explicit `--data` wins; otherwise returns
 * all distinct candidates in the project root (Instagram + WhatsApp + iMessage).
 */
export function resolveDataPaths(
  explicit: string | string[] | undefined,
  searchRoot = process.cwd()
): string[] {
  if (explicit) {
    const list = Array.isArray(explicit) ? explicit : [explicit];
    return list.map((p) => {
      const resolved = path.resolve(p);
      if (!fs.existsSync(resolved)) {
        throw new Error(`Data path not found: ${resolved}`);
      }
      return resolved;
    });
  }

  const candidates: string[] = [];

  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(searchRoot, { withFileTypes: true });
  } catch {
    return [];
  }

  for (const entry of entries) {
    const full = path.join(searchRoot, entry.name);
    if (entry.isFile()) {
      const lower = entry.name.toLowerCase();
      if (lower.endsWith('.zip')) candidates.push(full);
      if (EXPORT_NAMES.has(entry.name) || lower === '_chat.txt' || lower.endsWith('chat.db')) {
        candidates.push(full);
      }
    } else if (entry.isDirectory()) {
      if (isMetaMessagesTree(full) || isWhatsAppTree(full) || isIMessageTree(full)) {
        candidates.push(full);
      }
      try {
        for (const sub of fs.readdirSync(full, { withFileTypes: true })) {
          if (!sub.isDirectory() && !sub.isFile()) continue;
          const subPath = path.join(full, sub.name);
          if (sub.isFile()) {
            const lower = sub.name.toLowerCase();
            if (lower === 'chat.db' || lower === '_chat.txt') candidates.push(subPath);
          } else if (isMetaMessagesTree(subPath) || isWhatsAppTree(subPath) || isIMessageTree(subPath)) {
            candidates.push(subPath);
          }
        }
      } catch {
        // ignore unreadable subdirs
      }
    }
  }

  // Deduplicate and prefer distinct platform kinds (don't import same ZIP twice).
  const unique = [...new Set(candidates)];
  unique.sort((a, b) => scoreCandidate(b) - scoreCandidate(a));
  return unique;
}

/**
 * Resolve the export path from an explicit `--data` value or by scanning
 * `searchRoot` (defaults to cwd). Returns null when nothing is found.
 */
export function resolveDataPath(explicit: string | undefined, searchRoot = process.cwd()): string | null {
  const paths = resolveDataPaths(explicit, searchRoot);
  return paths[0] ?? null;
}

export function describeDataExpectations(): string {
  return [
    'Put your export(s) in the project root (or pass --data=):',
    '  • Meta ZIP or folder (Instagram / Messenger JSON)',
    '  • WhatsApp ZIP or folder with _chat.txt',
    '  • iMessage chat.db (copy from ~/Library/Messages/chat.db)',
    'Multiple exports can coexist — each import keeps other platforms.',
  ].join('\n');
}
