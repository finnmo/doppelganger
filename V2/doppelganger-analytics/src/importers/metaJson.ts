// Unified Meta JSON importer — Instagram and Facebook Messenger share the
// same export format from "Download Your Information".

import type { NormalizedConversation, PlatformImporter } from './types.js';
import { detectMetaJsonExport, parseMetaExport } from './meta/shared.js';

export const metaJsonImporter: PlatformImporter = {
  id: 'meta',
  displayName: 'Instagram / Messenger (Meta JSON)',

  async detect(rootDir: string): Promise<boolean> {
    return detectMetaJsonExport(rootDir);
  },

  async parse(rootDir: string): Promise<NormalizedConversation[]> {
    const parsed = await parseMetaExport(rootDir);
    return parsed.map(({ id, messages, source }) => ({ id, messages, source }));
  }
};

/** Per-conversation source tag (instagram vs messenger) for the DB `source` column. */
export async function parseMetaExportWithSource(rootDir: string) {
  return parseMetaExport(rootDir);
}

export type { MetaRawMessage } from './meta/shared.js';
