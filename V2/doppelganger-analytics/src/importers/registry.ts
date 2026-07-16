// Registry of platform importers. Detection runs in order; the first importer
// whose layout matches the export wins. Put the most specific signatures first.

import type { PlatformImporter } from './types.js';
import { imessageImporter } from './imessage.js';
import { whatsappImporter } from './whatsapp.js';
import { metaJsonImporter } from './metaJson.js';

export const IMPORTERS: PlatformImporter[] = [
  imessageImporter,
  whatsappImporter,
  metaJsonImporter
];

/** Returns the importer whose layout matches the extracted export, or null. */
export async function detectPlatform(rootDir: string): Promise<PlatformImporter | null> {
  for (const importer of IMPORTERS) {
    if (await importer.detect(rootDir)) {
      return importer;
    }
  }
  return null;
}

export function supportedPlatforms(): string {
  return IMPORTERS.map(i => i.displayName).join(', ');
}
