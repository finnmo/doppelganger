/**
 * Read embedding credentials from the same encrypted secrets file the dashboard uses,
 * or from environment variables.
 */

import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';
import type { EmbeddingProvider } from './vectors.js';
import { EMBEDDING_MODELS } from './vectors.js';

interface SecretsPayload {
  version: 1;
  embeddings?: {
    provider: EmbeddingProvider;
    apiKeyCipher: string;
    iv: string;
    tag: string;
    model: string;
    hint: string;
    updatedAt: string;
  };
}

function configDir(): string {
  if (process.env.DOPPELGANGER_CONFIG_DIR) {
    return path.resolve(process.env.DOPPELGANGER_CONFIG_DIR);
  }
  return path.join(os.homedir(), '.doppelgaenger-analytics');
}

function decrypt(ivB64: string, tagB64: string, dataB64: string, masterKey: Buffer): string {
  const iv = Buffer.from(ivB64, 'base64');
  const tag = Buffer.from(tagB64, 'base64');
  const data = Buffer.from(dataB64, 'base64');
  const decipher = crypto.createDecipheriv('aes-256-gcm', masterKey, iv);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(data), decipher.final()]).toString('utf8');
}

export interface EmbeddingCredentials {
  provider: EmbeddingProvider;
  apiKey: string;
  model: string;
  source: 'file' | 'environment';
}

/**
 * Resolve embeddings API credentials.
 * Priority: encrypted file → OPENAI_API_KEY → VOYAGE_API_KEY
 */
export function getEmbeddingCredentials(): EmbeddingCredentials | null {
  const secretsPath = path.join(configDir(), 'secrets.enc.json');
  const masterKeyPath = path.join(configDir(), '.master-key');

  if (fs.existsSync(secretsPath) && fs.existsSync(masterKeyPath)) {
    try {
      const payload = JSON.parse(fs.readFileSync(secretsPath, 'utf8')) as SecretsPayload;
      const emb = payload.embeddings;
      const master = fs.readFileSync(masterKeyPath);
      if (emb?.apiKeyCipher && master.length === 32) {
        const apiKey = decrypt(emb.iv, emb.tag, emb.apiKeyCipher, master);
        const provider = emb.provider === 'voyage' ? 'voyage' : 'openai';
        return {
          provider,
          apiKey,
          model: emb.model || EMBEDDING_MODELS[provider].id,
          source: 'file',
        };
      }
    } catch {
      // fall through to env
    }
  }

  if (process.env.OPENAI_API_KEY?.trim()) {
    return {
      provider: 'openai',
      apiKey: process.env.OPENAI_API_KEY.trim(),
      model: process.env.OPENAI_EMBEDDING_MODEL?.trim() || EMBEDDING_MODELS.openai.id,
      source: 'environment',
    };
  }

  if (process.env.VOYAGE_API_KEY?.trim()) {
    return {
      provider: 'voyage',
      apiKey: process.env.VOYAGE_API_KEY.trim(),
      model: process.env.VOYAGE_EMBEDDING_MODEL?.trim() || EMBEDDING_MODELS.voyage.id,
      source: 'environment',
    };
  }

  return null;
}
