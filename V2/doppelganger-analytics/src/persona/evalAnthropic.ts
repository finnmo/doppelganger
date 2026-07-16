/**
 * Resolve Anthropic credentials for CLI persona eval (env or encrypted secrets file).
 */

import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

interface SecretsPayload {
  version: 1;
  anthropic?: {
    apiKeyCipher: string;
    iv: string;
    tag: string;
    model: string;
  };
}

function configDir(): string {
  if (process.env.DOPPELGANGER_CONFIG_DIR) {
    return path.resolve(process.env.DOPPELGANGER_CONFIG_DIR);
  }
  return path.join(os.homedir(), '.doppelgaenger-analytics');
}

export function getAnthropicCredentialsForEval(): { apiKey: string; model: string } | null {
  const envKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (envKey) {
    return {
      apiKey: envKey,
      model: process.env.ANTHROPIC_MODEL?.trim() || 'claude-sonnet-5',
    };
  }

  const secretsPath = path.join(configDir(), 'secrets.enc.json');
  const masterKeyPath = path.join(configDir(), '.master-key');
  if (!fs.existsSync(secretsPath) || !fs.existsSync(masterKeyPath)) return null;

  try {
    const payload = JSON.parse(fs.readFileSync(secretsPath, 'utf8')) as SecretsPayload;
    const anth = payload.anthropic;
    const master = fs.readFileSync(masterKeyPath);
    if (!anth?.apiKeyCipher || master.length !== 32) return null;

    const iv = Buffer.from(anth.iv, 'base64');
    const tag = Buffer.from(anth.tag, 'base64');
    const data = Buffer.from(anth.apiKeyCipher, 'base64');
    const decipher = crypto.createDecipheriv('aes-256-gcm', master, iv);
    decipher.setAuthTag(tag);
    const apiKey = Buffer.concat([decipher.update(data), decipher.final()]).toString('utf8');
    return { apiKey, model: anth.model || 'claude-sonnet-5' };
  } catch {
    return null;
  }
}
