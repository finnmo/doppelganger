/**
 * Server-only Anthropic credential storage.
 *
 * Security model:
 * - Key never stored in localStorage / sessionStorage / cookies
 * - Encrypted at rest (AES-256-GCM) under ~/.doppelgaenger-analytics/
 * - Directory 0700, files 0600
 * - API routes only ever return { configured, hint, source } — never the raw key
 * - Prefer file store; fall back to ANTHROPIC_API_KEY env for Docker/CI
 */

import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

import type { EmbeddingProvider } from './embeddingsVectors';
import { EMBEDDING_MODELS } from './embeddingsVectors';

export const DEFAULT_ANTHROPIC_MODEL = 'claude-sonnet-5';

export const ANTHROPIC_MODELS = [
  { id: 'claude-sonnet-5', label: 'Claude Sonnet 5 (recommended)' },
  { id: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6' },
  { id: 'claude-opus-4-8', label: 'Claude Opus 4.8' },
  { id: 'claude-haiku-4-5', label: 'Claude Haiku 4.5 (faster/cheaper)' },
] as const;

const CONFIG_DIR_NAME = '.doppelgaenger-analytics';
const MASTER_KEY_FILE = '.master-key';
const SECRETS_FILE = 'secrets.enc.json';

/** Anthropic keys look like sk-ant-api03-… */
const KEY_PATTERN = /^sk-ant-[a-zA-Z0-9_-]{20,}$/;

export type CredentialSource = 'file' | 'environment' | 'none';

export interface AnthropicSettingsPublic {
  configured: boolean;
  source: CredentialSource;
  /** Masked hint only, e.g. sk-ant-…xxxx — never the full key */
  hint: string | null;
  model: string;
  updatedAt: string | null;
}

export interface EmbeddingsSettingsPublic {
  configured: boolean;
  source: CredentialSource;
  provider: EmbeddingProvider | null;
  hint: string | null;
  model: string | null;
  updatedAt: string | null;
}

interface SecretsPayload {
  version: 1;
  anthropic?: {
    apiKeyCipher: string;
    iv: string;
    tag: string;
    model: string;
    hint: string;
    updatedAt: string;
  };
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
  const override = process.env.DOPPELGANGER_CONFIG_DIR;
  if (override) return path.resolve(override);
  return path.join(os.homedir(), CONFIG_DIR_NAME);
}

function masterKeyPath(): string {
  return path.join(configDir(), MASTER_KEY_FILE);
}

function secretsPath(): string {
  return path.join(configDir(), SECRETS_FILE);
}

function ensureSecureDir(): void {
  const dir = configDir();
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  try {
    fs.chmodSync(dir, 0o700);
  } catch {
    // Windows may not support chmod the same way
  }
}

function readOrCreateMasterKey(): Buffer {
  ensureSecureDir();
  const keyPath = masterKeyPath();
  if (fs.existsSync(keyPath)) {
    const buf = fs.readFileSync(keyPath);
    if (buf.length !== 32) {
      throw new Error('Corrupt master key file — delete ~/.doppelgaenger-analytics/.master-key and re-save your API key.');
    }
    return buf;
  }
  const key = crypto.randomBytes(32);
  fs.writeFileSync(keyPath, key, { mode: 0o600 });
  try {
    fs.chmodSync(keyPath, 0o600);
  } catch {
    // ignore
  }
  return key;
}

function encrypt(plaintext: string): { iv: string; tag: string; data: string } {
  const key = readOrCreateMasterKey();
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  const enc = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return {
    iv: iv.toString('base64'),
    tag: tag.toString('base64'),
    data: enc.toString('base64'),
  };
}

function decrypt(ivB64: string, tagB64: string, dataB64: string): string {
  const key = readOrCreateMasterKey();
  const iv = Buffer.from(ivB64, 'base64');
  const tag = Buffer.from(tagB64, 'base64');
  const data = Buffer.from(dataB64, 'base64');
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(data), decipher.final()]).toString('utf8');
}

function maskKey(apiKey: string): string {
  if (apiKey.length < 12) return '••••';
  return `${apiKey.slice(0, 7)}…${apiKey.slice(-4)}`;
}

export function validateAnthropicKeyFormat(apiKey: string): { ok: true } | { ok: false; error: string } {
  const trimmed = apiKey.trim();
  if (!trimmed) return { ok: false, error: 'API key is required.' };
  if (/\s/.test(trimmed)) return { ok: false, error: 'API key must not contain spaces.' };
  if (!trimmed.startsWith('sk-ant-')) {
    return { ok: false, error: 'Claude API keys start with sk-ant-. Check you copied the key from console.anthropic.com.' };
  }
  if (!KEY_PATTERN.test(trimmed)) {
    return { ok: false, error: 'API key format looks invalid.' };
  }
  if (trimmed.length > 256) {
    return { ok: false, error: 'API key is unexpectedly long.' };
  }
  return { ok: true };
}

function readSecretsFile(): SecretsPayload | null {
  const p = secretsPath();
  if (!fs.existsSync(p)) return null;
  try {
    const raw = fs.readFileSync(p, 'utf8');
    return JSON.parse(raw) as SecretsPayload;
  } catch {
    return null;
  }
}

function writeSecretsFile(payload: SecretsPayload): void {
  ensureSecureDir();
  const p = secretsPath();
  const tmp = `${p}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(payload, null, 2), { mode: 0o600 });
  try {
    fs.chmodSync(tmp, 0o600);
  } catch {
    // ignore
  }
  fs.renameSync(tmp, p);
  try {
    fs.chmodSync(p, 0o600);
  } catch {
    // ignore
  }
}

/** Live check against Anthropic — does not persist. */
export async function verifyAnthropicApiKey(apiKey: string): Promise<{ ok: true } | { ok: false; error: string }> {
  const format = validateAnthropicKeyFormat(apiKey);
  if (!format.ok) return format;

  try {
    const res = await fetch('https://api.anthropic.com/v1/models', {
      method: 'GET',
      headers: {
        'x-api-key': apiKey.trim(),
        'anthropic-version': '2023-06-01',
      },
      signal: AbortSignal.timeout(15_000),
    });

    if (res.status === 401 || res.status === 403) {
      return { ok: false, error: 'Anthropic rejected this key (unauthorized). Create a new key at console.anthropic.com.' };
    }
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      // Never echo key material; truncate body
      const snippet = body.slice(0, 120).replace(/sk-ant-[a-zA-Z0-9_-]+/g, '[redacted]');
      return { ok: false, error: `Anthropic API error (${res.status})${snippet ? `: ${snippet}` : ''}` };
    }
    return { ok: true };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, error: `Could not reach Anthropic to verify the key: ${msg}` };
  }
}

export function getPublicAnthropicSettings(): AnthropicSettingsPublic {
  const file = readSecretsFile();
  if (file?.anthropic?.apiKeyCipher) {
    return {
      configured: true,
      source: 'file',
      hint: file.anthropic.hint,
      model: file.anthropic.model || DEFAULT_ANTHROPIC_MODEL,
      updatedAt: file.anthropic.updatedAt,
    };
  }

  const envKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (envKey) {
    return {
      configured: true,
      source: 'environment',
      hint: maskKey(envKey),
      model: process.env.ANTHROPIC_MODEL?.trim() || DEFAULT_ANTHROPIC_MODEL,
      updatedAt: null,
    };
  }

  return {
    configured: false,
    source: 'none',
    hint: null,
    model: DEFAULT_ANTHROPIC_MODEL,
    updatedAt: null,
  };
}

/**
 * Decrypt and return the API key for server-side Anthropic calls only.
 * Never pass this to the client.
 */
export function getAnthropicApiKey(): string | null {
  const file = readSecretsFile();
  if (file?.anthropic?.apiKeyCipher) {
    try {
      return decrypt(file.anthropic.iv, file.anthropic.tag, file.anthropic.apiKeyCipher);
    } catch {
      throw new Error(
        'Failed to decrypt stored API key. The master key may have been rotated — clear and re-save your key in Settings.'
      );
    }
  }
  const envKey = process.env.ANTHROPIC_API_KEY?.trim();
  return envKey || null;
}

export function getAnthropicModel(): string {
  const file = readSecretsFile();
  if (file?.anthropic?.model) return file.anthropic.model;
  return process.env.ANTHROPIC_MODEL?.trim() || DEFAULT_ANTHROPIC_MODEL;
}

export async function saveAnthropicApiKey(
  apiKey: string,
  model?: string
): Promise<{ ok: true; settings: AnthropicSettingsPublic } | { ok: false; error: string }> {
  const format = validateAnthropicKeyFormat(apiKey);
  if (!format.ok) return format;

  const verified = await verifyAnthropicApiKey(apiKey);
  if (!verified.ok) return verified;

  const chosenModel = (model?.trim() || DEFAULT_ANTHROPIC_MODEL);
  const allowed = ANTHROPIC_MODELS.some((m) => m.id === chosenModel);
  if (!allowed && !/^claude-/.test(chosenModel)) {
    return { ok: false, error: 'Unrecognized model id.' };
  }

  const { iv, tag, data } = encrypt(apiKey.trim());
  const existing = readSecretsFile() ?? { version: 1 as const };
  const updatedAt = new Date().toISOString();
  const hint = maskKey(apiKey.trim());

  writeSecretsFile({
    ...existing,
    version: 1,
    anthropic: {
      apiKeyCipher: data,
      iv,
      tag,
      model: chosenModel,
      hint,
      updatedAt,
    },
  });

  return { ok: true, settings: getPublicAnthropicSettings() };
}

export function clearAnthropicApiKey(): AnthropicSettingsPublic {
  const existing = readSecretsFile();
  if (existing) {
    const next: SecretsPayload = { version: 1 };
    if (existing.embeddings) next.embeddings = existing.embeddings;
    writeSecretsFile(next);
  }
  return getPublicAnthropicSettings();
}

export function saveAnthropicModelOnly(model: string): AnthropicSettingsPublic {
  const existing = readSecretsFile();
  if (!existing?.anthropic) {
    throw new Error('Save an API key before changing the model, or set ANTHROPIC_MODEL in the environment.');
  }
  existing.anthropic.model = model.trim() || DEFAULT_ANTHROPIC_MODEL;
  existing.anthropic.updatedAt = new Date().toISOString();
  writeSecretsFile(existing);
  return getPublicAnthropicSettings();
}

export function getPublicEmbeddingsSettings(): EmbeddingsSettingsPublic {
  const file = readSecretsFile();
  if (file?.embeddings?.apiKeyCipher) {
    const provider = file.embeddings.provider === 'voyage' ? 'voyage' : 'openai';
    return {
      configured: true,
      source: 'file',
      provider,
      hint: file.embeddings.hint,
      model: file.embeddings.model || EMBEDDING_MODELS[provider].id,
      updatedAt: file.embeddings.updatedAt,
    };
  }
  if (process.env.OPENAI_API_KEY?.trim()) {
    return {
      configured: true,
      source: 'environment',
      provider: 'openai',
      hint: maskKey(process.env.OPENAI_API_KEY.trim()),
      model: process.env.OPENAI_EMBEDDING_MODEL?.trim() || EMBEDDING_MODELS.openai.id,
      updatedAt: null,
    };
  }
  if (process.env.VOYAGE_API_KEY?.trim()) {
    return {
      configured: true,
      source: 'environment',
      provider: 'voyage',
      hint: maskKey(process.env.VOYAGE_API_KEY.trim()),
      model: process.env.VOYAGE_EMBEDDING_MODEL?.trim() || EMBEDDING_MODELS.voyage.id,
      updatedAt: null,
    };
  }
  return {
    configured: false,
    source: 'none',
    provider: null,
    hint: null,
    model: null,
    updatedAt: null,
  };
}

export function getEmbeddingApiCredentials(): {
  provider: EmbeddingProvider;
  apiKey: string;
  model: string;
} | null {
  const file = readSecretsFile();
  if (file?.embeddings?.apiKeyCipher) {
    const provider = file.embeddings.provider === 'voyage' ? 'voyage' : 'openai';
    return {
      provider,
      apiKey: decrypt(file.embeddings.iv, file.embeddings.tag, file.embeddings.apiKeyCipher),
      model: file.embeddings.model || EMBEDDING_MODELS[provider].id,
    };
  }
  if (process.env.OPENAI_API_KEY?.trim()) {
    return {
      provider: 'openai',
      apiKey: process.env.OPENAI_API_KEY.trim(),
      model: process.env.OPENAI_EMBEDDING_MODEL?.trim() || EMBEDDING_MODELS.openai.id,
    };
  }
  if (process.env.VOYAGE_API_KEY?.trim()) {
    return {
      provider: 'voyage',
      apiKey: process.env.VOYAGE_API_KEY.trim(),
      model: process.env.VOYAGE_EMBEDDING_MODEL?.trim() || EMBEDDING_MODELS.voyage.id,
    };
  }
  return null;
}

export async function saveEmbeddingsApiKey(
  apiKey: string,
  provider: EmbeddingProvider,
  model?: string
): Promise<{ ok: true; settings: EmbeddingsSettingsPublic } | { ok: false; error: string }> {
  const trimmed = apiKey.trim();
  if (!trimmed || /\s/.test(trimmed)) {
    return { ok: false, error: 'API key is required and must not contain spaces.' };
  }
  if (trimmed.length < 20) {
    return { ok: false, error: 'API key looks too short.' };
  }

  const chosenProvider = provider === 'voyage' ? 'voyage' : 'openai';
  const chosenModel = model?.trim() || EMBEDDING_MODELS[chosenProvider].id;

  // Light verify: embed a tiny string
  try {
    const { embedTexts } = await import('./embeddingsClient');
    await embedTexts(['ping'], {
      provider: chosenProvider,
      apiKey: trimmed,
      model: chosenModel,
    });
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : 'Embeddings provider rejected this key.',
    };
  }

  const { iv, tag, data } = encrypt(trimmed);
  const existing = readSecretsFile() ?? { version: 1 as const };
  writeSecretsFile({
    ...existing,
    version: 1,
    embeddings: {
      provider: chosenProvider,
      apiKeyCipher: data,
      iv,
      tag,
      model: chosenModel,
      hint: maskKey(trimmed),
      updatedAt: new Date().toISOString(),
    },
  });

  return { ok: true, settings: getPublicEmbeddingsSettings() };
}

export function clearEmbeddingsApiKey(): EmbeddingsSettingsPublic {
  const existing = readSecretsFile();
  if (existing) {
    const next: SecretsPayload = { version: 1 };
    if (existing.anthropic) next.anthropic = existing.anthropic;
    writeSecretsFile(next);
  }
  return getPublicEmbeddingsSettings();
}

/** Test helpers — only used from unit tests via DOPPELGANGER_CONFIG_DIR. */
export const __testing = {
  encrypt,
  decrypt,
  maskKey,
  configDir,
  secretsPath,
  masterKeyPath,
};
