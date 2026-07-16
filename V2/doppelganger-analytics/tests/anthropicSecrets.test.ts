/**
 * Unit tests for Anthropic secret storage (format, encrypt round-trip, public status).
 * Uses DOPPELGANGER_CONFIG_DIR so we never touch the real ~/.doppelgaenger-analytics.
 */

import { describe, test, expect, beforeEach, afterEach, jest } from '@jest/globals';
import fs from 'fs';
import os from 'os';
import path from 'path';
import * as secrets from '../dashboard/src/lib/server/anthropicSecrets.js';

describe('anthropicSecrets', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'doppel-secrets-'));
    process.env.DOPPELGANGER_CONFIG_DIR = tmpDir;
    delete process.env.ANTHROPIC_API_KEY;
  });

  afterEach(() => {
    delete process.env.DOPPELGANGER_CONFIG_DIR;
    delete process.env.ANTHROPIC_API_KEY;
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  test('rejects invalid key formats', () => {
    expect(secrets.validateAnthropicKeyFormat('').ok).toBe(false);
    expect(secrets.validateAnthropicKeyFormat('sk-openai-xyz').ok).toBe(false);
    expect(secrets.validateAnthropicKeyFormat('sk-ant- short').ok).toBe(false);
    expect(
      secrets.validateAnthropicKeyFormat('sk-ant-api03-' + 'a'.repeat(40)).ok
    ).toBe(true);
  });

  test('encrypt/decrypt round-trip', () => {
    const plain = 'sk-ant-api03-' + 'x'.repeat(40);
    const { iv, tag, data } = secrets.__testing.encrypt(plain);
    expect(secrets.__testing.decrypt(iv, tag, data)).toBe(plain);
  });

  test('maskKey never reveals middle of key', () => {
    const key = 'sk-ant-api03-' + 'abcdefghij'.repeat(5);
    const masked = secrets.__testing.maskKey(key);
    expect(masked).toContain('…');
    expect(masked).not.toContain('abcdefghij');
    expect(masked.startsWith('sk-ant-')).toBe(true);
  });

  test('public settings never expose raw key after save', async () => {
    const key = 'sk-ant-api03-' + 'b'.repeat(48);

    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => '',
    } as Response);

    const result = await secrets.saveAnthropicApiKey(key, 'claude-sonnet-5');
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(result.settings.configured).toBe(true);
    expect(result.settings.source).toBe('file');
    expect(result.settings.hint).toBeTruthy();
    expect(JSON.stringify(result.settings)).not.toContain(key);

    const loaded = secrets.getAnthropicApiKey();
    expect(loaded).toBe(key);

    const publicStatus = secrets.getPublicAnthropicSettings();
    expect(JSON.stringify(publicStatus)).not.toContain(key);

    const encPath = secrets.__testing.secretsPath();
    const disk = fs.readFileSync(encPath, 'utf8');
    expect(disk).not.toContain(key);
    expect(disk).toContain('apiKeyCipher');

    secrets.clearAnthropicApiKey();
    expect(secrets.getPublicAnthropicSettings().configured).toBe(false);
    expect(secrets.getAnthropicApiKey()).toBeNull();

    fetchSpy.mockRestore();
  });

  test('environment fallback when no file', () => {
    process.env.ANTHROPIC_API_KEY = 'sk-ant-api03-' + 'e'.repeat(40);
    const status = secrets.getPublicAnthropicSettings();
    expect(status.configured).toBe(true);
    expect(status.source).toBe('environment');
    expect(secrets.getAnthropicApiKey()).toBe(process.env.ANTHROPIC_API_KEY);
  });

  test('clearing Claude key preserves embeddings credentials', async () => {
    const anthKey = 'sk-ant-api03-' + 'c'.repeat(48);
    const embKey = 'sk-' + 'd'.repeat(48);

    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
      const url = String(input);
      if (url.includes('anthropic.com')) {
        return { ok: true, status: 200, text: async () => '', json: async () => ({}) } as Response;
      }
      // OpenAI embeddings verify
      return {
        ok: true,
        status: 200,
        json: async () => ({
          data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
        }),
      } as Response;
    });

    await secrets.saveAnthropicApiKey(anthKey);
    const embSave = await secrets.saveEmbeddingsApiKey(embKey, 'openai');
    expect(embSave.ok).toBe(true);

    secrets.clearAnthropicApiKey();
    expect(secrets.getPublicAnthropicSettings().configured).toBe(false);
    expect(secrets.getPublicEmbeddingsSettings().configured).toBe(true);
    expect(secrets.getEmbeddingApiCredentials()?.apiKey).toBe(embKey);

    secrets.clearEmbeddingsApiKey();
    expect(secrets.getPublicEmbeddingsSettings().configured).toBe(false);

    fetchSpy.mockRestore();
  });
});
