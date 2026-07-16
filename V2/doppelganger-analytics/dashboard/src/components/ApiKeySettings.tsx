'use client';

import React, { useCallback, useEffect, useState } from 'react';
import {
  KeyRound,
  Shield,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Trash2,
  ExternalLink,
  X,
} from 'lucide-react';

interface AnthropicStatus {
  configured: boolean;
  source: 'file' | 'environment' | 'none';
  hint: string | null;
  model: string;
  updatedAt: string | null;
  models?: Array<{ id: string; label: string }>;
  error?: string;
}

interface EmbeddingsStatus {
  configured: boolean;
  source: 'file' | 'environment' | 'none';
  provider: 'openai' | 'voyage' | null;
  hint: string | null;
  model: string | null;
  updatedAt: string | null;
  providers?: Array<{ id: 'openai' | 'voyage'; label: string; model: string }>;
  error?: string;
}

interface ApiKeySettingsProps {
  open: boolean;
  onClose: () => void;
}

export function ApiKeySettings({ open, onClose }: ApiKeySettingsProps) {
  const [status, setStatus] = useState<AnthropicStatus | null>(null);
  const [embStatus, setEmbStatus] = useState<EmbeddingsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('claude-sonnet-5');
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const [embKey, setEmbKey] = useState('');
  const [embProvider, setEmbProvider] = useState<'openai' | 'voyage'>('openai');
  const [showEmbKey, setShowEmbKey] = useState(false);
  const [savingEmb, setSavingEmb] = useState(false);
  const [clearingEmb, setClearingEmb] = useState(false);
  const [embMessage, setEmbMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(
    null
  );

  const loadStatus = useCallback(async () => {
    setLoading(true);
    setMessage(null);
    setEmbMessage(null);
    try {
      const [anthRes, embRes] = await Promise.all([
        fetch('/api/settings/anthropic', { cache: 'no-store' }),
        fetch('/api/settings/embeddings', { cache: 'no-store' }),
      ]);
      const anthData = (await anthRes.json()) as AnthropicStatus;
      if (!anthRes.ok) throw new Error(anthData.error || 'Failed to load Claude settings');
      setStatus(anthData);
      setModel(anthData.model || 'claude-sonnet-5');

      const embData = (await embRes.json()) as EmbeddingsStatus;
      if (embRes.ok) {
        setEmbStatus(embData);
        if (embData.provider === 'voyage' || embData.provider === 'openai') {
          setEmbProvider(embData.provider);
        }
      }
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to load settings',
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setApiKey('');
    setShowKey(false);
    setEmbKey('');
    setShowEmbKey(false);
    void loadStatus();
  }, [open, loadStatus]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setMessage(null);
    try {
      const body: { apiKey?: string; model: string } = { model };
      if (apiKey.trim()) body.apiKey = apiKey.trim();

      if (!body.apiKey && !status?.configured) {
        setMessage({ type: 'error', text: 'Paste your Claude API key to save.' });
        setSaving(false);
        return;
      }

      const res = await fetch('/api/settings/anthropic', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Save failed');

      setStatus(data);
      setApiKey('');
      setShowKey(false);
      setMessage({
        type: 'success',
        text: body.apiKey
          ? 'API key verified with Anthropic and saved securely on this machine.'
          : 'Model preference updated.',
      });
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Save failed',
      });
    } finally {
      setSaving(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm('Remove the saved Claude API key from this machine?')) return;
    setClearing(true);
    setMessage(null);
    try {
      const res = await fetch('/api/settings/anthropic', { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Clear failed');
      setStatus(data);
      setApiKey('');
      setMessage({
        type: 'success',
        text:
          data.source === 'environment'
            ? 'Cleared saved key. ANTHROPIC_API_KEY from the environment is still active.'
            : 'Saved API key removed.',
      });
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Clear failed',
      });
    } finally {
      setClearing(false);
    }
  };

  const handleSaveEmbeddings = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingEmb(true);
    setEmbMessage(null);
    try {
      if (!embKey.trim()) {
        setEmbMessage({ type: 'error', text: 'Paste an OpenAI or Voyage API key.' });
        setSavingEmb(false);
        return;
      }
      const res = await fetch('/api/settings/embeddings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey: embKey.trim(), provider: embProvider }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Save failed');
      setEmbStatus(data);
      setEmbKey('');
      setShowEmbKey(false);
      setEmbMessage({
        type: 'success',
        text: 'Embeddings key verified and saved. Re-run generate-metrics to build the vector index.',
      });
    } catch (err) {
      setEmbMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Save failed',
      });
    } finally {
      setSavingEmb(false);
    }
  };

  const handleClearEmbeddings = async () => {
    if (!window.confirm('Remove the saved embeddings API key from this machine?')) return;
    setClearingEmb(true);
    setEmbMessage(null);
    try {
      const res = await fetch('/api/settings/embeddings', { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Clear failed');
      setEmbStatus(data);
      setEmbKey('');
      setEmbMessage({
        type: 'success',
        text:
          data.source === 'environment'
            ? 'Cleared saved key. Env OPENAI_API_KEY / VOYAGE_API_KEY still applies.'
            : 'Embeddings key removed. Persona chat will use keyword RAG.',
      });
    } catch (err) {
      setEmbMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Clear failed',
      });
    } finally {
      setClearingEmb(false);
    }
  };

  const sourceLabel =
    status?.source === 'file'
      ? 'Saved on this machine (encrypted)'
      : status?.source === 'environment'
        ? 'From ANTHROPIC_API_KEY environment variable'
        : 'Not configured';

  const embSourceLabel =
    embStatus?.source === 'file'
      ? 'Saved on this machine (encrypted)'
      : embStatus?.source === 'environment'
        ? 'From OPENAI_API_KEY / VOYAGE_API_KEY'
        : 'Not configured — keyword RAG only';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="api-key-settings-title"
        className="relative z-10 flex max-h-[90vh] w-full max-w-lg flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-xl"
      >
        <div className="flex shrink-0 items-start justify-between border-b border-gray-200 px-5 py-4">
          <div>
            <h2
              id="api-key-settings-title"
              className="flex items-center gap-2 text-lg font-semibold text-gray-900"
            >
              <KeyRound className="h-5 w-5 text-gray-700" />
              API settings
            </h2>
            <p className="mt-1 text-sm text-gray-500">
              Claude for chat; OpenAI/Voyage for vector memory. Keys stay on this computer.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-700"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-6 overflow-y-auto px-5 py-4">
          <div className="flex gap-3 p-3 rounded-lg bg-slate-50 border border-slate-200 text-sm text-slate-700">
            <Shield className="w-5 h-5 text-slate-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="font-medium text-slate-900">Security</p>
              <ul className="text-xs text-slate-600 space-y-0.5 list-disc pl-4">
                <li>Encrypted at rest under ~/.doppelgaenger-analytics/ (not in the browser)</li>
                <li>Never written to git, localStorage, or dashboard public files</li>
                <li>Verified with the provider before saving</li>
                <li>The dashboard never displays your full key after save</li>
              </ul>
            </div>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-8 text-gray-500">
              <Loader2 className="w-5 h-5 animate-spin mr-2" />
              Loading…
            </div>
          ) : (
            <>
              <section className="space-y-4">
                <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">
                  Claude (persona chat)
                </h3>

                <div className="rounded-lg border border-gray-200 p-3">
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Status
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    {status?.configured ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-amber-600" />
                    )}
                    <span className="text-sm font-medium text-gray-900">
                      {status?.configured ? 'Configured' : 'Not configured'}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{sourceLabel}</p>
                  {status?.hint && (
                    <p className="text-sm font-mono text-gray-800 mt-1">{status.hint}</p>
                  )}
                </div>

                <form onSubmit={handleSave} className="space-y-4">
                  <div>
                    <label
                      htmlFor="anthropic-api-key"
                      className="block text-sm font-medium text-gray-900 mb-1"
                    >
                      {status?.configured && status.source === 'file'
                        ? 'Replace API key'
                        : 'Claude API key'}
                    </label>
                    <div className="relative">
                      <input
                        id="anthropic-api-key"
                        type={showKey ? 'text' : 'password'}
                        autoComplete="off"
                        spellCheck={false}
                        name="anthropic-api-key"
                        placeholder={
                          status?.configured ? 'Paste a new key to replace…' : 'sk-ant-api03-…'
                        }
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                      <button
                        type="button"
                        onClick={() => setShowKey((v) => !v)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-700"
                        aria-label={showKey ? 'Hide API key' : 'Show API key'}
                      >
                        {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                    <p className="mt-1.5 text-xs text-gray-500">
                      Create a key at{' '}
                      <a
                        href="https://console.anthropic.com/settings/keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:underline inline-flex items-center gap-0.5"
                      >
                        console.anthropic.com
                        <ExternalLink className="w-3 h-3" />
                      </a>
                    </p>
                  </div>

                  <div>
                    <label
                      htmlFor="anthropic-model"
                      className="block text-sm font-medium text-gray-900 mb-1"
                    >
                      Model
                    </label>
                    <select
                      id="anthropic-model"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {(
                        status?.models ?? [
                          { id: 'claude-sonnet-5', label: 'Claude Sonnet 5 (recommended)' },
                        ]
                      ).map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  {message && (
                    <div
                      className={`flex gap-2 p-3 rounded-lg text-sm ${
                        message.type === 'success'
                          ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                          : 'bg-red-50 text-red-800 border border-red-200'
                      }`}
                    >
                      {message.type === 'success' ? (
                        <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                      ) : (
                        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                      )}
                      <span>{message.text}</span>
                    </div>
                  )}

                  <div className="flex flex-wrap items-center gap-2 pt-1">
                    <button
                      type="submit"
                      disabled={saving}
                      className="inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium text-white bg-gray-900 hover:bg-gray-800 disabled:opacity-50"
                    >
                      {saving && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                      {apiKey.trim()
                        ? 'Verify & save key'
                        : status?.configured
                          ? 'Save model'
                          : 'Verify & save key'}
                    </button>

                    {status?.source === 'file' && (
                      <button
                        type="button"
                        onClick={() => void handleClear()}
                        disabled={clearing}
                        className="inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium text-red-700 bg-red-50 hover:bg-red-100 border border-red-200 disabled:opacity-50"
                      >
                        {clearing ? (
                          <Loader2 className="w-4 h-4 animate-spin mr-2" />
                        ) : (
                          <Trash2 className="w-4 h-4 mr-2" />
                        )}
                        Remove saved key
                      </button>
                    )}
                  </div>
                </form>
              </section>

              <div className="border-t border-gray-200" />

              <section className="space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">
                    Embeddings (vector RAG)
                  </h3>
                  <p className="text-xs text-gray-500 mt-1">
                    Optional. Improves memory for paraphrases and nicknames. After saving, run{' '}
                    <code className="font-mono text-[11px] bg-gray-100 px-1 rounded">
                      npm run generate-metrics
                    </code>
                    .
                  </p>
                </div>

                <div className="rounded-lg border border-gray-200 p-3">
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                    Status
                  </div>
                  <div className="mt-1 flex items-center gap-2">
                    {embStatus?.configured ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-amber-600" />
                    )}
                    <span className="text-sm font-medium text-gray-900">
                      {embStatus?.configured ? 'Configured' : 'Not configured'}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{embSourceLabel}</p>
                  {embStatus?.hint && (
                    <p className="text-sm font-mono text-gray-800 mt-1">
                      {embStatus.provider ? `${embStatus.provider} · ` : ''}
                      {embStatus.hint}
                    </p>
                  )}
                  {embStatus?.model && (
                    <p className="text-xs text-gray-400 mt-1">Model {embStatus.model}</p>
                  )}
                </div>

                <form onSubmit={handleSaveEmbeddings} className="space-y-4">
                  <div>
                    <label
                      htmlFor="embeddings-provider"
                      className="block text-sm font-medium text-gray-900 mb-1"
                    >
                      Provider
                    </label>
                    <select
                      id="embeddings-provider"
                      value={embProvider}
                      onChange={(e) =>
                        setEmbProvider(e.target.value === 'voyage' ? 'voyage' : 'openai')
                      }
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      {(
                        embStatus?.providers ?? [
                          {
                            id: 'openai' as const,
                            label: 'OpenAI text-embedding-3-small',
                            model: 'text-embedding-3-small',
                          },
                          {
                            id: 'voyage' as const,
                            label: 'Voyage voyage-3-lite',
                            model: 'voyage-3-lite',
                          },
                        ]
                      ).map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label
                      htmlFor="embeddings-api-key"
                      className="block text-sm font-medium text-gray-900 mb-1"
                    >
                      {embStatus?.configured && embStatus.source === 'file'
                        ? 'Replace embeddings key'
                        : 'Embeddings API key'}
                    </label>
                    <div className="relative">
                      <input
                        id="embeddings-api-key"
                        type={showEmbKey ? 'text' : 'password'}
                        autoComplete="off"
                        spellCheck={false}
                        placeholder={embProvider === 'voyage' ? 'pa-…' : 'sk-…'}
                        value={embKey}
                        onChange={(e) => setEmbKey(e.target.value)}
                        className="w-full px-3 py-2 pr-10 border border-gray-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                      <button
                        type="button"
                        onClick={() => setShowEmbKey((v) => !v)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-700"
                        aria-label={showEmbKey ? 'Hide API key' : 'Show API key'}
                      >
                        {showEmbKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>

                  {embMessage && (
                    <div
                      className={`flex gap-2 p-3 rounded-lg text-sm ${
                        embMessage.type === 'success'
                          ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                          : 'bg-red-50 text-red-800 border border-red-200'
                      }`}
                    >
                      {embMessage.type === 'success' ? (
                        <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                      ) : (
                        <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                      )}
                      <span>{embMessage.text}</span>
                    </div>
                  )}

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="submit"
                      disabled={savingEmb}
                      className="inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium text-white bg-gray-900 hover:bg-gray-800 disabled:opacity-50"
                    >
                      {savingEmb && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                      Verify & save embeddings key
                    </button>
                    {embStatus?.source === 'file' && (
                      <button
                        type="button"
                        onClick={() => void handleClearEmbeddings()}
                        disabled={clearingEmb}
                        className="inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium text-red-700 bg-red-50 hover:bg-red-100 border border-red-200 disabled:opacity-50"
                      >
                        {clearingEmb ? (
                          <Loader2 className="w-4 h-4 animate-spin mr-2" />
                        ) : (
                          <Trash2 className="w-4 h-4 mr-2" />
                        )}
                        Remove
                      </button>
                    )}
                  </div>
                </form>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/** Compact status + open button for headers. */
export function ApiKeySettingsButton() {
  const [open, setOpen] = useState(false);
  const [configured, setConfigured] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/settings/anthropic', { cache: 'no-store' });
        if (!res.ok) return;
        const data = (await res.json()) as AnthropicStatus;
        if (!cancelled) setConfigured(Boolean(data.configured));
      } catch {
        // ignore — button still works
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        title="API settings"
      >
        <KeyRound className="w-4 h-4" />
        <span className="hidden sm:inline">API key</span>
        {configured === true && (
          <span className="w-2 h-2 rounded-full bg-emerald-500" aria-label="Configured" />
        )}
        {configured === false && (
          <span className="w-2 h-2 rounded-full bg-amber-400" aria-label="Not configured" />
        )}
      </button>
      <ApiKeySettings open={open} onClose={() => setOpen(false)} />
    </>
  );
}
