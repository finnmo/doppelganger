'use client';

import React, { useCallback, useEffect, useState } from 'react';
import {
  Shield,
  Download,
  Trash2,
  X,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Lock,
} from 'lucide-react';

interface PrivacyStatus {
  dbPath: string;
  dbExists: boolean;
  messageCount: number;
  platforms: string[];
  lastImportAt: string | null;
  lastGenerateAt: string | null;
  configDir: string;
  secretsPresent: boolean;
  dashDataDirs: string[];
  wipeConfirmPhrase: string;
  notes: string[];
  stale?: boolean;
  reason?: string | null;
  error?: string;
}

interface PrivacySettingsProps {
  open: boolean;
  onClose: () => void;
}

export function PrivacySettings({ open, onClose }: PrivacySettingsProps) {
  const [status, setStatus] = useState<PrivacyStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [includeDatabase, setIncludeDatabase] = useState(false);
  const [wiping, setWiping] = useState(false);
  const [confirm, setConfirm] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(
    null
  );

  const load = useCallback(async () => {
    setLoading(true);
    setMessage(null);
    try {
      const res = await fetch('/api/privacy', { cache: 'no-store' });
      const data = (await res.json()) as PrivacyStatus;
      if (!res.ok) throw new Error(data.error || 'Failed to load privacy status');
      setStatus(data);
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Failed to load privacy status',
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    setConfirm('');
    setIncludeDatabase(false);
    void load();
  }, [open, load]);

  if (!open) return null;

  const onExport = async () => {
    setExporting(true);
    setMessage(null);
    try {
      const res = await fetch('/api/privacy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'export', includeDatabase }),
      });
      if (!res.ok) {
        const data = (await res.json()) as { error?: string };
        throw new Error(data.error || 'Export failed');
      }
      const blob = await res.blob();
      const cd = res.headers.get('Content-Disposition') || '';
      const match = /filename="([^"]+)"/.exec(cd);
      const filename = match?.[1] ?? 'doppelgaenger-export.json';
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      setMessage({
        type: 'success',
        text: includeDatabase
          ? 'Exported analytics + message database. Keep this file private.'
          : 'Exported analytics JSON (no secrets, no message database).',
      });
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Export failed',
      });
    } finally {
      setExporting(false);
    }
  };

  const onWipe = async () => {
    if (!status || confirm !== status.wipeConfirmPhrase) return;
    setWiping(true);
    setMessage(null);
    try {
      const res = await fetch('/api/privacy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'wipe', confirm }),
      });
      const data = (await res.json()) as { error?: string; ok?: boolean; note?: string };
      if (!res.ok) throw new Error(data.error || 'Wipe failed');
      setConfirm('');
      setMessage({
        type: 'success',
        text: data.note || 'Local message database and analytics files wiped.',
      });
      await load();
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Wipe failed',
      });
    } finally {
      setWiping(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <button
        type="button"
        className="absolute inset-0 bg-black/40"
        aria-label="Close"
        onClick={onClose}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="privacy-settings-title"
        className="relative z-10 flex max-h-[90vh] w-full max-w-lg flex-col overflow-hidden rounded-xl bg-white shadow-xl"
      >
        <div className="flex items-center justify-between border-b border-gray-100 px-5 py-4">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5 text-slate-700" />
            <h2 id="privacy-settings-title" className="text-lg font-semibold text-gray-900">
              Privacy & data
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
          {loading && (
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading…
            </div>
          )}

          {message && (
            <div
              className={`flex items-start gap-2 rounded-lg px-3 py-2 text-sm ${
                message.type === 'success'
                  ? 'bg-emerald-50 text-emerald-900'
                  : 'bg-red-50 text-red-900'
              }`}
            >
              {message.type === 'success' ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
              ) : (
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              )}
              {message.text}
            </div>
          )}

          {status && !loading && (
            <>
              <section className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-800">
                <div className="flex items-center gap-2 font-medium">
                  <Lock className="h-4 w-4" />
                  Local-only by default
                </div>
                <ul className="list-inside list-disc space-y-1 text-slate-700">
                  {(status.notes ?? []).map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              </section>

              <section className="space-y-1 text-sm text-gray-700">
                <p>
                  <span className="font-medium">Messages:</span>{' '}
                  {status.messageCount.toLocaleString()}
                  {status.platforms.length ? ` · ${status.platforms.join(', ')}` : ''}
                </p>
                <p className="break-all text-xs text-gray-500">DB: {status.dbPath}</p>
                <p className="break-all text-xs text-gray-500">Config: {status.configDir}</p>
                <p className="text-xs text-gray-500">
                  Encrypted API keys on disk: {status.secretsPresent ? 'yes' : 'no'}
                </p>
              </section>

              <section className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-900">Export</h3>
                <p className="text-sm text-gray-600">
                  Download analytics JSON for backup or moving machines. Secrets are never included.
                </p>
                <label className="flex items-start gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    className="mt-1"
                    checked={includeDatabase}
                    onChange={(e) => setIncludeDatabase(e.target.checked)}
                  />
                  <span>
                    Include full message database (sensitive — only if you need a complete backup)
                  </span>
                </label>
                <button
                  type="button"
                  onClick={() => void onExport()}
                  disabled={exporting}
                  className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-3 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-60"
                >
                  {exporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                  Download export
                </button>
              </section>

              <section className="space-y-3 border-t border-gray-100 pt-4">
                <h3 className="text-sm font-semibold text-red-800">Wipe local data</h3>
                <p className="text-sm text-gray-600">
                  Deletes the message database and generated analytics files. Does not delete API
                  keys. Type{' '}
                  <span className="font-mono text-xs">{status.wipeConfirmPhrase}</span> to confirm.
                </p>
                <input
                  type="text"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  placeholder={status.wipeConfirmPhrase}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
                />
                <button
                  type="button"
                  onClick={() => void onWipe()}
                  disabled={wiping || confirm !== status.wipeConfirmPhrase}
                  className="inline-flex items-center gap-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm font-medium text-red-800 hover:bg-red-100 disabled:opacity-50"
                >
                  {wiping ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                  Wipe database & analytics
                </button>
              </section>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export function PrivacySettingsButton() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
        title="Privacy & data"
      >
        <Shield className="h-4 w-4" />
        <span className="hidden sm:inline">Privacy</span>
      </button>
      <PrivacySettings open={open} onClose={() => setOpen(false)} />
    </>
  );
}
