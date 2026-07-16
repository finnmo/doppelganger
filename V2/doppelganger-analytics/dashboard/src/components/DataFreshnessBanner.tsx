'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface PipelineStatus {
  stale: boolean;
  reason: string | null;
  generateHint: string;
  lastImportAt: string | null;
  lastGenerateAt: string | null;
  lastImportSources: string[];
}

export function DataFreshnessBanner() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch('/api/pipeline/status', { cache: 'no-store' });
      if (!res.ok) return;
      const data = (await res.json()) as PipelineStatus;
      setStatus(data);
    } catch {
      // ignore — banner is optional
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (!status?.stale || !status.reason) return null;

  return (
    <div
      className="mb-4 flex flex-col gap-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-3 text-sm text-amber-950 sm:flex-row sm:items-start sm:px-4"
      role="status"
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div className="min-w-0 flex-1">
          <p className="font-medium">Analytics may be out of date</p>
          <p className="mt-0.5 text-amber-900/90">{status.reason}</p>
          <p className="mt-1 break-all font-mono text-xs text-amber-800">
            {status.generateHint}
            {status.lastImportSources?.length
              ? ` · last import: ${status.lastImportSources.join(', ')}`
              : ''}
          </p>
        </div>
      </div>
      <button
        type="button"
        onClick={() => void load()}
        className="inline-flex shrink-0 items-center justify-center gap-1 self-start rounded-md border border-amber-300 bg-white px-2 py-1 text-xs font-medium text-amber-900 hover:bg-amber-100 sm:self-auto"
        title="Refresh status"
      >
        <RefreshCw className="h-3 w-3" />
        Refresh
      </button>
    </div>
  );
}
