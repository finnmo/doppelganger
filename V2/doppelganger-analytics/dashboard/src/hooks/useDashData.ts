'use client';

import { useEffect, useState } from 'react';

// Module-level cache keyed by filename so multiple components (and repeated
// tab switches) share a single fetch per data file instead of re-fetching.
const cache = new Map<string, Promise<unknown>>();

function loadFile(file: string): Promise<unknown> {
  let pending = cache.get(file);
  if (!pending) {
    pending = fetch(`/data/${file}`).then(response => {
      if (!response.ok) {
        throw new Error(`Failed to load ${file}: ${response.status}`);
      }
      return response.json();
    });
    cache.set(file, pending);
  }
  return pending;
}

export interface DashDataState<T> {
  data: T | null;
  isLoading: boolean;
  error: Error | null;
}

/**
 * Loads a JSON file from /data/, cached across the app. Returns loading and
 * error state so components don't each reimplement fetch boilerplate.
 */
export function useDashData<T>(file: string): DashDataState<T> {
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let active = true;
    setIsLoading(true);

    loadFile(file)
      .then(result => {
        if (active) {
          setData(result as T);
          setError(null);
        }
      })
      .catch(err => {
        if (active) {
          setError(err instanceof Error ? err : new Error(String(err)));
          // Drop the cached rejection so a later mount can retry.
          cache.delete(file);
        }
      })
      .finally(() => {
        if (active) setIsLoading(false);
      });

    return () => {
      active = false;
    };
  }, [file]);

  return { data, isLoading, error };
}
