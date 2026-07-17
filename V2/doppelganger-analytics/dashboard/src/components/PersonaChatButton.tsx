'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Sparkles, X } from 'lucide-react';
import { PersonaChatPanel } from '@/components/PersonaChatPanel';

export function PersonaChatButton() {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const close = useCallback(() => setOpen(false), []);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener('keydown', onKey);
    };
  }, [open, close]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex items-center gap-1.5 rounded-md border border-purple-200 bg-purple-50 px-2.5 py-2 text-sm font-medium text-purple-700 shadow-sm transition-colors hover:bg-purple-100 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 sm:px-3"
        title="Chat with an AI simulation of someone from your messages"
      >
        <Sparkles className="h-4 w-4" />
        <span className="hidden sm:inline">Persona Chat</span>
      </button>

      {mounted &&
        open &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            className="fixed inset-0 z-[110] flex flex-col bg-white"
            role="dialog"
            aria-modal="true"
            aria-label="Persona Chat"
          >
            <header className="flex shrink-0 items-center justify-between border-b border-gray-200 px-4 py-3">
              <div className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-purple-600" />
                <h2 className="text-lg font-semibold text-gray-900">Persona Chat</h2>
              </div>
              <button
                type="button"
                onClick={close}
                className="rounded-md p-2 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-700"
                aria-label="Close Persona Chat"
              >
                <X className="h-5 w-5" />
              </button>
            </header>
            <div className="min-h-0 flex-1 overflow-hidden">
              <PersonaChatPanel />
            </div>
          </div>,
          document.body
        )}
    </>
  );
}
