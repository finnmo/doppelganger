'use client';

import React from 'react';

const CHIP_COLORS = [
  'bg-blue-500',
  'bg-emerald-500',
  'bg-amber-500',
  'bg-rose-500',
  'bg-violet-500',
  'bg-cyan-500',
  'bg-orange-500',
  'bg-lime-500',
  'bg-pink-500',
  'bg-slate-500',
];

export function participantColor(name: string, participants: string[]): string {
  const idx = participants.indexOf(name);
  return CHIP_COLORS[(idx >= 0 ? idx : name.length) % CHIP_COLORS.length];
}

interface ParticipantToggleChipsProps {
  participants: string[];
  active: Set<string>;
  onToggle: (participant: string) => void;
  className?: string;
}

/**
 * Color-dotted chips to toggle participant visibility in multi-series charts.
 * Default: all participants active (parent initializes active = all).
 */
export function ParticipantToggleChips({
  participants,
  active,
  onToggle,
  className = '',
}: ParticipantToggleChipsProps) {
  if (participants.length <= 1) return null;

  return (
    <div className={`flex flex-wrap items-center gap-1.5 ${className}`}>
      {participants.map((name) => {
        const on = active.has(name);
        const dot = participantColor(name, participants);
        return (
          <button
            key={name}
            type="button"
            onClick={() => onToggle(name)}
            className={`inline-flex max-w-[10rem] items-center gap-1.5 truncate rounded-full border px-2 py-0.5 text-xs font-medium transition-colors ${
              on
                ? 'border-gray-300 bg-white text-gray-800 shadow-sm'
                : 'border-transparent bg-gray-100 text-gray-400'
            }`}
            title={on ? `Hide ${name}` : `Show ${name}`}
            aria-pressed={on}
          >
            <span className={`h-2 w-2 shrink-0 rounded-full ${on ? dot : 'bg-gray-300'}`} />
            <span className="truncate">{name}</span>
          </button>
        );
      })}
    </div>
  );
}

/** Initialize all participants as active. */
export function allParticipantsActive(names: string[]): Set<string> {
  return new Set(names);
}

/** Toggle one chip; never leave zero active — re-enable all if last is turned off. */
export function toggleParticipant(active: Set<string>, name: string, all: string[]): Set<string> {
  const next = new Set(active);
  if (next.has(name)) {
    next.delete(name);
    if (next.size === 0) return allParticipantsActive(all);
    return next;
  }
  next.add(name);
  return next;
}
