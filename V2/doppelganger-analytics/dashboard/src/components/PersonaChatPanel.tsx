'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Bot,
  KeyRound,
  Loader2,
  Send,
  Sparkles,
  Trash2,
  AlertCircle,
  User,
} from 'lucide-react';
import { ApiKeySettings } from '@/components/ApiKeySettings';
import { PlatformBadge } from '@/components/PlatformBadge';
import { useConversationFilter } from '@/contexts/ConversationContext';

interface PersonaSummary {
  sender: string;
  messageCount: number;
  styleSummary: string;
  exampleCount: number;
  sources: string[];
  avgWordsPerMessage: number;
  avgEmojiPerMessage: number;
  responsivenessLabel: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export function PersonaChatPanel() {
  const { selectedConversations, conversations } = useConversationFilter();
  const [profiles, setProfiles] = useState<PersonaSummary[]>([]);
  const [loadingProfiles, setLoadingProfiles] = useState(true);
  const [anthropicConfigured, setAnthropicConfigured] = useState(false);
  const [model, setModel] = useState<string>('');
  const [loadError, setLoadError] = useState<string | null>(null);

  const [sender, setSender] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [lastMemoryCount, setLastMemoryCount] = useState<number | null>(null);

  /** Once the user picks someone, don't auto-switch back to the open chat's participant. */
  const userPickedSender = useRef(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const openConversationId =
    selectedConversations.length === 1 ? selectedConversations[0] : undefined;

  const conversationParticipants = React.useMemo(() => {
    if (!openConversationId) return null;
    return conversations.find((c) => c.conversation_id === openConversationId)?.participants ?? null;
  }, [openConversationId, conversations]);

  /** Only pass conversation scope when the selected persona is actually in that chat. */
  const conversationId = React.useMemo(() => {
    if (!openConversationId || !sender) return undefined;
    if (!conversationParticipants?.length) return openConversationId;
    if (conversationParticipants.includes(sender)) return openConversationId;
    return undefined;
  }, [openConversationId, sender, conversationParticipants]);

  const loadProfiles = useCallback(async () => {
    setLoadingProfiles(true);
    setLoadError(null);
    try {
      const res = await fetch('/api/persona/profiles', { cache: 'no-store' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to load personas');
      setProfiles(data.profiles ?? []);
      setAnthropicConfigured(Boolean(data.anthropicConfigured));
      setModel(data.model || '');
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Failed to load personas');
    } finally {
      setLoadingProfiles(false);
    }
  }, []);

  useEffect(() => {
    void loadProfiles();
  }, [loadProfiles, settingsOpen]);

  // Default persona: prefer the other person in the open DM, else first profile.
  // Never override after the user has picked someone manually.
  useEffect(() => {
    if (profiles.length === 0) return;
    if (userPickedSender.current && sender && profiles.some((p) => p.sender === sender)) {
      return;
    }
    if (sender && profiles.some((p) => p.sender === sender) && !conversationParticipants) {
      return;
    }

    if (conversationParticipants?.length && !userPickedSender.current) {
      const others = conversationParticipants.filter(
        (p) => p !== 'Me' && profiles.some((pr) => pr.sender === p)
      );
      if (others.length > 0) {
        const ranked = others
          .map((name) => profiles.find((p) => p.sender === name)!)
          .sort((a, b) => b.messageCount - a.messageCount);
        setSender(ranked[0].sender);
        return;
      }
      const match = profiles.find((p) => conversationParticipants.includes(p.sender));
      if (match) {
        setSender(match.sender);
        return;
      }
    }

    if (!sender || !profiles.some((p) => p.sender === sender)) {
      setSender(profiles[0].sender);
    }
  }, [profiles, conversationParticipants, sender]);

  // Switching persona or chat scope clears the live transcript (start blank).
  useEffect(() => {
    setMessages([]);
    setChatError(null);
    setLastMemoryCount(null);
    setInput('');
  }, [sender, conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sending]);

  const activeProfile = profiles.find((p) => p.sender === sender) ?? null;

  const clearChat = () => {
    setChatError(null);
    setLastMemoryCount(null);
    setMessages([]);
    setInput('');
  };

  const handleSenderChange = (next: string) => {
    userPickedSender.current = true;
    setSender(next);
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || !sender || sending) return;

    setChatError(null);
    setInput('');
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
    };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setSending(true);

    const payloadMessages = nextMessages
      .map((m) => ({ role: m.role, content: m.content }))
      .slice(-50);

    try {
      const res = await fetch('/api/persona/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sender,
          conversationId,
          messages: payloadMessages,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        if (res.status === 401) setAnthropicConfigured(false);
        throw new Error(data.error || 'Chat failed');
      }
      setLastMemoryCount(typeof data.memoryCount === 'number' ? data.memoryCount : null);
      const bubbles: string[] = Array.isArray(data.bubbles) && data.bubbles.length > 0
        ? data.bubbles
        : [data.reply as string];
      setMessages((prev) => [
        ...prev,
        ...bubbles.map((content, i) => ({
          id: `a-${Date.now()}-${i}`,
          role: 'assistant' as const,
          content,
        })),
      ]);
    } catch (err) {
      setChatError(err instanceof Error ? err.message : 'Chat failed');
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  };

  if (loadingProfiles) {
    return (
      <div className="flex items-center justify-center py-24 text-gray-500">
        <Loader2 className="w-5 h-5 animate-spin mr-2" />
        Loading personas…
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-violet-600" />
            Persona chat
          </h2>
          <p className="text-gray-600 mt-1 text-sm max-w-2xl">
            Start a fresh chat. Replies use their style, relationship facts, and retrieved memories
            from your message history.
          </p>
        </div>
        {model && anthropicConfigured && (
          <div className="text-xs text-gray-500 bg-white border border-gray-200 rounded-lg px-3 py-2 space-y-0.5">
            <div>
              Model: <span className="font-mono text-gray-800">{model}</span>
            </div>
            {lastMemoryCount !== null && (
              <div>
                Last reply used <span className="font-medium text-gray-800">{lastMemoryCount}</span>{' '}
                retrieved memories
              </div>
            )}
          </div>
        )}
      </div>

      {loadError && (
        <div className="flex gap-2 p-3 rounded-lg bg-red-50 text-red-800 border border-red-200 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          {loadError}
        </div>
      )}

      {!anthropicConfigured && (
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 p-4 rounded-lg bg-amber-50 border border-amber-200">
          <KeyRound className="w-5 h-5 text-amber-700 shrink-0" />
          <div className="flex-1 text-sm text-amber-900">
            <p className="font-medium">Claude API key required</p>
            <p className="text-amber-800 mt-0.5">
              Save your key securely before chatting. It stays encrypted on this machine.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setSettingsOpen(true)}
            className="inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium bg-amber-900 text-white hover:bg-amber-800"
          >
            Add API key
          </button>
        </div>
      )}

      {profiles.length === 0 && !loadError && (
        <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 text-sm text-slate-700">
          <p className="font-medium text-slate-900">No persona profiles yet</p>
          <p className="mt-1">
            Run <code className="font-mono text-xs bg-white px-1 py-0.5 rounded border">npm run generate-metrics</code>{' '}
            after importing chats, then restart the dashboard so{' '}
            <code className="font-mono text-xs bg-white px-1 py-0.5 rounded border">personaProfiles.json</code> is
            available.
          </p>
        </div>
      )}

      {profiles.length > 0 && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3 lg:items-stretch">
          {/* Mobile: compact persona picker */}
          <div className="lg:hidden">
            <label htmlFor="persona-select" className="mb-1.5 block text-sm font-medium text-gray-700">
              Who to simulate
            </label>
            <select
              id="persona-select"
              value={sender}
              onChange={(e) => handleSenderChange(e.target.value)}
              className="w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm focus:border-violet-500 focus:ring-2 focus:ring-violet-500"
            >
              {profiles.map((p) => (
                <option key={p.sender} value={p.sender}>
                  {p.sender} ({p.messageCount.toLocaleString()} msgs)
                </option>
              ))}
            </select>
          </div>

          {/* Desktop: persona list */}
          <div className="hidden max-h-[min(70dvh,720px)] flex-col overflow-hidden rounded-xl border border-gray-200 bg-white lg:flex">
            <div className="shrink-0 border-b border-gray-100 px-4 py-3">
              <h3 className="text-sm font-semibold text-gray-900">Who to simulate</h3>
              <p className="mt-0.5 text-xs text-gray-500">
                {profiles.length} persona{profiles.length === 1 ? '' : 's'} from your data
              </p>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">
              {profiles.map((p) => {
                const active = p.sender === sender;
                return (
                  <button
                    key={p.sender}
                    type="button"
                    onClick={() => handleSenderChange(p.sender)}
                    className={`w-full border-b border-gray-50 px-4 py-3 text-left transition-colors ${
                      active ? 'bg-violet-50' : 'hover:bg-gray-50'
                    }`}
                  >
                    <div className="truncate font-medium text-gray-900">{p.sender}</div>
                    <div className="mt-0.5 text-xs tabular-nums text-gray-500">
                      {p.messageCount.toLocaleString()} msgs · {p.exampleCount} examples
                    </div>
                    {p.sources.length > 0 && (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {p.sources.slice(0, 3).map((s) => (
                          <PlatformBadge key={s} source={s} />
                        ))}
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex h-[min(70dvh,720px)] min-h-[420px] flex-col overflow-hidden rounded-xl border border-gray-200 bg-white lg:col-span-2">
            <div className="flex shrink-0 items-start justify-between gap-3 border-b border-gray-100 px-3 py-3 sm:px-4">
              <div className="min-w-0">
                <div className="truncate font-semibold text-gray-900">
                  {activeProfile ? `Chat as ${activeProfile.sender}` : 'Select a person'}
                </div>
                {activeProfile && (
                  <p className="mt-1 line-clamp-2 text-xs text-gray-500">
                    {activeProfile.styleSummary}
                  </p>
                )}
                {openConversationId && !conversationId && activeProfile && (
                  <p className="mt-1.5 text-xs text-amber-700">
                    This person isn&apos;t in the open chat — using their global style/memories only.
                  </p>
                )}
                {conversationId && activeProfile && (
                  <p className="mt-1.5 text-xs text-violet-700">
                    Scoped to the open conversation for memory + voice.
                  </p>
                )}
              </div>
              <button
                type="button"
                onClick={clearChat}
                disabled={messages.length === 0 && !input}
                className="inline-flex shrink-0 items-center gap-1 text-xs text-gray-500 hover:text-gray-800 disabled:opacity-40"
                title="Clear chat"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Clear</span>
              </button>
            </div>

            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto bg-slate-50/50 px-3 py-4 sm:px-4">
              {messages.length === 0 && activeProfile && (
                <div className="mx-auto max-w-md py-10 text-center text-sm text-gray-500">
                  <Bot className="mx-auto mb-2 h-8 w-8 text-violet-400" />
                  <p>
                    Message as if you&apos;re texting <strong>{activeProfile.sender}</strong>.
                    Chat starts blank — replies use style examples plus retrieved memories.
                  </p>
                </div>
              )}

              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex gap-2 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {m.role === 'assistant' && (
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-100 text-violet-700">
                      <Bot className="h-4 w-4" />
                    </div>
                  )}
                  <div
                    className={`max-w-[min(85%,28rem)] whitespace-pre-wrap break-words rounded-2xl px-3.5 py-2 text-sm ${
                      m.role === 'user'
                        ? 'rounded-br-md bg-gray-900 text-white'
                        : 'rounded-bl-md border border-gray-200 bg-white text-gray-900 shadow-sm'
                    }`}
                  >
                    {m.content}
                  </div>
                  {m.role === 'user' && (
                    <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gray-200 text-gray-600">
                      <User className="h-4 w-4" />
                    </div>
                  )}
                </div>
              ))}

              {sending && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="flex h-7 w-7 items-center justify-center rounded-full bg-violet-100 text-violet-700">
                    <Loader2 className="h-4 w-4 animate-spin" />
                  </div>
                  <span>{activeProfile?.sender ?? 'Persona'} is typing…</span>
                </div>
              )}

              <div ref={bottomRef} />
            </div>

            {chatError && (
              <div className="mx-3 mb-2 flex gap-2 rounded-lg border border-red-200 bg-red-50 p-2.5 text-xs text-red-800 sm:mx-4">
                <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                {chatError}
              </div>
            )}

            <div className="shrink-0 border-t border-gray-100 p-3">
              <div className="flex items-end gap-2">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  rows={2}
                  disabled={!activeProfile || !anthropicConfigured || sending}
                  placeholder={
                    !anthropicConfigured
                      ? 'Add an API key to start chatting…'
                      : activeProfile
                        ? `Message ${activeProfile.sender}…`
                        : 'Select someone to chat with…'
                  }
                  className="flex-1 resize-none rounded-xl border border-gray-300 px-3 py-2 text-sm focus:border-violet-500 focus:ring-2 focus:ring-violet-500 disabled:bg-gray-50 disabled:text-gray-400"
                />
                <button
                  type="button"
                  onClick={() => void sendMessage()}
                  disabled={!input.trim() || !activeProfile || !anthropicConfigured || sending}
                  className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-40 disabled:hover:bg-violet-600"
                  aria-label="Send"
                >
                  {sending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </button>
              </div>
              <p className="mt-1.5 px-1 text-[11px] text-gray-400">
                Enter to send · Shift+Enter for newline
              </p>
            </div>
          </div>
        </div>
      )}

      <ApiKeySettings
        open={settingsOpen}
        onClose={() => {
          setSettingsOpen(false);
          void loadProfiles();
        }}
      />
    </div>
  );
}
