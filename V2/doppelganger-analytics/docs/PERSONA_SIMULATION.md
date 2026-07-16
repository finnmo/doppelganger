# AI Persona Simulation — Path to ~85% Similarity

This project analyzes messaging history and exports **persona profiles** ready for an LLM. It does not call a model yet — wire `buildPersonaPrompt()` to your provider to generate replies.

## What generate produces

`dash-data/personaProfiles.json` per sender:

| Field | Role |
|--------|------|
| `styleSummary` | System-prompt fragment (length, emoji, tone, reply speed) |
| `fewShotExamples` | Real `(context → their reply)` pairs, diversified across chats |
| `relationshipCard` | How they address you, recurring people/places, tone with you + **with-you register** (openers, question-back rate, teasing samples) |
| `withYouFewShotExamples` | Few-shot pairs where they are replying to you specifically |
| `conversationVoices` | Per-chat voice notes (DM vs group); injected when that chat is open |
| `platformVoices` | Per-app voice notes (Instagram vs WhatsApp, …); injected from conversation namespace |
| `sources` | Platforms they appear on (instagram, whatsapp, …) |
| vocabulary / sentiment / responsiveness | Supporting stats |

Live prompt assembly: `dashboard/src/lib/server/buildPersonaPrompt.ts` → `buildAnthropicPersonaRequest`.

## Import freshness

`npm run import` regenerates analytics by default (`--no-generate` to skip). Import/generate timestamps live in the SQLite `meta` table; the dashboard shows a staleness banner when profiles lag the DB.

## Accuracy target: ~85%

| Tier | Method | Expected similarity |
|------|--------|---------------------|
| 1 | Style summary only | ~40–60% |
| 2 | + RAG over history | ~60–75% |
| **3** | **+ 20–40 few-shot pairs (shipped)** | **~70–85%** |
| 4 | Fine-tune on held-out threads | ~85–92% |

**85% is reachable at Tier 3** with a strong chat model (GPT-4-class / Claude Sonnet-class), 20–40 diverse few-shot pairs, and evaluation on held-out replies from the same person. Fine-tuning (Tier 4) is optional polish, not required for the goal.

## Integration sketch

```ts
import { buildPersonaPrompt } from './persona/buildPrompt.js';
// load personaProfiles.json → pick sender
const bundle = buildPersonaPrompt(profile, profile.fewShotExamples);
// bundle.messages → send to OpenAI / Anthropic / local model
// then append the live user message and generate
```

## Evaluation (do this before claiming 85%)

1. Hold out 50–100 real reply pairs (not in few-shot).
2. For each: prompt with context, generate, compare to actual reply.
3. Score with embedding cosine similarity **and** human blind ratings (style match 1–5).
4. Target: median embedding ≥ 0.75 **or** human style score ≥ 4.0 ≈ “~85% feels like them.”

## Privacy

Only simulate people who consented. Treat exports and `personaProfiles.json` as confidential.

## Next implementation steps

1. ~~CLI/API persona reply with few-shot~~ (dashboard Persona Chat)
2. ~~Conversation thread seeding + keyword RAG over SQLite~~ (shipped)
3. ~~Embedding-based RAG~~ (OpenAI/Voyage at generate-time → `message_embeddings`; chat-time vector retrieval with keyword fallback)
4. ~~Smarter few-shot~~ (rank stored examples by similarity to the live message; larger generate-time pool)
5. ~~Relationship / fact card~~ (per-person card injected into the system prompt; self = most 1:1 chats or `DOPPELGANGER_SELF_NAME`)
6. ~~Per-conversation voice~~ (DM vs group style notes; injected when that conversation is open in Persona Chat)
7. ~~With-you register~~ (openers/teasing/question-back + with-you few-shot/RAG bias)
8. ~~Multi-bubble replies~~ (`<<<BUBBLE>>>` delimiter + UI split)
9. ~~Held-out eval harness~~ (`personaEval.json`; `npm run persona-eval [-- --live]`)
10. ~~Fine-tune export scaffold~~ (`npm run persona-finetune-export` → `dash-data/personaFineTune.jsonl`)

### Embedding RAG setup

1. Save an OpenAI or Voyage key in dashboard **API settings**.
2. Run `npm run generate-metrics` (builds `message_embeddings` for eligible senders).
3. Persona chat returns `retrievalMode: "vector"` when the index is live; otherwise `"keyword"`.
