import fs from 'fs';
import path from 'path';

export interface RelationshipCard {
  withPerson: string;
  addressForms: string[];
  recurringPeople: string[];
  recurringPlaces: string[];
  toneWithYou: string;
  sharedMessageCount: number;
  sharedConversationCount: number;
  openers?: string[];
  closers?: string[];
  questionBackRate?: number;
  avgWordsWithYou?: number;
  avgWordsGlobal?: number;
  teasingSamples?: string[];
  sharedConversationIds?: string[];
  summary: string;
  registerSummary?: string;
}

export interface ConversationVoice {
  conversationId: string;
  source: string;
  participantCount: number;
  messageCount: number;
  avgWordsPerMessage: number;
  avgEmojiPerMessage: number;
  lengthLabel: string;
  chatType: 'dm' | 'small_group' | 'group';
  styleSummary: string;
}

export interface PersonaFewShotExample {
  context: string;
  reply: string;
  conversationId: string;
  source: string;
  withYou?: boolean;
}

export interface PersonaProfile {
  sender: string;
  messageCount: number;
  vocabulary: {
    topWords: Array<{ word: string; count: number }>;
    avgWordsPerMessage: number;
    avgEmojiPerMessage: number;
  };
  sentiment: {
    avgCompound: number;
    positiveRatio: number;
    negativeRatio: number;
  };
  responsiveness: {
    medianReplyMs: number | null;
    p90ReplyMs: number | null;
    label: string;
  };
  starters: {
    conversationStarts: number;
    startRatio: number;
  };
  styleSummary: string;
  fewShotExamples: PersonaFewShotExample[];
  withYouFewShotExamples?: PersonaFewShotExample[];
  sources: string[];
  relationshipCard?: RelationshipCard | null;
  conversationVoices?: ConversationVoice[];
  bubbleHabits?: {
    avgBubblesPerTurn: number;
    medianBubblesWhenMulti?: number;
    multiBubbleRate: number;
    sampleTurns: string[][];
    contextualSamples?: Array<{ context: string; bubbles: string[] }>;
    styleSummary: string;
  } | null;
  turnTakingHabits?: {
    questionRate: number;
    doubleTextRate: number;
    expandsOnPartnerRate: number;
    compressesRate: number;
    avgReplyLatencyMs: number | null;
    styleSummary: string;
  } | null;
  sharedTimeline?: {
    withPerson: string;
    facts: Array<{ date: string; text: string; conversationId: string; score: number }>;
    summary: string;
  } | null;
}

export interface PersonaProfilesFile {
  generatedAt?: string;
  profileCount?: number;
  profiles: PersonaProfile[];
  note?: string;
}

function candidatePaths(): string[] {
  const cwd = process.cwd();
  const fromEnv = process.env.DOPPELGANGER_DASH_DIR
    ? [path.join(process.env.DOPPELGANGER_DASH_DIR, 'personaProfiles.json')]
    : [];
  return [
    ...fromEnv,
    path.join(cwd, 'public', 'data', 'personaProfiles.json'),
    path.join(cwd, 'dash-data', 'personaProfiles.json'),
    path.join(cwd, '..', 'dash-data', 'personaProfiles.json'),
  ];
}

export function resolvePersonaProfilesPath(): string | null {
  for (const p of candidatePaths()) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

let profilesCache: { path: string; mtimeMs: number; data: PersonaProfilesFile } | null = null;

export function loadPersonaProfiles(): PersonaProfilesFile | null {
  const filePath = resolvePersonaProfilesPath();
  if (!filePath) return null;
  try {
    const mtimeMs = fs.statSync(filePath).mtimeMs;
    if (profilesCache && profilesCache.path === filePath && profilesCache.mtimeMs === mtimeMs) {
      return profilesCache.data;
    }
    const raw = fs.readFileSync(filePath, 'utf8');
    const data = JSON.parse(raw) as PersonaProfilesFile;
    if (!Array.isArray(data.profiles)) return null;
    profilesCache = { path: filePath, mtimeMs, data };
    return data;
  } catch {
    return null;
  }
}

export function findPersonaProfile(sender: string): PersonaProfile | null {
  const data = loadPersonaProfiles();
  if (!data) return null;
  return data.profiles.find((p) => p.sender === sender) ?? null;
}

export interface PersonaProfileSummary {
  sender: string;
  messageCount: number;
  styleSummary: string;
  exampleCount: number;
  sources: string[];
  avgWordsPerMessage: number;
  avgEmojiPerMessage: number;
  responsivenessLabel: string;
}

export function listPersonaSummaries(): PersonaProfileSummary[] {
  const data = loadPersonaProfiles();
  if (!data) return [];
  return data.profiles
    .map((p) => ({
      sender: p.sender,
      messageCount: p.messageCount,
      styleSummary: p.styleSummary,
      exampleCount: p.fewShotExamples?.length ?? 0,
      sources: p.sources ?? [],
      avgWordsPerMessage: p.vocabulary?.avgWordsPerMessage ?? 0,
      avgEmojiPerMessage: p.vocabulary?.avgEmojiPerMessage ?? 0,
      responsivenessLabel: p.responsiveness?.label ?? 'unknown',
    }))
    .sort((a, b) => b.messageCount - a.messageCount);
}
