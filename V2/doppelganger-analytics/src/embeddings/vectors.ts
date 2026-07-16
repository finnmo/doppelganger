/** Float32 embedding helpers (shared math, no I/O). */

export function float32ToBuffer(values: number[]): Buffer {
  const buf = Buffer.allocUnsafe(values.length * 4);
  for (let i = 0; i < values.length; i++) {
    buf.writeFloatLE(values[i], i * 4);
  }
  return buf;
}

export function bufferToFloat32(buf: Buffer): Float32Array {
  const out = new Float32Array(buf.length / 4);
  for (let i = 0; i < out.length; i++) {
    out[i] = buf.readFloatLE(i * 4);
  }
  return out;
}

export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i];
    const y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

export type EmbeddingProvider = 'openai' | 'voyage';

export const EMBEDDING_MODELS: Record<
  EmbeddingProvider,
  { id: string; dims: number; label: string }
> = {
  openai: { id: 'text-embedding-3-small', dims: 1536, label: 'OpenAI text-embedding-3-small' },
  voyage: { id: 'voyage-3-lite', dims: 512, label: 'Voyage voyage-3-lite' },
};
