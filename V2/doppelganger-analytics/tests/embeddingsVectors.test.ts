import { describe, test, expect } from '@jest/globals';
import {
  bufferToFloat32,
  cosineSimilarity,
  float32ToBuffer,
} from '../src/embeddings/vectors.js';

describe('embedding vectors', () => {
  test('float32 round-trip', () => {
    const values = [0.1, -0.5, 1.25, 0];
    const buf = float32ToBuffer(values);
    const back = bufferToFloat32(buf);
    expect([...back].map((v) => Number(v.toFixed(5)))).toEqual(
      values.map((v) => Number(v.toFixed(5)))
    );
  });

  test('cosineSimilarity of identical vectors is ~1', () => {
    const a = [1, 2, 3];
    expect(cosineSimilarity(a, a)).toBeCloseTo(1, 5);
  });

  test('cosineSimilarity of orthogonal vectors is ~0', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 5);
  });
});
