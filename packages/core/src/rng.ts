/**
 * Seeded pseudo-random number generator.
 *
 * Optimization runs are expensive and long-lived, so every stochastic decision
 * in the engine flows through an explicit Rng instance rather than Math.random.
 * That makes runs reproducible and checkpoints replayable.
 */
export interface Rng {
  next(): number;
  nextInt(maxExclusive: number): number;
  pick<T>(items: readonly T[]): T;
  shuffle<T>(items: readonly T[]): T[];
  /** `k` distinct items, or every item when `k` exceeds the input length. */
  sample<T>(items: readonly T[], k: number): T[];
  /** Picks proportionally to `weights`; falls back to uniform when all are 0. */
  weighted<T>(items: readonly T[], weights: readonly number[]): T;
  /** Position in the stream, for checkpointing. Restore via `createSeededRng`. */
  state(): number;
}

const DEFAULT_SEED = 0x9e3779b9;

export function createSeededRng(seed: number, resumeState?: number): Rng {
  let state = resumeState ?? normalizeSeed(seed);

  function next(): number {
    // mulberry32
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function nextInt(maxExclusive: number): number {
    if (!Number.isFinite(maxExclusive) || maxExclusive <= 0) {
      throw new Error(
        `nextInt requires a positive bound, received ${maxExclusive}`,
      );
    }
    return Math.floor(next() * maxExclusive);
  }

  function pick<T>(items: readonly T[]): T {
    if (items.length === 0) {
      throw new Error("Cannot pick from an empty array");
    }
    return items[nextInt(items.length)] as T;
  }

  function shuffle<T>(items: readonly T[]): T[] {
    const copy = [...items];
    for (let i = copy.length - 1; i > 0; i -= 1) {
      const j = nextInt(i + 1);
      const swap = copy[i] as T;
      copy[i] = copy[j] as T;
      copy[j] = swap;
    }
    return copy;
  }

  function sample<T>(items: readonly T[], k: number): T[] {
    if (k <= 0) {
      return [];
    }
    return shuffle(items).slice(0, k);
  }

  function weighted<T>(items: readonly T[], weights: readonly number[]): T {
    if (items.length === 0) {
      throw new Error("Cannot pick from an empty array");
    }

    const positive = items.map((_, index) => Math.max(0, weights[index] ?? 0));
    const total = positive.reduce((sum, weight) => sum + weight, 0);

    if (total <= 0) {
      return pick(items);
    }

    let threshold = next() * total;
    for (let index = 0; index < items.length; index += 1) {
      threshold -= positive[index] as number;
      if (threshold < 0) {
        return items[index] as T;
      }
    }
    return items[items.length - 1] as T;
  }

  return { next, nextInt, pick, shuffle, sample, weighted, state: () => state };
}

function normalizeSeed(seed: number): number {
  if (!Number.isFinite(seed)) {
    return DEFAULT_SEED;
  }
  const normalized = Math.floor(seed) >>> 0;
  return normalized === 0 ? DEFAULT_SEED : normalized;
}
