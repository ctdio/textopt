import type { Candidate, EvaluationSplit } from "./types.js";

/** What `stableHash` returns for any value that serializes to `{}`. */
const EMPTY_OBJECT_HASH = stableHash({});

/**
 * What the cache stores per (candidate, instance): the metric the frontier is
 * built from, plus the per-objective breakdown when the adapter reports one.
 * Both come from the same rollout, so caching the score without the objectives
 * would force a re-run to recover them.
 */
export interface CachedScore {
  score: number;
  objectiveScores?: Record<string, number>;
}

export interface EvaluationCache {
  get(key: string): CachedScore | undefined;
  set(key: string, cached: CachedScore): void;
  /** Entries for checkpointing. Omit on caches that are already durable. */
  entries?(): [string, CachedScore][];
}

/**
 * The loop re-evaluates unchanged candidates against the same validation
 * instances constantly (every accepted child inherits most of its parent's
 * text). Caching per (split, candidate text, instance) is the single largest
 * cost saver available, and cached hits are not charged to the metric budget.
 */
export function evaluationCacheKey(args: {
  /** The candidate's `candidateHash`, computed once per candidate. */
  hash: string;
  instanceId: string;
  split: EvaluationSplit;
  /**
   * Names the system the score was measured under — model, decoding settings,
   * scorer version. A score is a measurement of a candidate *and* of what ran
   * it, and nothing else in the key records the second half.
   */
  namespace?: string;
}): string {
  const { hash, instanceId, split, namespace } = args;

  const scope = namespace === undefined ? split : `${namespace}:${split}`;
  return `${scope}:${hash}:${instanceId}`;
}

/**
 * Identifies a candidate by its text. Hoisted out of the key so a sweep over a
 * thousand validation instances hashes the candidate once instead of a
 * thousand times — the candidate is the long part of the key, the instance id
 * is not.
 */
export function candidateHash(candidate: Candidate): string {
  const serialized = Object.keys(candidate)
    .sort()
    .map((name) => `${name}\u0000${candidate[name]}`)
    .join("\u0001");

  return `${hash32(serialized, 0x811c9dc5)}${hash32(serialized, 0x01000193)}`;
}

/**
 * A short, collision-resistant id for arbitrary data, used to name validation
 * instances. Two independent 32-bit passes rather than one: a single pass
 * collides at roughly one pair per thousand instances, and a collision here
 * would silently share cached scores between two different examples.
 */
export function stableHash(value: unknown): string {
  let serialized: string;
  try {
    serialized = JSON.stringify(value) ?? "undefined";
  } catch {
    return "";
  }

  return `${hash32(serialized, 0x811c9dc5)}${hash32(serialized, 0x01000193)}`;
}

/**
 * Names one data instance for the evaluation cache: a content hash, so the same
 * row is the same instance wherever it appears in a run.
 *
 * Falls back to the row's position when the datum carries nothing the hash can
 * read. A Map, a Set and a class instance holding its state privately all
 * serialize to `{}`, and an id two rows share serves each of them the score the
 * other measured. Position is a weaker id — it is only stable while the data is
 * — but it is one instance per row, which is what the cache needs to be sound.
 */
export function defaultInstanceId(args: {
  datum: unknown;
  index: number;
}): string {
  const hash = stableHash(args.datum);
  return hash === "" || hash === EMPTY_OBJECT_HASH ? String(args.index) : hash;
}

export function createMemoryCache(
  args: {
    maxEntries?: number;
    /** Entries from a previous run's checkpoint. */
    entries?: readonly [string, CachedScore][];
  } = {},
): EvaluationCache {
  const { maxEntries = 100_000, entries: initial = [] } = args;
  const entries = new Map<string, CachedScore>(initial);

  return {
    get: (key: string) => entries.get(key),
    set: (key: string, cached: CachedScore) => {
      if (entries.size >= maxEntries && !entries.has(key)) {
        const oldest = entries.keys().next();
        if (!oldest.done) {
          entries.delete(oldest.value);
        }
      }
      entries.set(key, cached);
    },
    entries: () => [...entries],
  };
}

function hash32(value: string, seed: number): string {
  let hash = seed;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36).padStart(7, "0");
}
