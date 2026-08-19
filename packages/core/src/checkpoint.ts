import { stableHash } from "./cache.js";
import { componentNames } from "./types.js";
import type { Candidate } from "./types.js";

/**
 * Identifies the run a checkpoint came from: seed candidate, instance ids,
 * seed, and cache namespace.
 *
 * Hashed rather than embedded, because it goes into every snapshot and is only
 * ever compared for equality. The namespace is part of it because a snapshot
 * carries cached scores, and resuming under a different one would replay
 * measurements of a system the run is no longer running. The test set is
 * deliberately absent: it never touches selection, so adding one to a resumed
 * run changes nothing about what that run would have done.
 */
export function runFingerprint(args: {
  seedCandidate: Candidate;
  trainingIds: readonly string[];
  validationIds: readonly string[];
  seed?: number;
  cacheNamespace?: string;
}): string {
  const { seedCandidate, trainingIds, validationIds, seed, cacheNamespace } =
    args;

  return stableHash({
    seed,
    seedCandidate: candidateFingerprint(seedCandidate),
    trainingIds,
    validationIds,
    cacheNamespace,
  });
}

/**
 * Refuses a checkpoint from a different run rather than silently scoring old
 * candidates against new data — the failure mode that produces a plausible
 * result nobody can reproduce.
 */
export function assertResumable(args: {
  fingerprint: string;
  snapshot?: { fingerprint: string };
}): void {
  const { fingerprint, snapshot } = args;

  if (snapshot !== undefined && snapshot.fingerprint !== fingerprint) {
    throw new Error(
      "checkpoint does not belong to this run: the seed candidate, instance ids, seed or cache namespace differ from the ones it was taken with",
    );
  }
}

/** A candidate's identity: component names and their text, order-independent. */
export function candidateFingerprint<K extends string>(
  candidate: Candidate<K>,
): string {
  return JSON.stringify(
    componentNames(candidate)
      .sort()
      .map((name) => [name, candidate[name]]),
  );
}
