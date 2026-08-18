import type { Rng } from "../rng.js";
import { componentNames } from "../types.js";
import type { Candidate } from "../types.js";
import type { CandidateRecord } from "./types.js";

export interface MergeProposal<K extends string = string> {
  candidate: Candidate<K>;
  parentIds: [number, number];
  ancestorId: number;
  /** Identifies the (parent, parent, ancestor) triplet this proposal came from. */
  attemptKey: string;
  /** Identifies which parent each component was taken from. */
  descriptionKey: string;
}

export interface ProposeMergeArgs<K extends string = string> {
  records: readonly CandidateRecord<K>[];
  /**
   * Candidates eligible to be merged — the Pareto dominators. Merging off the
   * frontier wastes rollouts on lineages nothing else needs.
   */
  pool: readonly number[];
  rng: Rng;
  /** Triplet keys already proposed this run, accepted or not. */
  attempted: ReadonlySet<string>;
  /** Description keys already proposed this run, accepted or not. */
  attemptedDescriptions: ReadonlySet<string>;
  maxAttempts?: number;
}

const DEFAULT_MAX_ATTEMPTS = 10;

/**
 * System-aware merge (GEPA's crossover). Two dominator lineages that descend
 * from a common ancestor neither of them regressed against can be recombined
 * for free: for every component, take the version from whichever descendant
 * actually moved it. Sampling is random rather than greedy — always merging the
 * two strongest lineages collapses the frontier's diversity, which is the thing
 * merge exists to exploit.
 *
 * Returns a proposal to be *tested*, not an accepted candidate: the caller
 * still has to score it.
 */
export function proposeMerge<K extends string>(
  args: ProposeMergeArgs<K>,
): MergeProposal<K> | null {
  const {
    records,
    pool,
    rng,
    attempted,
    attemptedDescriptions,
    maxAttempts = DEFAULT_MAX_ATTEMPTS,
  } = args;

  if (pool.length < 2 || records.length < 3) {
    return null;
  }

  const ancestries = buildAncestries(records);
  const existing = new Set(
    records.map((record) => fingerprint(record.candidate)),
  );

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const triplet = sampleTriplet({
      records,
      pool,
      rng,
      attempted,
      ancestries,
      maxAttempts,
    });
    if (triplet === null) {
      continue;
    }

    const { parentIds, ancestorId } = triplet;
    const merged = mergeComponents({ records, parentIds, ancestorId, rng });
    const descriptionKey = `${parentIds[0]}:${parentIds[1]}:${merged.sources.join(",")}`;

    if (
      attemptedDescriptions.has(descriptionKey) ||
      existing.has(fingerprint(merged.candidate))
    ) {
      continue;
    }

    return {
      candidate: merged.candidate,
      parentIds,
      ancestorId,
      attemptKey: `${parentIds[0]}:${parentIds[1]}:${ancestorId}`,
      descriptionKey,
    };
  }

  return null;
}

/**
 * Picks the validation instances a merge is judged on: up to `size` ids, drawn
 * evenly from the instances each parent uniquely wins plus the ones they tie
 * on. A uniform sample would usually miss the handful of instances that
 * distinguish the parents at all, which is exactly where a merge either pays
 * off or breaks.
 */
export function selectMergeSubsample(args: {
  scores1: readonly (number | undefined)[];
  scores2: readonly (number | undefined)[];
  rng: Rng;
  size?: number;
}): number[] {
  const { scores1, scores2, rng, size = 5 } = args;

  // Only instances both parents were scored on can discriminate between them.
  const ids = scores1
    .map((_, index) => index)
    .filter(
      (index) => scores1[index] !== undefined && scores2[index] !== undefined,
    );
  if (ids.length === 0) {
    return [];
  }

  const firstWins = ids.filter(
    (id) => (scores1[id] as number) > (scores2[id] as number),
  );
  const secondWins = ids.filter(
    (id) => (scores2[id] as number) > (scores1[id] as number),
  );
  const ties = ids.filter(
    (id) => !firstWins.includes(id) && !secondWins.includes(id),
  );

  const perBucket = Math.max(1, Math.ceil(size / 3));
  const selected: number[] = [];

  for (const bucket of [firstWins, secondWins, ties]) {
    if (selected.length >= size) {
      break;
    }
    const available = bucket.filter((id) => !selected.includes(id));
    const take = Math.min(available.length, perBucket, size - selected.length);
    if (take > 0) {
      selected.push(...rng.sample(available, take));
    }
  }

  const remaining = size - selected.length;
  if (remaining > 0) {
    const unused = ids.filter((id) => !selected.includes(id));
    if (unused.length >= remaining) {
      selected.push(...rng.sample(unused, remaining));
    } else {
      for (let index = 0; index < remaining; index += 1) {
        selected.push(rng.pick(ids));
      }
    }
  }

  return selected.slice(0, size);
}

function sampleTriplet<K extends string>(args: {
  records: readonly CandidateRecord<K>[];
  pool: readonly number[];
  rng: Rng;
  attempted: ReadonlySet<string>;
  ancestries: readonly ReadonlySet<number>[];
  maxAttempts: number;
}): { parentIds: [number, number]; ancestorId: number } | null {
  const { records, pool, rng, attempted, ancestries, maxAttempts } = args;

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const [first, second] = rng.sample(pool, 2) as [number, number];
    if (first === second) {
      continue;
    }

    const left = Math.min(first, second);
    const right = Math.max(first, second);
    const leftAncestry = ancestries[left] as ReadonlySet<number>;
    const rightAncestry = ancestries[right] as ReadonlySet<number>;

    // One descending from the other means there is nothing to recombine.
    if (leftAncestry.has(right) || rightAncestry.has(left)) {
      continue;
    }

    const common = [...leftAncestry].filter((id) => rightAncestry.has(id));
    const eligible = common.filter((ancestorId) =>
      isEligibleAncestor({ records, left, right, ancestorId, attempted }),
    );
    if (eligible.length === 0) {
      continue;
    }

    return {
      parentIds: [left, right],
      ancestorId: rng.weighted(
        eligible,
        eligible.map(
          (id) => (records[id] as CandidateRecord<K>).aggregateScore,
        ),
      ),
    };
  }

  return null;
}

function isEligibleAncestor<K extends string>(args: {
  records: readonly CandidateRecord<K>[];
  left: number;
  right: number;
  ancestorId: number;
  attempted: ReadonlySet<string>;
}): boolean {
  const { records, left, right, ancestorId, attempted } = args;

  if (attempted.has(`${left}:${right}:${ancestorId}`)) {
    return false;
  }

  const ancestor = records[ancestorId] as CandidateRecord<K>;
  const leftRecord = records[left] as CandidateRecord<K>;
  const rightRecord = records[right] as CandidateRecord<K>;

  // An ancestor that outscores a descendant is not a baseline the descendants
  // agree on, so recombining around it has no reason to help.
  if (
    ancestor.aggregateScore > leftRecord.aggregateScore ||
    ancestor.aggregateScore > rightRecord.aggregateScore
  ) {
    return false;
  }

  return hasComplementaryComponent({
    ancestor: ancestor.candidate,
    left: leftRecord.candidate,
    right: rightRecord.candidate,
  });
}

/**
 * True when at least one component was changed by exactly one descendant. Two
 * lineages that rewrote every shared component give the merge nothing to
 * attribute an improvement to.
 */
function hasComplementaryComponent<K extends string>(args: {
  ancestor: Candidate<K>;
  left: Candidate<K>;
  right: Candidate<K>;
}): boolean {
  const { ancestor, left, right } = args;

  return componentNames(ancestor).some((name) => {
    const base = ancestor[name];
    return (
      (base === left[name] || base === right[name]) &&
      left[name] !== right[name]
    );
  });
}

function mergeComponents<K extends string>(args: {
  records: readonly CandidateRecord<K>[];
  parentIds: [number, number];
  ancestorId: number;
  rng: Rng;
}): { candidate: Candidate<K>; sources: number[] } {
  const { records, parentIds, ancestorId, rng } = args;
  const [leftId, rightId] = parentIds;

  const ancestor = (records[ancestorId] as CandidateRecord<K>).candidate;
  const leftRecord = records[leftId] as CandidateRecord<K>;
  const rightRecord = records[rightId] as CandidateRecord<K>;

  const candidate: Candidate<K> = { ...ancestor };
  const sources: number[] = [];

  for (const name of componentNames(ancestor)) {
    const base = ancestor[name];
    const leftText = leftRecord.candidate[name];
    const rightText = rightRecord.candidate[name];

    const sourceId = resolveComponentSource({
      base,
      leftText,
      rightText,
      leftRecord,
      rightRecord,
      rng,
    });

    candidate[name] = (records[sourceId] as CandidateRecord<K>).candidate[name];
    sources.push(sourceId);
  }

  return { candidate, sources };
}

function resolveComponentSource<K extends string>(args: {
  base: string | undefined;
  leftText: string | undefined;
  rightText: string | undefined;
  leftRecord: CandidateRecord<K>;
  rightRecord: CandidateRecord<K>;
  rng: Rng;
}): number {
  const { base, leftText, rightText, leftRecord, rightRecord, rng } = args;

  // Exactly one lineage moved this component: that one carries the signal.
  if ((base === leftText || base === rightText) && leftText !== rightText) {
    return base === leftText ? rightRecord.id : leftRecord.id;
  }

  // Both moved it, differently. There is no way to tell which rewrite earned
  // the gain without another rollout, so defer to the stronger lineage.
  if (base !== leftText && base !== rightText) {
    if (leftRecord.aggregateScore > rightRecord.aggregateScore) {
      return leftRecord.id;
    }
    if (rightRecord.aggregateScore > leftRecord.aggregateScore) {
      return rightRecord.id;
    }
    return rng.pick([leftRecord.id, rightRecord.id]);
  }

  // Both agree — either both left it alone or both made the same change.
  return leftRecord.id;
}

/** Strict ancestors of each record, keyed by record id. */
function buildAncestries<K extends string>(
  records: readonly CandidateRecord<K>[],
): Set<number>[] {
  const ancestries: Set<number>[] = [];

  for (const record of records) {
    const ancestry = new Set<number>();
    for (const parentId of record.parentIds) {
      ancestry.add(parentId);
      for (const id of ancestries[parentId] ?? []) {
        ancestry.add(id);
      }
    }
    ancestries[record.id] = ancestry;
  }

  return ancestries;
}

function fingerprint<K extends string>(candidate: Candidate<K>): string {
  return JSON.stringify(
    componentNames(candidate)
      .sort()
      .map((name) => [name, candidate[name]]),
  );
}
