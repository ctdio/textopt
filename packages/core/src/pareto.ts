import type { Rng } from "./rng.js";

/**
 * GEPA's Pareto frontier is taken over *validation instances*, not over
 * objectives. The entire optimizer state that matters here is a matrix of
 * per-instance scores, `scoreMatrix[candidateIndex][instanceIndex]`, and every
 * function in this module is pure arithmetic over that matrix.
 */
export interface InstanceFrontArgs {
  scoreMatrix: readonly (readonly number[])[];
  epsilon?: number;
}

export interface PruneFrontsArgs {
  fronts: readonly ReadonlySet<number>[];
  aggregateScores: readonly number[];
}

export interface SelectParetoCandidateArgs {
  fronts: readonly ReadonlySet<number>[];
  aggregateScores: readonly number[];
  rng: Rng;
}

export function computeInstanceBests(
  scoreMatrix: readonly (readonly number[])[],
): number[] {
  const first = scoreMatrix[0];
  if (first === undefined) {
    return [];
  }

  const bests = [...first];
  for (let candidate = 1; candidate < scoreMatrix.length; candidate += 1) {
    const row = scoreMatrix[candidate] as readonly number[];
    for (let instance = 0; instance < bests.length; instance += 1) {
      const score = row[instance] ?? Number.NEGATIVE_INFINITY;
      if (score > (bests[instance] as number)) {
        bests[instance] = score;
      }
    }
  }
  return bests;
}

/**
 * For each instance, the set of candidates achieving the best score on it.
 * Candidates within `epsilon` of the best count as tied, which keeps noisy
 * metrics from collapsing the frontier to a single lineage.
 */
export function buildInstanceFronts(args: InstanceFrontArgs): Set<number>[] {
  const { scoreMatrix, epsilon = 0 } = args;
  const bests = computeInstanceBests(scoreMatrix);

  return bests.map((best, instance) => {
    const front = new Set<number>();
    for (let candidate = 0; candidate < scoreMatrix.length; candidate += 1) {
      const score = (scoreMatrix[candidate] as readonly number[])[instance];
      if (score !== undefined && score >= best - epsilon) {
        front.add(candidate);
      }
    }
    return front;
  });
}

/**
 * Drop candidates that contribute nothing unique: a candidate is dominated when
 * every instance it wins is also won by some surviving candidate. Candidates are
 * considered in ascending aggregate score, so weaker duplicates are removed
 * first. Mirrors `remove_dominated_programs` in the reference implementation.
 */
export function pruneDominatedFronts(args: PruneFrontsArgs): Set<number>[] {
  const { fronts, aggregateScores } = args;

  const candidates = collectCandidates(fronts).sort(
    (a, b) => (aggregateScores[a] ?? 0) - (aggregateScores[b] ?? 0),
  );
  const dominated = new Set<number>();

  let removedOne = true;
  while (removedOne) {
    removedOne = false;
    for (const candidate of candidates) {
      if (dominated.has(candidate)) {
        continue;
      }
      const others = new Set(
        candidates.filter(
          (other) => other !== candidate && !dominated.has(other),
        ),
      );
      if (isDominated({ candidate, others, fronts })) {
        dominated.add(candidate);
        removedOne = true;
        break;
      }
    }
  }

  return fronts.map(
    (front) =>
      new Set([...front].filter((candidate) => !dominated.has(candidate))),
  );
}

/**
 * Sample a parent candidate with probability proportional to the number of
 * instances it is best on, after dominated candidates are pruned.
 */
export function selectParetoCandidate(args: SelectParetoCandidateArgs): number {
  const { fronts, aggregateScores, rng } = args;

  const pruned = pruneDominatedFronts({ fronts, aggregateScores });
  const samplingPool: number[] = [];
  for (const front of pruned) {
    for (const candidate of front) {
      samplingPool.push(candidate);
    }
  }

  if (samplingPool.length === 0) {
    return argmax(aggregateScores);
  }
  return rng.pick(samplingPool);
}

export function argmax(values: readonly number[]): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index] as number;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

export function mean(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  let total = 0;
  for (const value of values) {
    total += value;
  }
  return total / values.length;
}

export function sum(values: readonly number[]): number {
  let total = 0;
  for (const value of values) {
    total += value;
  }
  return total;
}

function collectCandidates(fronts: readonly ReadonlySet<number>[]): number[] {
  const seen = new Set<number>();
  for (const front of fronts) {
    for (const candidate of front) {
      seen.add(candidate);
    }
  }
  return [...seen];
}

function isDominated(args: {
  candidate: number;
  others: ReadonlySet<number>;
  fronts: readonly ReadonlySet<number>[];
}): boolean {
  const { candidate, others, fronts } = args;

  for (const front of fronts) {
    if (!front.has(candidate)) {
      continue;
    }
    const hasSubstitute = [...front].some((other) => others.has(other));
    if (!hasSubstitute) {
      return false;
    }
  }
  return true;
}
