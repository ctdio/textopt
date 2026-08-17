import type { Rng } from "./rng.js";

/**
 * GEPA's Pareto frontier is taken over *validation instances*, not over
 * objectives. The entire optimizer state that matters here is a matrix of
 * per-instance scores, `scoreMatrix[candidateIndex][instanceIndex]`, and every
 * function in this module is pure arithmetic over that matrix.
 */
export interface InstanceFrontArgs {
  scoreMatrix: readonly (readonly (number | undefined)[])[];
  epsilon?: number;
}

export interface ObjectiveFrontArgs {
  objectiveScores: readonly (Readonly<Record<string, number>> | undefined)[];
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
  scoreMatrix: readonly (readonly (number | undefined)[])[],
): number[] {
  const first = scoreMatrix[0];
  if (first === undefined) {
    return [];
  }

  const bests = first.map((score) => score ?? Number.NEGATIVE_INFINITY);
  for (let candidate = 1; candidate < scoreMatrix.length; candidate += 1) {
    const row = scoreMatrix[candidate] as readonly (number | undefined)[];
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
      const score = (scoreMatrix[candidate] as readonly (number | undefined)[])[
        instance
      ];
      if (score !== undefined && score >= best - epsilon) {
        front.add(candidate);
      }
    }
    return front;
  });
}

/** The best value any candidate reached on each objective. */
export function objectiveBests(
  objectiveScores: readonly (Readonly<Record<string, number>> | undefined)[],
): Record<string, number> {
  const bests: Record<string, number> = {};

  for (const scores of objectiveScores) {
    for (const [objective, value] of Object.entries(scores ?? {})) {
      const best = bests[objective];
      if (best === undefined || value > best) {
        bests[objective] = value;
      }
    }
  }
  return bests;
}

/**
 * The objective-wise counterpart of `buildInstanceFronts`: one front per named
 * objective, holding the candidates that lead it. Optimizing a system against
 * several metrics at once — accuracy against cost, quality against latency —
 * means the interesting candidates are the ones that lead *an* objective, which
 * an average over instances hides.
 */
export function buildObjectiveFronts(args: ObjectiveFrontArgs): Set<number>[] {
  const { objectiveScores, epsilon = 0 } = args;
  const bests = objectiveBests(objectiveScores);

  return Object.entries(bests).map(([objective, best]) => {
    const front = new Set<number>();
    for (
      let candidate = 0;
      candidate < objectiveScores.length;
      candidate += 1
    ) {
      const value = objectiveScores[candidate]?.[objective];
      if (value !== undefined && value >= best - epsilon) {
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

/** Mean over the values that exist; unscored instances are not zeros. */
export function mean(values: readonly (number | undefined)[]): number {
  let total = 0;
  let count = 0;

  for (const value of values) {
    if (value !== undefined) {
      total += value;
      count += 1;
    }
  }
  return count === 0 ? 0 : total / count;
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
