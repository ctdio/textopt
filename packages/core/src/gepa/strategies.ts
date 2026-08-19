import { argmax, sum } from "../math.js";
import { componentNames } from "../types.js";
import {
  buildInstanceFronts,
  buildObjectiveFronts,
  selectParetoCandidate,
} from "./pareto.js";
import type {
  AcceptancePolicy,
  CandidateRecord,
  CandidateSelector,
  ComponentSelector,
  ParetoFrontier,
  ValEvaluationPolicy,
} from "./types.js";

/**
 * Default parent selection: sample from the instance-wise Pareto frontier with
 * probability proportional to how many validation instances a candidate is best
 * on. This is what keeps GEPA from tunnelling into one lineage.
 *
 * `frontier` chooses what the fronts are taken over. "instance" is GEPA as
 * published. "objective" tracks candidates leading each named objective the
 * adapter reports, and "hybrid" pools both — a candidate then earns selection
 * weight for every instance it wins *and* every objective it leads.
 */
export function paretoSelector(
  args: { epsilon?: number; frontier?: ParetoFrontier } = {},
): CandidateSelector {
  const { epsilon = 0, frontier = "instance" } = args;

  return ({ state, rng }) => {
    const fronts: Set<number>[] = [];

    if (frontier !== "objective") {
      fronts.push(
        ...buildInstanceFronts({ scoreMatrix: state.scoreMatrix, epsilon }),
      );
    }
    if (frontier !== "instance") {
      const objectiveScores = state.objectiveScores ?? [];
      const objectiveFronts = buildObjectiveFronts({
        objectiveScores,
        epsilon,
      });
      if (objectiveFronts.length === 0) {
        throw new Error(
          `paretoSelector frontier "${frontier}" needs objective scores, but no candidate has any; have the adapter return objectiveScores or use frontier "instance"`,
        );
      }
      fronts.push(...objectiveFronts);
    }

    return selectParetoCandidate({
      fronts,
      aggregateScores: state.aggregateScores,
      rng,
    });
  };
}

/** Greedy hill climbing. Useful as an ablation baseline. */
export function currentBestSelector(): CandidateSelector {
  return ({ state }) => argmax(state.aggregateScores);
}

export function epsilonGreedySelector(args: {
  epsilon: number;
}): CandidateSelector {
  const { epsilon } = args;

  return ({ state, rng }) => {
    if (rng.next() < epsilon) {
      return rng.nextInt(state.aggregateScores.length);
    }
    return argmax(state.aggregateScores);
  };
}

/** Pareto selection restricted to the top k candidates by aggregate score. */
export function topKParetoSelector(args: {
  k: number;
  epsilon?: number;
}): CandidateSelector {
  const { k, epsilon = 0 } = args;

  return ({ state, rng }) => {
    const ranked = state.aggregateScores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((entry) => entry.index);
    const allowed = new Set(ranked);

    const fronts = buildInstanceFronts({
      scoreMatrix: state.scoreMatrix,
      epsilon,
    }).map((front) => new Set([...front].filter((id) => allowed.has(id))));

    const nonEmpty = fronts.filter((front) => front.size > 0);
    if (nonEmpty.length === 0) {
      return argmax(state.aggregateScores);
    }

    return selectParetoCandidate({
      fronts: nonEmpty,
      aggregateScores: state.aggregateScores,
      rng,
    });
  };
}

/**
 * Score every accepted candidate on the whole validation set. This is GEPA as
 * published: the frontier is exact, and the cost is one full sweep per
 * acceptance.
 */
export function fullEvaluationPolicy<
  Datum = unknown,
  K extends string = string,
>(): ValEvaluationPolicy<Datum, K> {
  return {
    selectInstances: ({ validationSet }) =>
      validationSet.map((_, index) => index),
    bestCandidate: bestByMeanThenCoverage,
  };
}

/**
 * Score each candidate on a random subset of the validation set. Cheaper per
 * acceptance, at the cost of comparing candidates measured on different
 * instances — coverage breaks ties, so a candidate cannot win by having been
 * asked fewer questions.
 */
export function subsampledEvaluationPolicy<
  Datum = unknown,
  K extends string = string,
>(args: { size: number }): ValEvaluationPolicy<Datum, K> {
  const { size } = args;

  if (!Number.isFinite(size) || size <= 0) {
    throw new Error(
      `subsampledEvaluationPolicy requires a positive size, received ${size}`,
    );
  }

  return {
    selectInstances: ({ validationSet, rng }) =>
      rng.sample(
        validationSet.map((_, index) => index),
        size,
      ),
    bestCandidate: bestByMeanThenCoverage,
  };
}

/**
 * One component per selection, cycling in declaration order from the parent's
 * own cursor. Updating a single component at a time is what makes the minibatch
 * acceptance test attributable; keying off the parent's cursor rather than the
 * global iteration is what guarantees every component of a rarely-selected
 * lineage eventually gets a turn.
 */
export function roundRobinComponentSelector<
  K extends string = string,
>(): ComponentSelector<K> {
  return ({ candidate, cursor }) => {
    const names = componentNames(candidate);
    if (names.length === 0) {
      throw new Error("Candidate has no components to update");
    }
    return [names[cursor % names.length]];
  };
}

/** Update every component in a single reflection call. */
export function allComponentsSelector<
  K extends string = string,
>(): ComponentSelector<K> {
  return ({ candidate }) => componentNames(candidate);
}

/**
 * Accept a mutation only when it beats its parent on the same minibatch. Cheap
 * gate that keeps full validation sweeps for candidates that showed a signal.
 */
export function improvementAcceptance(
  args: { minImprovement?: number } = {},
): AcceptancePolicy {
  const { minImprovement = 0 } = args;

  return ({ parentScores, childScores }) =>
    sum(childScores) > sum(parentScores) + minImprovement;
}

/**
 * Highest mean over the instances it was scored on, with wider coverage
 * winning a tie: a candidate measured on more instances has earned the same
 * mean against more evidence.
 */
function bestByMeanThenCoverage<K extends string>(
  records: readonly CandidateRecord<K>[],
): number {
  let bestId = 0;
  let bestScore = Number.NEGATIVE_INFINITY;
  let bestCoverage = -1;

  for (const record of records) {
    const coverage = record.instanceScores.filter(
      (score) => score !== undefined,
    ).length;

    if (
      record.aggregateScore > bestScore ||
      (record.aggregateScore === bestScore && coverage > bestCoverage)
    ) {
      bestId = record.id;
      bestScore = record.aggregateScore;
      bestCoverage = coverage;
    }
  }
  return bestId;
}
