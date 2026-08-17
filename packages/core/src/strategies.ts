import {
  argmax,
  buildInstanceFronts,
  selectParetoCandidate,
  sum,
} from "./pareto.js";
import type { Rng } from "./rng.js";
import type {
  AcceptancePolicy,
  BatchSampler,
  CandidateSelector,
  ComponentSelector,
} from "./types.js";

/**
 * Default parent selection: sample from the instance-wise Pareto frontier with
 * probability proportional to how many validation instances a candidate is best
 * on. This is what keeps GEPA from tunnelling into one lineage.
 */
export function paretoSelector(
  args: { epsilon?: number } = {},
): CandidateSelector {
  const { epsilon = 0 } = args;

  return ({ state, rng }) =>
    selectParetoCandidate({
      fronts: buildInstanceFronts({ scoreMatrix: state.scoreMatrix, epsilon }),
      aggregateScores: state.aggregateScores,
      rng,
    });
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
 * One component per selection, cycling in declaration order from the parent's
 * own cursor. Updating a single component at a time is what makes the minibatch
 * acceptance test attributable; keying off the parent's cursor rather than the
 * global iteration is what guarantees every component of a rarely-selected
 * lineage eventually gets a turn.
 */
export function roundRobinComponentSelector(): ComponentSelector {
  return ({ candidate, cursor }) => {
    const names = Object.keys(candidate);
    if (names.length === 0) {
      throw new Error("Candidate has no components to update");
    }
    return [names[cursor % names.length] as string];
  };
}

/** Update every component in a single reflection call. */
export function allComponentsSelector(): ComponentSelector {
  return ({ candidate }) => Object.keys(candidate);
}

/**
 * Shuffles the trainset once per epoch and walks it in fixed-size chunks, so
 * every training example is seen once before any is seen twice.
 */
export function createEpochShuffledSampler<Datum>(args: {
  minibatchSize: number;
}): BatchSampler<Datum> {
  const { minibatchSize } = args;

  let shuffled: number[] = [];
  let epoch = -1;
  let lastTrainsetSize = -1;

  return ({ trainset, iteration, rng }) => {
    if (trainset.length === 0) {
      throw new Error("Cannot sample a minibatch from an empty trainset");
    }

    const baseIndex = iteration * minibatchSize;
    const currentEpoch =
      shuffled.length === 0 ? 0 : Math.floor(baseIndex / shuffled.length);

    if (
      shuffled.length === 0 ||
      trainset.length !== lastTrainsetSize ||
      currentEpoch > epoch
    ) {
      epoch = currentEpoch;
      lastTrainsetSize = trainset.length;
      shuffled = buildPaddedShuffle({
        size: trainset.length,
        minibatchSize,
        rng,
      });
    }

    const start = baseIndex % shuffled.length;
    return shuffled.slice(start, start + minibatchSize);
  };
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

function buildPaddedShuffle(args: {
  size: number;
  minibatchSize: number;
  rng: Rng;
}): number[] {
  const { size, minibatchSize, rng } = args;

  const indices = rng.shuffle(Array.from({ length: size }, (_, i) => i));
  const remainder = indices.length % minibatchSize;
  const padding = remainder === 0 ? 0 : minibatchSize - remainder;

  const frequencies = new Map<number, number>(
    indices.map((index) => [index, 1]),
  );
  for (let i = 0; i < padding; i += 1) {
    const leastUsed = indices.reduce((best, index) =>
      (frequencies.get(index) as number) < (frequencies.get(best) as number)
        ? index
        : best,
    );
    indices.push(leastUsed);
    frequencies.set(leastUsed, (frequencies.get(leastUsed) as number) + 1);
  }

  return indices;
}
