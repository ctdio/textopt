import { createBudget } from "./budget.js";
import { createMemoryCache, evaluationCacheKey } from "./cache.js";
import { proposeMerge, selectMergeSubsample } from "./merge.js";
import {
  argmax,
  buildInstanceFronts,
  mean,
  pruneDominatedFronts,
  sum,
} from "./pareto.js";
import { createDefaultProposer } from "./reflection.js";
import { createSeededRng } from "./rng.js";
import {
  createEpochShuffledSampler,
  improvementAcceptance,
  paretoSelector,
  roundRobinComponentSelector,
} from "./strategies.js";
import type {
  AcceptancePolicy,
  Adapter,
  BatchSampler,
  Candidate,
  CandidateRecord,
  CandidateSelector,
  ComponentSelector,
  EvaluationCache,
  EvaluationPhase,
  EvaluationSplit,
  OptimizationResult,
  OptimizerEvent,
  Reflector,
} from "./types.js";

export interface OptimizeOptions<Datum, Traj = unknown, Out = unknown> {
  /** Starting text for every component under optimization. */
  seedCandidate: Candidate;
  /** Examples used to build minibatches and reflective feedback. */
  trainset: readonly Datum[];
  /** Instances the Pareto frontier is tracked over. Defaults to the trainset. */
  valset?: readonly Datum[];
  adapter: Adapter<Datum, Traj, Out>;
  reflect: Reflector;
  /** Hard ceiling on rollouts. Cached evaluations are not charged. */
  maxMetricCalls: number;
  minibatchSize?: number;
  maxIterations?: number;
  seed?: number;
  candidateSelector?: CandidateSelector;
  componentSelector?: ComponentSelector;
  batchSampler?: BatchSampler<Datum>;
  acceptance?: AcceptancePolicy;
  /** Pass `false` to disable caching entirely. */
  cache?: EvaluationCache | false;
  instanceId?: (args: { datum: Datum; index: number }) => string;
  /**
   * System-aware merge. Enabled by default for multi-component candidates,
   * where two lineages can improve different components independently.
   */
  merge?: { enabled?: boolean; maxInvocations?: number };
  /**
   * Skip reflection when the parent already scores `perfectScore` on every
   * minibatch instance. There is no failure to diagnose, so the rollouts a
   * proposal would cost are better spent elsewhere. Default true.
   */
  skipPerfectScore?: boolean;
  /** Per-instance score treated as leaving no room to improve. Default 1. */
  perfectScore?: number;
  onEvent?: (event: OptimizerEvent) => void;
  signal?: AbortSignal;
  /** Rethrow adapter failures instead of skipping the iteration. Default true. */
  raiseOnError?: boolean;
}

const DEFAULT_MINIBATCH_SIZE = 3;
const DEFAULT_MAX_MERGES = 5;
const MERGE_SUBSAMPLE_SIZE = 5;

export async function optimize<Datum, Traj = unknown, Out = unknown>(
  options: OptimizeOptions<Datum, Traj, Out>,
): Promise<OptimizationResult> {
  const {
    seedCandidate,
    trainset,
    valset = trainset,
    adapter,
    reflect,
    maxMetricCalls,
    minibatchSize = DEFAULT_MINIBATCH_SIZE,
    maxIterations = Number.POSITIVE_INFINITY,
    seed = 0,
    candidateSelector = paretoSelector(),
    componentSelector = roundRobinComponentSelector(),
    batchSampler = createEpochShuffledSampler<Datum>({ minibatchSize }),
    acceptance = improvementAcceptance(),
    cache,
    instanceId = defaultInstanceId,
    merge,
    skipPerfectScore = true,
    perfectScore = 1,
    onEvent,
    signal,
    raiseOnError = true,
  } = options;

  const mergeConfig = {
    enabled: merge?.enabled ?? Object.keys(seedCandidate).length > 1,
    maxInvocations: merge?.maxInvocations ?? DEFAULT_MAX_MERGES,
  };

  if (trainset.length === 0) {
    throw new Error("optimize requires a non-empty trainset");
  }
  if (valset.length === 0) {
    throw new Error(
      "optimize requires a non-empty valset; the Pareto frontier is tracked over validation instances",
    );
  }
  if (Object.keys(seedCandidate).length === 0) {
    throw new Error(
      "optimize requires a seed candidate with at least one component",
    );
  }

  const rng = createSeededRng(seed);
  const budget = createBudget({ maxMetricCalls });
  const evaluationCache =
    cache === false ? undefined : (cache ?? createMemoryCache());
  const propose =
    adapter.proposeNewTexts?.bind(adapter) ?? createDefaultProposer();

  const trainIds = trainset.map((datum, index) => instanceId({ datum, index }));
  const valIds = valset.map((datum, index) => instanceId({ datum, index }));

  const records: CandidateRecord[] = [];
  const seenCandidates = new Set<string>();
  let cacheHits = 0;
  let iteration = 0;

  const mergeAttempts = new Set<string>();
  const mergeDescriptions = new Set<string>();
  let mergesDue = 0;
  let totalMergesTested = 0;
  let lastIterationAccepted = false;

  function emit(event: OptimizerEvent): void {
    onEvent?.(event);
  }

  /**
   * Evaluates a candidate on a batch, serving what it can from the cache and
   * charging the budget only for instances that actually run.
   */
  async function evaluateCached(args: {
    candidate: Candidate;
    batch: readonly Datum[];
    ids: readonly string[];
    split: EvaluationSplit;
    phase: EvaluationPhase;
    candidateId: number | null;
  }): Promise<number[]> {
    const { candidate, batch, ids, split, phase, candidateId } = args;

    const scores = new Array<number>(batch.length);
    const pendingIndices: number[] = [];

    for (let index = 0; index < batch.length; index += 1) {
      const cached = evaluationCache?.get(
        evaluationCacheKey({
          candidate,
          instanceId: ids[index] as string,
          split,
        }),
      );
      if (cached === undefined) {
        pendingIndices.push(index);
      } else {
        scores[index] = cached;
      }
    }

    cacheHits += batch.length - pendingIndices.length;

    if (pendingIndices.length > 0) {
      const evaluation = await adapter.evaluate({
        batch: pendingIndices.map((index) => batch[index] as Datum),
        candidate,
        captureTraces: false,
        signal,
      });
      assertScores({
        scores: evaluation.scores,
        expected: pendingIndices.length,
      });
      budget.charge(pendingIndices.length);

      pendingIndices.forEach((batchIndex, resultIndex) => {
        const score = evaluation.scores[resultIndex] as number;
        scores[batchIndex] = score;

        // A transient score says nothing about the candidate, so caching it
        // would pin this instance to an infrastructure failure forever.
        if (evaluation.transient?.[resultIndex] === true) {
          return;
        }
        evaluationCache?.set(
          evaluationCacheKey({
            candidate,
            instanceId: ids[batchIndex] as string,
            split,
          }),
          score,
        );
      });
    }

    emit({
      type: "evaluation",
      iteration,
      phase,
      candidateId,
      metricCalls: pendingIndices.length,
      cacheHits: batch.length - pendingIndices.length,
      meanScore: mean(scores),
    });

    return scores;
  }

  function countUncached(args: {
    candidate: Candidate;
    ids: readonly string[];
    split: EvaluationSplit;
  }): number {
    const { candidate, ids, split } = args;

    if (evaluationCache === undefined) {
      return ids.length;
    }
    return ids.filter(
      (id) =>
        evaluationCache.get(
          evaluationCacheKey({ candidate, instanceId: id, split }),
        ) === undefined,
    ).length;
  }

  function addCandidate(args: {
    candidate: Candidate;
    parentIds: number[];
    instanceScores: number[];
    source: CandidateRecord["source"];
    updatedComponents: string[];
  }): CandidateRecord {
    const record: CandidateRecord = {
      id: records.length,
      candidate: args.candidate,
      parentIds: args.parentIds,
      instanceScores: args.instanceScores,
      aggregateScore: mean(args.instanceScores),
      source: args.source,
      updatedComponents: args.updatedComponents,
      iteration,
      componentCursor: inheritedCursor(args.parentIds),
    };
    records.push(record);
    seenCandidates.add(candidateFingerprint(args.candidate));

    // A new frontier member is what makes a merge worth attempting, so every
    // acceptance schedules one.
    lastIterationAccepted = true;
    if (mergeConfig.enabled && totalMergesTested < mergeConfig.maxInvocations) {
      mergesDue += 1;
    }

    return record;
  }

  function inheritedCursor(parentIds: readonly number[]): number {
    let cursor = 0;
    for (const parentId of parentIds) {
      const parent = records[parentId];
      if (parent !== undefined && parent.componentCursor > cursor) {
        cursor = parent.componentCursor;
      }
    }
    return cursor;
  }

  emit({
    type: "start",
    components: Object.keys(seedCandidate),
    valsetSize: valset.length,
  });

  if (!budget.canAfford(valset.length)) {
    throw new Error(
      `maxMetricCalls (${maxMetricCalls}) is smaller than the validation set (${valset.length}); the seed candidate cannot be scored`,
    );
  }

  const seedScores = await evaluateCached({
    candidate: seedCandidate,
    batch: valset,
    ids: valIds,
    split: "val",
    phase: "seed",
    candidateId: 0,
  });
  addCandidate({
    candidate: seedCandidate,
    parentIds: [],
    instanceScores: seedScores,
    source: "seed",
    updatedComponents: [],
  });
  lastIterationAccepted = false;
  mergesDue = 0;

  let stopReason: OptimizationResult["stopReason"] = "budgetExhausted";

  /**
   * Proposes and gates one merge. Returns "none" when nothing was tested — the
   * iteration then falls through to reflective mutation, exactly as it would
   * have without merging enabled. A merge that cannot be afforded is skipped,
   * never treated as the end of the run.
   */
  async function tryMerge(): Promise<"none" | "attempted"> {
    const proposal = proposeMerge({
      records,
      pool: collectDominatorIds(records),
      rng,
      attempted: mergeAttempts,
      attemptedDescriptions: mergeDescriptions,
    });
    if (proposal === null) {
      return "none";
    }

    const [leftId, rightId] = proposal.parentIds;
    const left = records[leftId] as CandidateRecord;
    const right = records[rightId] as CandidateRecord;

    const subsample = selectMergeSubsample({
      scores1: left.instanceScores,
      scores2: right.instanceScores,
      rng,
      size: MERGE_SUBSAMPLE_SIZE,
    });
    if (subsample.length === 0) {
      return "none";
    }

    const unique = [...new Set(subsample)];
    const uniqueIds = unique.map((index) => valIds[index] as string);
    if (
      !budget.canAfford(
        countUncached({
          candidate: proposal.candidate,
          ids: uniqueIds,
          split: "val",
        }),
      )
    ) {
      return "none";
    }

    // Recorded before scoring: a triplet that was tested and lost must not be
    // proposed again, or the run relitigates the same merge forever.
    mergeAttempts.add(proposal.attemptKey);
    mergeDescriptions.add(proposal.descriptionKey);

    const uniqueScores = await evaluateCached({
      candidate: proposal.candidate,
      batch: unique.map((index) => valset[index] as Datum),
      ids: uniqueIds,
      split: "val",
      phase: "minibatch",
      candidateId: null,
    });
    const scoreByIndex = new Map<number, number>(
      unique.map((index, position) => [
        index,
        uniqueScores[position] as number,
      ]),
    );

    const mergedSum = sum(
      subsample.map((index) => scoreByIndex.get(index) as number),
    );
    const parentBest = Math.max(
      sum(subsample.map((index) => left.instanceScores[index] as number)),
      sum(subsample.map((index) => right.instanceScores[index] as number)),
    );

    if (mergedSum < parentBest) {
      emit({
        type: "candidateRejected",
        iteration,
        parentId: leftId,
        parentScore: parentBest,
        childScore: mergedSum,
        source: "merge",
      });
      return "attempted";
    }

    if (
      !budget.canAfford(
        countUncached({
          candidate: proposal.candidate,
          ids: valIds,
          split: "val",
        }),
      )
    ) {
      return "attempted";
    }

    const scores = await evaluateCached({
      candidate: proposal.candidate,
      batch: valset,
      ids: valIds,
      split: "val",
      phase: "validation",
      candidateId: records.length,
    });

    const ancestor = records[proposal.ancestorId] as CandidateRecord;
    const record = addCandidate({
      candidate: proposal.candidate,
      parentIds: [...proposal.parentIds],
      instanceScores: scores,
      source: "merge",
      updatedComponents: Object.keys(proposal.candidate).filter(
        (name) => proposal.candidate[name] !== ancestor.candidate[name],
      ),
    });
    mergesDue -= 1;
    totalMergesTested += 1;

    emit({
      type: "candidateAccepted",
      iteration,
      candidateId: record.id,
      parentIds: record.parentIds,
      aggregateScore: record.aggregateScore,
      source: "merge",
    });
    return "attempted";
  }

  while (true) {
    if (signal?.aborted) {
      stopReason = "aborted";
      break;
    }
    if (iteration >= maxIterations) {
      stopReason = "maxIterations";
      break;
    }
    if (!budget.canAfford(minibatchSize * 2)) {
      stopReason = "budgetExhausted";
      break;
    }

    const spentBeforeIteration = budget.spent();

    try {
      const mergeScheduled =
        mergeConfig.enabled && mergesDue > 0 && lastIterationAccepted;
      lastIterationAccepted = false;

      if (mergeScheduled && (await tryMerge()) === "attempted") {
        iteration += 1;
        continue;
      }

      const selectionState = {
        scoreMatrix: records.map((record) => record.instanceScores),
        aggregateScores: records.map((record) => record.aggregateScore),
      };
      const parentIndex = candidateSelector({ state: selectionState, rng });
      const parent = records[parentIndex] as CandidateRecord;

      emit({ type: "iterationStart", iteration, parentId: parent.id });

      const batchIndices = batchSampler({ trainset, iteration, rng });
      const batch = batchIndices.map((index) => trainset[index] as Datum);
      const batchIds = batchIndices.map((index) => trainIds[index] as string);

      // Traces are required for reflection, so this evaluation always runs and
      // is always charged — but only once it has actually produced scores.
      if (!budget.canAfford(batch.length)) {
        stopReason = "budgetExhausted";
        break;
      }
      const parentEvaluation = await adapter.evaluate({
        batch,
        candidate: parent.candidate,
        captureTraces: true,
        signal,
      });
      assertScores({
        scores: parentEvaluation.scores,
        expected: batch.length,
      });
      budget.charge(batch.length);
      emit({
        type: "evaluation",
        iteration,
        phase: "minibatch",
        candidateId: parent.id,
        metricCalls: batch.length,
        cacheHits: 0,
        meanScore: mean(parentEvaluation.scores),
      });

      if (
        skipPerfectScore &&
        parentEvaluation.scores.every((score) => score >= perfectScore)
      ) {
        iteration += 1;
        continue;
      }

      const componentsToUpdate = componentSelector({
        candidate: parent.candidate,
        cursor: parent.componentCursor,
        iteration,
        rng,
      });
      parent.componentCursor =
        (parent.componentCursor + 1) %
        Math.max(1, Object.keys(parent.candidate).length);

      const reflectiveDataset = await adapter.makeReflectiveDataset({
        candidate: parent.candidate,
        batch,
        evaluation: parentEvaluation,
        componentsToUpdate,
      });
      const proposed = await propose({
        candidate: parent.candidate,
        reflectiveDataset,
        componentsToUpdate,
        reflect,
        signal,
      });

      const child: Candidate = { ...parent.candidate, ...proposed };
      const changed =
        Object.keys(proposed).length > 0 &&
        !seenCandidates.has(candidateFingerprint(child));

      emit({
        type: "proposal",
        iteration,
        parentId: parent.id,
        componentsToUpdate: [...componentsToUpdate],
        changed,
      });

      if (!changed) {
        iteration += 1;
        continue;
      }

      if (
        !budget.canAfford(
          countUncached({ candidate: child, ids: batchIds, split: "train" }),
        )
      ) {
        stopReason = "budgetExhausted";
        break;
      }
      const childBatchScores = await evaluateCached({
        candidate: child,
        batch,
        ids: batchIds,
        split: "train",
        phase: "minibatch",
        candidateId: null,
      });

      if (
        !acceptance({
          parentScores: parentEvaluation.scores,
          childScores: childBatchScores,
        })
      ) {
        emit({
          type: "candidateRejected",
          iteration,
          parentId: parent.id,
          parentScore: mean(parentEvaluation.scores),
          childScore: mean(childBatchScores),
          source: "mutation",
        });
        iteration += 1;
        continue;
      }

      if (
        !budget.canAfford(
          countUncached({ candidate: child, ids: valIds, split: "val" }),
        )
      ) {
        stopReason = "budgetExhausted";
        break;
      }
      const childValScores = await evaluateCached({
        candidate: child,
        batch: valset,
        ids: valIds,
        split: "val",
        phase: "validation",
        candidateId: records.length,
      });

      const record = addCandidate({
        candidate: child,
        parentIds: [parent.id],
        instanceScores: childValScores,
        source: "mutation",
        updatedComponents: Object.keys(proposed),
      });
      emit({
        type: "candidateAccepted",
        iteration,
        candidateId: record.id,
        parentIds: record.parentIds,
        aggregateScore: record.aggregateScore,
        source: "mutation",
      });
    } catch (err) {
      // An adapter that honours the signal reports cancellation by throwing.
      // That is the run ending on request, not a failure to report or rethrow.
      if (signal?.aborted) {
        stopReason = "aborted";
        break;
      }
      // An iteration that failed without producing a single evaluation made no
      // progress, so tolerating it would just burn the remaining iterations on
      // the identical failure.
      if (raiseOnError || budget.spent() === spentBeforeIteration) {
        throw err;
      }
      emit({ type: "error", iteration, err });
    }

    iteration += 1;
  }

  const aggregateScores = records.map((record) => record.aggregateScore);
  const bestCandidateId = argmax(aggregateScores);
  const best = records[bestCandidateId] as CandidateRecord;

  emit({
    type: "finish",
    reason: stopReason,
    bestCandidateId,
    metricCalls: budget.spent(),
  });

  return {
    bestCandidate: best.candidate,
    bestScore: best.aggregateScore,
    bestCandidateId,
    candidates: records,
    paretoFrontier: collectDominatorIds(records).map(
      (id) => records[id] as CandidateRecord,
    ),
    scoreMatrix: records.map((record) => [...record.instanceScores]),
    metricCalls: budget.spent(),
    cacheHits,
    iterations: iteration,
    stopReason,
  };
}

/**
 * Candidates that uniquely win at least one validation instance once dominated
 * lineages are pruned. This is both the reported frontier and the pool merge
 * draws its parents from.
 */
function collectDominatorIds(records: readonly CandidateRecord[]): number[] {
  const fronts = pruneDominatedFronts({
    fronts: buildInstanceFronts({
      scoreMatrix: records.map((record) => record.instanceScores),
    }),
    aggregateScores: records.map((record) => record.aggregateScore),
  });

  const ids = new Set<number>();
  for (const front of fronts) {
    for (const id of front) {
      ids.add(id);
    }
  }
  return [...ids].sort((a, b) => a - b);
}

function candidateFingerprint(candidate: Candidate): string {
  return JSON.stringify(
    Object.keys(candidate)
      .sort()
      .map((name) => [name, candidate[name]]),
  );
}

function defaultInstanceId(args: { datum: unknown; index: number }): string {
  try {
    return JSON.stringify(args.datum) ?? String(args.index);
  } catch {
    return String(args.index);
  }
}

/**
 * A NaN score never raises on its own: every comparison that decides fronts or
 * the best candidate is false for NaN, so the candidate silently becomes an
 * unselectable phantom that still consumed budget. Catch it at the boundary.
 */
function assertScores(args: {
  scores: readonly number[];
  expected: number;
}): void {
  const { scores, expected } = args;

  if (scores.length !== expected) {
    throw new Error(
      `Adapter returned ${scores.length} scores for a batch of ${expected}; scores must align one-to-one with the batch`,
    );
  }

  for (let index = 0; index < scores.length; index += 1) {
    const score = scores[index];
    if (typeof score !== "number" || !Number.isFinite(score)) {
      throw new Error(
        `Adapter returned a non-finite score at index ${index}: ${String(score)}`,
      );
    }
  }
}
