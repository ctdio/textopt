import { createBudget } from "./budget.js";
import { createMemoryCache, evaluationCacheKey } from "./cache.js";
import { proposeMerge, selectMergeSubsample } from "./merge.js";
import {
  buildInstanceFronts,
  mean,
  objectiveBests,
  pruneDominatedFronts,
  sum,
} from "./pareto.js";
import { createDefaultProposer } from "./reflection.js";
import { createSeededRng } from "./rng.js";
import {
  createEpochShuffledSampler,
  fullEvaluationPolicy,
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
  ComponentPatch,
  ComponentSelector,
  EvaluationCache,
  EvaluationPhase,
  EvaluationSplit,
  OptimizationResult,
  OptimizerEvent,
  OptimizerSnapshot,
  Reflector,
  RejectedProposal,
  ValEvaluationPolicy,
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
  /**
   * Which validation instances each candidate is scored on. Defaults to a full
   * sweep per accepted candidate, which is what makes the frontier exact.
   */
  valEvaluationPolicy?: ValEvaluationPolicy<Datum>;
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
  /**
   * How many rejected proposals per component are shown back to the reflection
   * model, most recent first. 0 disables the feedback. Default 3.
   */
  rejectedProposalMemory?: number;
  onEvent?: (event: OptimizerEvent) => void;
  /**
   * Called with a resumable snapshot after the seed is scored and after every
   * iteration. Persist it and a killed run costs the last iteration, not all
   * of them.
   */
  onCheckpoint?: (snapshot: OptimizerSnapshot) => void | Promise<void>;
  /** Snapshot to continue from, instead of starting at the seed candidate. */
  resumeFrom?: OptimizerSnapshot;
  signal?: AbortSignal;
  /** Rethrow adapter failures instead of skipping the iteration. Default true. */
  raiseOnError?: boolean;
}

/** Scores plus, when the adapter reports them, their per-objective breakdown. */
interface ScoredBatch {
  scores: number[];
  objectiveScores: (Record<string, number> | undefined)[];
}

/** A scored batch spread over the whole valset, with gaps where it was not. */
interface EvaluatedBatch {
  scores: (number | undefined)[];
  objectiveScores: (Record<string, number> | undefined)[];
}

const DEFAULT_MINIBATCH_SIZE = 3;
const DEFAULT_REJECTED_PROPOSAL_MEMORY = 3;
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
    valEvaluationPolicy = fullEvaluationPolicy<Datum>(),
    cache,
    instanceId = defaultInstanceId,
    merge,
    skipPerfectScore = true,
    perfectScore = 1,
    rejectedProposalMemory = DEFAULT_REJECTED_PROPOSAL_MEMORY,
    onEvent,
    onCheckpoint,
    resumeFrom,
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

  const evaluationCache =
    cache === false ? undefined : (cache ?? createMemoryCache());
  const propose =
    adapter.proposeNewTexts?.bind(adapter) ?? createDefaultProposer();

  const trainIds = trainset.map((datum, index) => instanceId({ datum, index }));
  const valIds = valset.map((datum, index) => instanceId({ datum, index }));

  const fingerprint = runFingerprint({
    seedCandidate,
    trainIds,
    valIds,
    seed,
  });
  if (resumeFrom !== undefined && resumeFrom.fingerprint !== fingerprint) {
    throw new Error(
      "checkpoint does not belong to this run: the seed candidate, instance ids or seed differ from the ones it was taken with",
    );
  }

  const rng = createSeededRng(seed, resumeFrom?.rngState);
  const budget = createBudget({
    maxMetricCalls,
    spent: resumeFrom?.metricCalls ?? 0,
  });

  const records: CandidateRecord[] = [...(resumeFrom?.records ?? [])];
  const seenCandidates = new Set(
    records.map((record) => candidateFingerprint(record.candidate)),
  );
  const rejectedProposals: Record<string, RejectedProposal[]> = {
    ...resumeFrom?.rejectedProposals,
  };
  let cacheHits = resumeFrom?.cacheHits ?? 0;
  let iteration = resumeFrom?.iteration ?? 0;

  const mergeAttempts = new Set<string>(resumeFrom?.merge.attempts);
  const mergeDescriptions = new Set<string>(resumeFrom?.merge.descriptions);
  let mergesDue = resumeFrom?.merge.due ?? 0;
  let totalMergesTested = resumeFrom?.merge.tested ?? 0;
  let lastIterationAccepted = resumeFrom?.merge.lastIterationAccepted ?? false;

  for (const [key, cached] of resumeFrom?.cache ?? []) {
    evaluationCache?.set(key, cached);
  }

  function emit(event: OptimizerEvent): void {
    onEvent?.(event);
  }

  /**
   * Copies everything mutable: a snapshot handed to `onCheckpoint` is a record
   * of that moment, and would otherwise keep growing as the run continues.
   */
  function takeSnapshot(): OptimizerSnapshot {
    const cached = evaluationCache?.entries?.();

    return {
      version: 1,
      fingerprint,
      records: records.map((record) => ({
        ...record,
        parentIds: [...record.parentIds],
        instanceScores: [...record.instanceScores],
        updatedComponents: [...record.updatedComponents],
        ...(record.objectiveScores === undefined
          ? {}
          : { objectiveScores: { ...record.objectiveScores } }),
      })),
      iteration,
      metricCalls: budget.spent(),
      cacheHits,
      rejectedProposals: Object.fromEntries(
        Object.entries(rejectedProposals).map(([component, history]) => [
          component,
          history.map((entry) => ({ ...entry })),
        ]),
      ),
      rngState: rng.state(),
      merge: {
        attempts: [...mergeAttempts],
        descriptions: [...mergeDescriptions],
        due: mergesDue,
        tested: totalMergesTested,
        lastIterationAccepted,
      },
      ...(cached === undefined ? {} : { cache: cached }),
    };
  }

  async function checkpoint(): Promise<void> {
    if (onCheckpoint === undefined) {
      return;
    }
    await onCheckpoint(takeSnapshot());
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
  }): Promise<ScoredBatch> {
    const { candidate, batch, ids, split, phase, candidateId } = args;

    const scores = new Array<number>(batch.length);
    const objectiveScores = new Array<Record<string, number> | undefined>(
      batch.length,
    );
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
        scores[index] = cached.score;
        objectiveScores[index] = cached.objectiveScores;
      }
    }

    cacheHits += batch.length - pendingIndices.length;

    if (pendingIndices.length > 0) {
      const evaluation = await adapter.evaluate({
        batch: pendingIndices.map((index) => batch[index] as Datum),
        candidate,
        captureTraces: false,
        run: { iteration, phase, split, candidateId },
        signal,
      });
      assertScores({
        scores: evaluation.scores,
        expected: pendingIndices.length,
      });
      budget.charge(pendingIndices.length);

      pendingIndices.forEach((batchIndex, resultIndex) => {
        const score = evaluation.scores[resultIndex] as number;
        const objectives = evaluation.objectiveScores?.[resultIndex];
        scores[batchIndex] = score;
        objectiveScores[batchIndex] = objectives;

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
          objectives === undefined
            ? { score }
            : { score, objectiveScores: objectives },
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

    return { scores, objectiveScores };
  }

  /**
   * Scores a candidate on the validation instances the policy selects, and
   * spreads the result back over the full validation set — instances the
   * policy skipped stay `undefined`, which every consumer reads as unknown
   * rather than as a zero.
   */
  async function evaluateValidation(args: {
    candidate: Candidate;
    instances: readonly number[];
    phase: EvaluationPhase;
    candidateId: number | null;
  }): Promise<EvaluatedBatch> {
    const { candidate, instances, phase, candidateId } = args;

    const dense = await evaluateCached({
      candidate,
      batch: instances.map((index) => valset[index] as Datum),
      ids: instances.map((index) => valIds[index] as string),
      split: "val",
      phase,
      candidateId,
    });

    const scores = new Array<number | undefined>(valset.length).fill(undefined);
    const objectiveScores = new Array<Record<string, number> | undefined>(
      valset.length,
    ).fill(undefined);

    instances.forEach((instance, position) => {
      scores[instance] = dense.scores[position];
      objectiveScores[instance] = dense.objectiveScores[position];
    });

    return { scores, objectiveScores };
  }

  /** The validation instances this candidate should be scored on. */
  function selectValInstances(candidate: Candidate): number[] {
    const selected = valEvaluationPolicy.selectInstances({
      valset,
      candidate,
      records,
      iteration,
      rng,
    });

    if (selected.length === 0) {
      throw new Error(
        "valEvaluationPolicy selected no validation instances; a candidate cannot be scored",
      );
    }
    return selected;
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
    evaluation: EvaluatedBatch;
    source: CandidateRecord["source"];
    updatedComponents: string[];
  }): CandidateRecord {
    const objectiveScores = meanObjectives(args.evaluation.objectiveScores);
    const record: CandidateRecord = {
      id: records.length,
      candidate: args.candidate,
      parentIds: args.parentIds,
      instanceScores: args.evaluation.scores,
      aggregateScore: mean(args.evaluation.scores),
      ...(objectiveScores === undefined ? {} : { objectiveScores }),
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

  /** Keeps the most recent rejections per component, oldest dropped first. */
  function rememberRejection(args: {
    proposed: ComponentPatch;
    parentScore: number;
    childScore: number;
  }): void {
    const { proposed, parentScore, childScore } = args;

    if (rejectedProposalMemory <= 0) {
      return;
    }
    for (const [component, text] of Object.entries(proposed)) {
      const history = rejectedProposals[component] ?? [];
      history.unshift({ text, parentScore, childScore });
      rejectedProposals[component] = history.slice(0, rejectedProposalMemory);
    }
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

  // A resumed run already has its seed scored; re-scoring it would charge the
  // budget twice for the same rollouts.
  if (records.length === 0) {
    const seedInstances = selectValInstances(seedCandidate);

    if (!budget.canAfford(seedInstances.length)) {
      throw new Error(
        `maxMetricCalls (${maxMetricCalls}) is smaller than the ${seedInstances.length} validation instances selected for scoring; the seed candidate cannot be scored`,
      );
    }

    const seedEvaluation = await evaluateValidation({
      candidate: seedCandidate,
      instances: seedInstances,
      phase: "seed",
      candidateId: 0,
    });
    addCandidate({
      candidate: seedCandidate,
      parentIds: [],
      evaluation: seedEvaluation,
      source: "seed",
      updatedComponents: [],
    });
    lastIterationAccepted = false;
    mergesDue = 0;
    await checkpoint();
  }

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

    const uniqueEvaluation = await evaluateCached({
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
        uniqueEvaluation.scores[position] as number,
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

    const mergeInstances = selectValInstances(proposal.candidate);
    if (
      !budget.canAfford(
        countUncached({
          candidate: proposal.candidate,
          ids: mergeInstances.map((index) => valIds[index] as string),
          split: "val",
        }),
      )
    ) {
      return "attempted";
    }

    const evaluation = await evaluateValidation({
      candidate: proposal.candidate,
      instances: mergeInstances,
      phase: "validation",
      candidateId: records.length,
    });

    const ancestor = records[proposal.ancestorId] as CandidateRecord;
    const record = addCandidate({
      candidate: proposal.candidate,
      parentIds: [...proposal.parentIds],
      evaluation,
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

    iterationBody: try {
      const mergeScheduled =
        mergeConfig.enabled && mergesDue > 0 && lastIterationAccepted;
      lastIterationAccepted = false;

      if (mergeScheduled && (await tryMerge()) === "attempted") {
        break iterationBody;
      }

      const selectionState = {
        scoreMatrix: records.map((record) => record.instanceScores),
        aggregateScores: records.map((record) => record.aggregateScore),
        objectiveScores: records.map((record) => record.objectiveScores),
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
        run: {
          iteration,
          phase: "minibatch",
          split: "train",
          candidateId: parent.id,
        },
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
        break iterationBody;
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
        rejectedProposals,
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
        break iterationBody;
      }

      if (
        !budget.canAfford(
          countUncached({ candidate: child, ids: batchIds, split: "train" }),
        )
      ) {
        stopReason = "budgetExhausted";
        break;
      }
      const childBatchEvaluation = await evaluateCached({
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
          childScores: childBatchEvaluation.scores,
        })
      ) {
        const parentScore = mean(parentEvaluation.scores);
        const childScore = mean(childBatchEvaluation.scores);

        rememberRejection({ proposed, parentScore, childScore });
        emit({
          type: "candidateRejected",
          iteration,
          parentId: parent.id,
          parentScore,
          childScore,
          source: "mutation",
        });
        break iterationBody;
      }

      const childInstances = selectValInstances(child);
      if (
        !budget.canAfford(
          countUncached({
            candidate: child,
            ids: childInstances.map((index) => valIds[index] as string),
            split: "val",
          }),
        )
      ) {
        stopReason = "budgetExhausted";
        break;
      }
      const childValEvaluation = await evaluateValidation({
        candidate: child,
        instances: childInstances,
        phase: "validation",
        candidateId: records.length,
      });

      const record = addCandidate({
        candidate: child,
        parentIds: [parent.id],
        evaluation: childValEvaluation,
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
    await checkpoint();
  }

  const bestCandidateId = valEvaluationPolicy.bestCandidate(records);
  const best = records[bestCandidateId] as CandidateRecord;

  emit({
    type: "finish",
    reason: stopReason,
    bestCandidateId,
    metricCalls: budget.spent(),
  });

  const perObjectiveBest = collectPerObjectiveBest(records);

  return {
    bestCandidate: best.candidate,
    bestScore: best.aggregateScore,
    bestCandidateId,
    candidates: records,
    paretoFrontier: collectDominatorIds(records).map(
      (id) => records[id] as CandidateRecord,
    ),
    ...(perObjectiveBest === undefined ? {} : { perObjectiveBest }),
    scoreMatrix: records.map((record) => [...record.instanceScores]),
    metricCalls: budget.spent(),
    cacheHits,
    iterations: iteration,
    stopReason,
    snapshot: takeSnapshot(),
  };
}

/** Mean of each objective over the instances that reported it. */
function meanObjectives(
  rows: readonly (Record<string, number> | undefined)[],
): Record<string, number> | undefined {
  const totals = new Map<string, { total: number; count: number }>();

  for (const row of rows) {
    for (const [objective, value] of Object.entries(row ?? {})) {
      const running = totals.get(objective) ?? { total: 0, count: 0 };
      running.total += value;
      running.count += 1;
      totals.set(objective, running);
    }
  }

  if (totals.size === 0) {
    return undefined;
  }
  return Object.fromEntries(
    [...totals].map(([objective, { total, count }]) => [
      objective,
      total / count,
    ]),
  );
}

function collectPerObjectiveBest(
  records: readonly CandidateRecord[],
): OptimizationResult["perObjectiveBest"] {
  const bests = objectiveBests(records.map((record) => record.objectiveScores));
  if (Object.keys(bests).length === 0) {
    return undefined;
  }

  return Object.fromEntries(
    Object.entries(bests).map(([objective, score]) => [
      objective,
      {
        score,
        candidateIds: records
          .filter((record) => record.objectiveScores?.[objective] === score)
          .map((record) => record.id),
      },
    ]),
  );
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

/**
 * Identifies the configuration a checkpoint was taken under. Resuming against
 * different data would score old candidates on new instances and quietly
 * corrupt the frontier, so the mismatch is refused instead.
 */
function runFingerprint(args: {
  seedCandidate: Candidate;
  trainIds: readonly string[];
  valIds: readonly string[];
  seed: number;
}): string {
  const { seedCandidate, trainIds, valIds, seed } = args;

  return JSON.stringify({
    seed,
    seedCandidate: candidateFingerprint(seedCandidate),
    trainIds,
    valIds,
  });
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
