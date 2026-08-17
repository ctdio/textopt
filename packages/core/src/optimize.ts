import { createBudget } from "./budget.js";
import { mapWithConcurrency } from "./concurrency.js";
import {
  candidateHash,
  createMemoryCache,
  evaluationCacheKey,
  stableHash,
} from "./cache.js";
import { proposeMerge, selectMergeSubsample } from "./merge.js";
import {
  buildInstanceFronts,
  mean,
  objectiveBests,
  pruneDominatedFronts,
  sum,
} from "./pareto.js";
import {
  type ReflectionLimits,
  type ReflectionPromptBuilder,
  createDefaultProposer,
} from "./reflection.js";
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
  EvaluationBatch,
  EvaluationCache,
  EvaluationPhase,
  EvaluationSplit,
  OptimizationResult,
  OptimizerEvent,
  OptimizerSnapshot,
  Reflector,
  RejectedProposal,
  SelectionState,
  StopReason,
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
  /**
   * How many mutations an iteration proposes, and what happens to them.
   *
   * One proposal per iteration is GEPA as published. Raising `perIteration`
   * samples the reflection model more than once against the same frontier —
   * more shots at an improvement, screened on cheap minibatches before any of
   * them costs a validation sweep — and `concurrency` is what turns that into
   * wall-clock savings rather than just more rollouts.
   */
  proposals?: {
    /** Proposals drawn per iteration, each with its own parent and minibatch. */
    perIteration?: number;
    /** How many of them may be in flight at once. Default 1. */
    concurrency?: number;
    /**
     * Which improving proposals are kept. "all" accepts every proposal that
     * beat its own parent, "best" only the largest improvement, `{ keep: n }`
     * the strongest n. Default "all".
     */
    selection?: "all" | "best" | { keep: number };
  };
  /**
   * Bounds on the reflection model, which no metric budget covers: reflection
   * calls are often the most expensive part of a run and the prompt carries
   * traces of unbounded size.
   */
  reflection?: ReflectionLimits & {
    /** Hard ceiling on reflection calls. The run stops once it is reached. */
    maxCalls?: number;
    /** Replaces the default prompt template. Ignored by custom proposers. */
    buildPrompt?: ReflectionPromptBuilder;
  };
  /**
   * Include cached instance scores in every checkpoint. Leaving them out keeps
   * snapshots small at the cost of a resumed run re-paying for rollouts it
   * cannot look up. Default true.
   */
  checkpointCache?: boolean;
  /**
   * Keep what each candidate produced on the validation instances it was
   * scored on, so the winning outputs can be read back without re-running.
   * Costs memory proportional to the outputs of every accepted candidate.
   */
  trackBestOutputs?: boolean;
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
interface ScoredBatch<Out> {
  scores: number[];
  objectiveScores: (Record<string, number> | undefined)[];
  /** Populated only under `trackBestOutputs`, and only for fresh rollouts. */
  outputs: (Out | undefined)[];
}

/** A scored batch spread over the whole valset, with gaps where it was not. */
interface EvaluatedBatch<Out> {
  scores: (number | undefined)[];
  objectiveScores: (Record<string, number> | undefined)[];
  outputs: (Out | undefined)[];
}

/** One mutation an iteration intends to make, drawn before any of them runs. */
interface ProposalPlan<Datum> {
  parent: CandidateRecord;
  batch: Datum[];
  batchIds: string[];
  componentsToUpdate: string[];
}

interface ScreenedProposal<Datum> {
  status: "screened";
  plan: ProposalPlan<Datum>;
  child: Candidate;
  proposed: ComponentPatch;
  parentScore: number;
  childScore: number;
  /** Total score gained over the parent on its own minibatch. */
  improvement: number;
  accepted: boolean;
}

type ProposalOutcome<Datum> =
  | ScreenedProposal<Datum>
  | { status: "skipped" }
  | { status: "budgetExhausted" }
  | { status: "reflectionExhausted" };

const DEFAULT_MINIBATCH_SIZE = 3;
const DEFAULT_REJECTED_PROPOSAL_MEMORY = 3;
const DEFAULT_MAX_MERGES = 5;
const MERGE_SUBSAMPLE_SIZE = 5;

/**
 * Raised when a reservation cannot be met mid-flight. A concurrent proposal
 * cannot check the budget and then spend it — another proposal may take the
 * remainder in between — so running out is reported where it happens and
 * turned into a stop reason by the loop.
 */
class BudgetExhausted extends Error {}

class ReflectionBudgetExhausted extends Error {}

export async function optimize<Datum, Traj = unknown, Out = unknown>(
  options: OptimizeOptions<Datum, Traj, Out>,
): Promise<OptimizationResult<Out>> {
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
    proposals,
    reflection,
    checkpointCache = true,
    trackBestOutputs = false,
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
  const proposalsPerIteration = proposals?.perIteration ?? 1;
  const proposalConcurrency = proposals?.concurrency ?? 1;
  const survivorsPerIteration = keepCount(proposals?.selection ?? "all");

  if (!Number.isInteger(proposalsPerIteration) || proposalsPerIteration < 1) {
    throw new Error(
      `proposals.perIteration must be a positive integer, received ${proposalsPerIteration}`,
    );
  }
  if (!Number.isInteger(proposalConcurrency) || proposalConcurrency < 1) {
    throw new Error(
      `proposals.concurrency must be a positive integer, received ${proposalConcurrency}`,
    );
  }

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
    adapter.proposeNewTexts?.bind(adapter) ??
    createDefaultProposer({
      ...(reflection?.buildPrompt === undefined
        ? {}
        : { buildPrompt: reflection.buildPrompt }),
      limits: {
        ...(reflection?.maxRecords === undefined
          ? {}
          : { maxRecords: reflection.maxRecords }),
        ...(reflection?.maxCharacters === undefined
          ? {}
          : { maxCharacters: reflection.maxCharacters }),
      },
    });

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
  if (resumeFrom?.sampler !== undefined) {
    batchSampler.restore?.(resumeFrom.sampler);
  }

  let reflectionCalls = resumeFrom?.reflectionCalls ?? 0;

  /**
   * The reflection budget is enforced at the call, not at the proposal: an
   * adapter's own proposer may make any number of calls, and a cap that only
   * counted proposals would not bound it.
   */
  const countedReflect: Reflector = async (args) => {
    if (
      reflection?.maxCalls !== undefined &&
      reflectionCalls >= reflection.maxCalls
    ) {
      throw new ReflectionBudgetExhausted();
    }
    reflectionCalls += 1;
    return reflect(args);
  };

  const records: CandidateRecord[] = [...(resumeFrom?.records ?? [])];
  const seenCandidates = new Set(
    records.map((record) => candidateFingerprint(record.candidate)),
  );
  /**
   * Children claimed by a proposal that has not been committed yet. Two
   * proposals in the same iteration routinely converge on the same text, and
   * without a claim both would pay to screen it.
   */
  const claimedCandidates = new Set<string>();
  const outputsByCandidate = new Map<number, (Out | undefined)[]>();
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
    const cached = checkpointCache ? evaluationCache?.entries?.() : undefined;
    const samplerState = batchSampler.state?.();

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
      reflectionCalls,
      cacheHits,
      ...(samplerState === undefined ? {} : { sampler: samplerState }),
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
  }): Promise<ScoredBatch<Out>> {
    const { candidate, batch, ids, split, phase, candidateId } = args;

    const hash = candidateHash(candidate);
    const scores = new Array<number>(batch.length);
    const objectiveScores = new Array<Record<string, number> | undefined>(
      batch.length,
    );
    const outputs = new Array<Out | undefined>(batch.length).fill(undefined);
    const pendingIndices: number[] = [];

    for (let index = 0; index < batch.length; index += 1) {
      const cached = evaluationCache?.get(
        evaluationCacheKey({
          hash,
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
      if (!budget.reserve(pendingIndices.length)) {
        throw new BudgetExhausted();
      }

      let evaluation: EvaluationBatch<Traj, Out>;
      try {
        evaluation = await adapter.evaluate({
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
      } catch (err) {
        // Nothing was measured, so nothing is owed.
        budget.refund(pendingIndices.length);
        throw err;
      }

      pendingIndices.forEach((batchIndex, resultIndex) => {
        const score = evaluation.scores[resultIndex] as number;
        const objectives = evaluation.objectiveScores?.[resultIndex];
        scores[batchIndex] = score;
        objectiveScores[batchIndex] = objectives;
        if (trackBestOutputs) {
          outputs[batchIndex] = evaluation.outputs[resultIndex];
        }

        // A transient score says nothing about the candidate, so caching it
        // would pin this instance to an infrastructure failure forever.
        if (evaluation.transient?.[resultIndex] === true) {
          return;
        }
        evaluationCache?.set(
          evaluationCacheKey({
            hash,
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

    return { scores, objectiveScores, outputs };
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
  }): Promise<EvaluatedBatch<Out>> {
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
    const outputs = new Array<Out | undefined>(valset.length).fill(undefined);

    instances.forEach((instance, position) => {
      scores[instance] = dense.scores[position];
      objectiveScores[instance] = dense.objectiveScores[position];
      outputs[instance] = dense.outputs[position];
    });

    return { scores, objectiveScores, outputs };
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
    const hash = candidateHash(candidate);
    return ids.filter(
      (id) =>
        evaluationCache.get(
          evaluationCacheKey({ hash, instanceId: id, split }),
        ) === undefined,
    ).length;
  }

  function addCandidate(args: {
    candidate: Candidate;
    parentIds: number[];
    evaluation: EvaluatedBatch<Out>;
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
    if (trackBestOutputs) {
      outputsByCandidate.set(record.id, args.evaluation.outputs);
    }

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

  /**
   * Claims a child for the proposal holding it. False when the same text has
   * already been recorded or claimed, which is the signal to abandon the
   * proposal rather than pay to screen a duplicate.
   */
  function claimCandidate(child: Candidate): boolean {
    const fingerprint = candidateFingerprint(child);

    if (seenCandidates.has(fingerprint) || claimedCandidates.has(fingerprint)) {
      return false;
    }
    claimedCandidates.add(fingerprint);
    return true;
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

  let stopReason: StopReason = "budgetExhausted";

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
        reason: "worse",
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

  /**
   * Draws every proposal an iteration will make, before any of them runs.
   *
   * Each draw consumes the random stream — parent, minibatch, component — so
   * they all happen here, in order, on the same frontier snapshot. Doing it
   * inside the concurrent phase instead would make the whole run's trajectory
   * depend on which network call returned first.
   */
  function planProposals(): ProposalPlan<Datum>[] {
    const state: SelectionState = {
      scoreMatrix: records.map((record) => record.instanceScores),
      aggregateScores: records.map((record) => record.aggregateScore),
      objectiveScores: records.map((record) => record.objectiveScores),
    };
    const plans: ProposalPlan<Datum>[] = [];

    for (let slot = 0; slot < proposalsPerIteration; slot += 1) {
      const parent = records[
        candidateSelector({ state, rng })
      ] as CandidateRecord;
      const batchIndices = batchSampler({
        trainset,
        // Each proposal takes the next minibatch in the sampler's schedule, so
        // siblings in one iteration diagnose different failures.
        iteration: iteration * proposalsPerIteration + slot,
        rng,
      });
      const componentsToUpdate = componentSelector({
        candidate: parent.candidate,
        cursor: parent.componentCursor,
        iteration,
        rng,
      });
      parent.componentCursor =
        (parent.componentCursor + 1) %
        Math.max(1, Object.keys(parent.candidate).length);

      plans.push({
        parent,
        batch: batchIndices.map((index) => trainset[index] as Datum),
        batchIds: batchIndices.map((index) => trainIds[index] as string),
        componentsToUpdate,
      });
    }
    return plans;
  }

  /**
   * Reflects on one parent and screens the result on that parent's own
   * minibatch. Everything here is IO the run can overlap; nothing here mutates
   * the candidate pool, which is what makes overlapping it safe.
   */
  async function runProposal(
    plan: ProposalPlan<Datum>,
  ): Promise<ProposalOutcome<Datum>> {
    const { parent, batch, batchIds, componentsToUpdate } = plan;

    // Traces are required for reflection, so this evaluation always runs and
    // is never served from the cache.
    if (!budget.reserve(batch.length)) {
      return { status: "budgetExhausted" };
    }

    let parentEvaluation: EvaluationBatch<Traj, Out>;
    try {
      parentEvaluation = await adapter.evaluate({
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
      assertScores({ scores: parentEvaluation.scores, expected: batch.length });
    } catch (err) {
      budget.refund(batch.length);
      throw err;
    }

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
      return { status: "skipped" };
    }

    const reflectiveDataset = await adapter.makeReflectiveDataset({
      candidate: parent.candidate,
      batch,
      evaluation: parentEvaluation,
      componentsToUpdate,
    });

    let proposed: ComponentPatch;
    try {
      proposed = await propose({
        candidate: parent.candidate,
        reflectiveDataset,
        componentsToUpdate,
        rejectedProposals,
        reflect: countedReflect,
        signal,
      });
    } catch (err) {
      if (err instanceof ReflectionBudgetExhausted) {
        return { status: "reflectionExhausted" };
      }
      throw err;
    }

    const child: Candidate = { ...parent.candidate, ...proposed };
    const changed = Object.keys(proposed).length > 0 && claimCandidate(child);

    emit({
      type: "proposal",
      iteration,
      parentId: parent.id,
      componentsToUpdate: [...componentsToUpdate],
      changed,
    });

    if (!changed) {
      return { status: "skipped" };
    }

    let childEvaluation: ScoredBatch<Out>;
    try {
      childEvaluation = await evaluateCached({
        candidate: child,
        batch,
        ids: batchIds,
        split: "train",
        phase: "minibatch",
        candidateId: null,
      });
    } catch (err) {
      if (err instanceof BudgetExhausted) {
        return { status: "budgetExhausted" };
      }
      throw err;
    }

    return {
      status: "screened",
      plan,
      child,
      proposed,
      parentScore: mean(parentEvaluation.scores),
      childScore: mean(childEvaluation.scores),
      improvement: sum(childEvaluation.scores) - sum(parentEvaluation.scores),
      accepted: acceptance({
        parentScores: parentEvaluation.scores,
        childScores: childEvaluation.scores,
      }),
    };
  }

  /**
   * Turns screened proposals into candidates: rejections first, then the
   * survivors the selection policy keeps, each paying for its own validation
   * sweep. Returns a stop reason when the iteration ran the run out of budget.
   */
  async function commitProposals(
    outcomes: readonly ProposalOutcome<Datum>[],
  ): Promise<StopReason | undefined> {
    let stop: StopReason | undefined;
    const improved: ScreenedProposal<Datum>[] = [];

    for (const outcome of outcomes) {
      if (outcome.status === "skipped") {
        continue;
      }
      if (outcome.status === "budgetExhausted") {
        stop ??= "budgetExhausted";
        continue;
      }
      if (outcome.status === "reflectionExhausted") {
        stop ??= "reflectionBudgetExhausted";
        continue;
      }
      if (!outcome.accepted) {
        rememberRejection({
          proposed: outcome.proposed,
          parentScore: outcome.parentScore,
          childScore: outcome.childScore,
        });
        emit({
          type: "candidateRejected",
          iteration,
          parentId: outcome.plan.parent.id,
          parentScore: outcome.parentScore,
          childScore: outcome.childScore,
          source: "mutation",
          reason: "worse",
        });
        continue;
      }
      improved.push(outcome);
    }

    const survivors = selectSurvivors(improved);

    for (const outcome of improved) {
      if (survivors.includes(outcome)) {
        continue;
      }
      // Losing to a stronger sibling is not evidence the idea was bad, so it
      // is reported but never fed back to reflection as a dead end.
      emit({
        type: "candidateRejected",
        iteration,
        parentId: outcome.plan.parent.id,
        parentScore: outcome.parentScore,
        childScore: outcome.childScore,
        source: "mutation",
        reason: "notSelected",
      });
    }

    // Ids are assigned before the sweeps so concurrent validations can report
    // the id their candidate will be recorded under.
    const baseId = records.length;
    const scheduled = survivors.map((outcome, index) => ({
      outcome,
      candidateId: baseId + index,
      instances: selectValInstances(outcome.child),
    }));

    const scored = await mapWithConcurrency({
      items: scheduled,
      limit: proposalConcurrency,
      signal,
      task: async (item) => {
        try {
          return {
            item,
            evaluation: await evaluateValidation({
              candidate: item.outcome.child,
              instances: item.instances,
              phase: "validation",
              candidateId: item.candidateId,
            }),
          };
        } catch (err) {
          if (err instanceof BudgetExhausted) {
            return { item, evaluation: undefined };
          }
          throw err;
        }
      },
    });

    for (const { item, evaluation } of scored) {
      // Stopping at the first shortfall keeps the ids handed to the adapter
      // aligned with the ids the records actually get.
      if (evaluation === undefined) {
        stop ??= "budgetExhausted";
        break;
      }
      const record = addCandidate({
        candidate: item.outcome.child,
        parentIds: [item.outcome.plan.parent.id],
        evaluation,
        source: "mutation",
        updatedComponents: Object.keys(item.outcome.proposed),
      });
      emit({
        type: "candidateAccepted",
        iteration,
        candidateId: record.id,
        parentIds: record.parentIds,
        aggregateScore: record.aggregateScore,
        source: "mutation",
      });
    }

    return stop;
  }

  /** The improving proposals an iteration keeps, in the order they were made. */
  function selectSurvivors(
    improved: readonly ScreenedProposal<Datum>[],
  ): ScreenedProposal<Datum>[] {
    if (improved.length <= survivorsPerIteration) {
      return [...improved];
    }

    return improved
      .map((outcome, index) => ({ outcome, index }))
      .sort(
        (a, b) =>
          b.outcome.improvement - a.outcome.improvement || a.index - b.index,
      )
      .slice(0, survivorsPerIteration)
      .sort((a, b) => a.index - b.index)
      .map((entry) => entry.outcome);
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
    if (
      reflection?.maxCalls !== undefined &&
      reflectionCalls >= reflection.maxCalls
    ) {
      stopReason = "reflectionBudgetExhausted";
      break;
    }

    const spentBeforeIteration = budget.spent();
    let pendingStop: StopReason | undefined;
    claimedCandidates.clear();

    try {
      const mergeScheduled =
        mergeConfig.enabled && mergesDue > 0 && lastIterationAccepted;
      lastIterationAccepted = false;

      const merged = mergeScheduled && (await tryMerge()) === "attempted";

      if (!merged) {
        const plans = planProposals();

        emit({
          type: "iterationStart",
          iteration,
          parentIds: plans.map((plan) => plan.parent.id),
        });

        const outcomes = await mapWithConcurrency({
          items: plans,
          limit: proposalConcurrency,
          task: runProposal,
          signal,
        });
        pendingStop = await commitProposals(outcomes);
      }
    } catch (err) {
      // An adapter that honours the signal reports cancellation by throwing.
      // That is the run ending on request, not a failure to report or rethrow.
      if (signal?.aborted) {
        stopReason = "aborted";
        break;
      }
      // Any reservation that could not be met ends the run rather than the
      // iteration: there is nothing left to spend on the next one either.
      if (err instanceof BudgetExhausted) {
        stopReason = "budgetExhausted";
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

    if (pendingStop !== undefined) {
      stopReason = pendingStop;
      break;
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
  const bestOutputs = outputsByCandidate.get(bestCandidateId);

  return {
    bestCandidate: best.candidate,
    bestScore: best.aggregateScore,
    bestCandidateId,
    ...(bestOutputs === undefined ? {} : { bestOutputs }),
    candidates: records,
    paretoFrontier: collectDominatorIds(records).map(
      (id) => records[id] as CandidateRecord,
    ),
    ...(perObjectiveBest === undefined ? {} : { perObjectiveBest }),
    scoreMatrix: records.map((record) => [...record.instanceScores]),
    metricCalls: budget.spent(),
    reflectionCalls,
    cacheHits,
    iterations: iteration,
    stopReason,
    snapshot: takeSnapshot(),
  };
}

/** How many improving proposals an iteration is allowed to keep. */
function keepCount(selection: "all" | "best" | { keep: number }): number {
  if (selection === "all") {
    return Number.POSITIVE_INFINITY;
  }
  if (selection === "best") {
    return 1;
  }
  if (!Number.isInteger(selection.keep) || selection.keep < 1) {
    throw new Error(
      `proposals.selection.keep must be a positive integer, received ${selection.keep}`,
    );
  }
  return selection.keep;
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

  // Hashed, not embedded: this goes into every snapshot, and only ever gets
  // compared for equality.
  return stableHash({
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

/**
 * Names an instance by a hash of its content rather than by the content
 * itself: the id ends up inside every cache key and inside the checkpoint
 * fingerprint, and embedding whole examples there costs memory proportional to
 * the dataset for no benefit. Data that will not serialize falls back to its
 * position, which is stable for as long as the dataset order is.
 */
function defaultInstanceId(args: { datum: unknown; index: number }): string {
  const hash = stableHash(args.datum);
  return hash === "" ? String(args.index) : hash;
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
