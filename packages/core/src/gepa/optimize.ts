import { createDeadline } from "../deadline.js";
import { createBudget } from "../budget.js";
import { createMemoryCache, defaultInstanceId } from "../cache.js";
import {
  assertResumable,
  candidateFingerprint,
  runFingerprint,
} from "../checkpoint.js";
import type { EvaluationCache } from "../cache.js";
import { mapWithConcurrency } from "../concurrency.js";
import {
  BudgetExhausted,
  costExhausted,
  createEvaluator,
  measuredMean,
} from "../evaluation.js";
import type { ScoredBatch } from "../evaluation.js";
import { mean, sum } from "../math.js";
import type {
  Optimizer,
  OptimizerResult,
  OptimizerTask,
} from "../optimizer.js";
import { createEmitter, flushReporters, instanceRow } from "../reporting.js";
import type { Reporter } from "../reporting.js";
import { createSeededRng } from "../rng.js";
import { createEpochShuffledSampler } from "../sampling.js";
import type { BatchSampler } from "../sampling.js";
import { componentNames } from "../types.js";
import type {
  Candidate,
  EvaluationPhase,
  EvaluationSplit,
  TextModel,
} from "../types.js";
import { proposeMerge, selectMergeSubsample } from "./merge.js";
import {
  buildInstanceFronts,
  objectiveBests,
  pruneDominatedFronts,
} from "./pareto.js";
import {
  type ReflectionPromptBuilder,
  createDefaultProposer,
} from "./reflection.js";
import {
  fullEvaluationPolicy,
  improvementAcceptance,
  paretoSelector,
  roundRobinComponentSelector,
} from "./strategies.js";
import type {
  AcceptancePolicy,
  CandidateRecord,
  CandidateSelector,
  ComponentPatch,
  ComponentSelector,
  GepaAdapter,
  GepaEvent,
  GepaSnapshot,
  GepaStopReason,
  RejectedProposal,
  SelectionState,
  ValEvaluationPolicy,
} from "./types.js";

/**
 * How GEPA searches. Immutable, reusable across runs, and free of both the
 * component names and the datum type — the honest line between this and
 * `GepaTask` is "type-free and stateless" rather than "how it searches".
 *
 * Holds no run state: the candidate pool, the budget spent, the position of the
 * random stream, rejected proposals and merge bookkeeping all live in
 * `GepaSnapshot`.
 */
export interface GepaConfig {
  minibatchSize?: number;
  maxIterations?: number;
  seed?: number;
  candidateSelector?: CandidateSelector;
  acceptance?: AcceptancePolicy;
  /**
   * System-aware merge. Enabled by default for multi-component candidates,
   * where two lineages can improve different components independently.
   *
   * The reference defaults `use_merge=False`, and the paper reports merge as
   * the separate GEPA+Merge variant rather than as part of GEPA — a variant
   * that helped on most models it was tried on and hurt on one. On by default
   * here because the case it needs, several components moving along different
   * lineages, is the case this library is usually pointed at. The cost is that
   * a multi-component run is GEPA+Merge unless this is turned off, which
   * spends rollouts on merge attempts and consumes the random stream
   * differently, so trajectories will not line up with a reference run at the
   * same seed. Set `enabled: false` for GEPA as published.
   */
  merge?: {
    enabled?: boolean;
    maxInvocations?: number;
    /**
     * Validation instances two lineages must share before they may be merged.
     * Default 5, GEPA's `val_overlap_floor`. A merge is judged only on
     * instances both parents were scored on, so below this the gate deciding
     * whether to keep the child is reading noise. A validation set smaller than the
     * floor can never merge.
     */
    valOverlapFloor?: number;
  };
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
   *
   * Not in the paper or the reference implementation, where a rejected
   * proposal fires a callback and is otherwise forgotten: whenever a proposal
   * has been rejected, the prompt this builds is not the published one. Set 0
   * for GEPA as written.
   *
   * It is not the only default that departs from the reference — the
   * evaluation cache and, for multi-component seeds, merge are both on here
   * and off there. See those options.
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
  reflection?: {
    /** Hard ceiling on reflection calls. The run stops once it is reached. */
    maxCalls?: number;
    /** Records shown per component. The worst scoring ones are kept. */
    maxRecords?: number;
    /** Rough ceiling on the characters the records serialize to. */
    maxCharacters?: number;
    /** Replaces the default prompt template. Ignored by custom proposers. */
    buildPrompt?: ReflectionPromptBuilder;
    /**
     * Prompt templates rotated one per proposal, so raising
     * `proposals.perIteration` samples different directions rather than the
     * same one repeatedly. `diverseReflectionStrategies()` is a ready set.
     * Mutually exclusive with `buildPrompt`. Ignored by custom proposers.
     */
    strategies?: readonly ReflectionPromptBuilder[];
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
  /** Rethrow adapter failures instead of skipping the iteration. Default true. */
  raiseOnError?: boolean;
}

/**
 * One run: what is being optimized, over what data, with what run-scoped state
 * and IO.
 *
 * `seedCandidate` and `trainingSet` are the inference sites for the component
 * names and the datum type; every other position is `NoInfer`, so it is checked
 * against them instead of widening them.
 */
export interface GepaTask<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> extends OptimizerTask<Datum, Trajectory, Output, K> {
  adapter: GepaAdapter<Datum, Trajectory, Output, NoInfer<K>>;
  reflect: TextModel;
  componentSelector?: ComponentSelector<NoInfer<K>>;
  batchSampler?: BatchSampler<NoInfer<Datum>>;
  /**
   * Which validation instances each candidate is scored on. Defaults to a full
   * sweep per accepted candidate, which is what makes the frontier exact.
   */
  valEvaluationPolicy?: ValEvaluationPolicy<NoInfer<Datum>, NoInfer<K>>;
  instanceId?: (args: { datum: NoInfer<Datum>; index: number }) => string;
  /**
   * Pass `false` to disable caching entirely.
   *
   * On by default, where the reference defaults `cache_evaluation=False`. A
   * cache hit is free and uncharged, so the same `maxMetricCalls` buys a
   * longer run here than there — the budget counts fresh rollouts, which is
   * what costs money, rather than scorings. Pass `false` to compare rollout
   * counts against a reference run directly.
   */
  cache?: EvaluationCache | false;
  /**
   * Where the run's events go. An array because a run usually has more than
   * one audience — a progress line on the terminal and a permanent record
   * somewhere else — and teeing one callback by hand is how one of them ends
   * up silently dropped.
   */
  reporters?: readonly Reporter<GepaEvent<NoInfer<K>>>[];
  /**
   * Called with a resumable snapshot after the seed is scored and after every
   * iteration. Persist it and a killed run costs the last iteration, not all
   * of them.
   */
  onCheckpoint?: (snapshot: GepaSnapshot) => void | Promise<void>;
  /** Snapshot to continue from, instead of starting at the seed candidate. */
  resumeFrom?: GepaSnapshot;
}

export interface GepaResult<
  K extends string = string,
  Output = unknown,
> extends OptimizerResult<K, GepaStopReason, Output> {
  bestCandidateId: number;
  candidates: CandidateRecord<K>[];
  paretoFrontier: CandidateRecord<K>[];
  /**
   * Per objective: the best value reached and every candidate that reached it.
   * Absent when the adapter reports no objective scores.
   */
  perObjectiveBest?: Record<string, { score: number; candidateIds: number[] }>;
  scoreMatrix: (number | undefined)[][];
  iterations: number;
  /** Calls made to the reflection model, which no metric budget covers. */
  reflectionCalls: number;
  cacheHits: number;
  /** State as of the last iteration, ready to hand back as `resumeFrom`. */
  snapshot: GepaSnapshot;
}

/** A scored batch spread over the whole validationSet, with gaps where it was not. */
interface EvaluatedBatch<Output> {
  scores: (number | undefined)[];
  objectiveScores: (Record<string, number> | undefined)[];
  outputs: (Output | undefined)[];
}

/** One mutation an iteration intends to make, drawn before any of them runs. */
interface ProposalPlan<Datum, K extends string> {
  parent: CandidateRecord<K>;
  batch: Datum[];
  batchIds: string[];
  componentsToUpdate: K[];
  /** This proposal's position in the run, counted across all iterations. */
  attempt: number;
}

interface ScreenedProposal<Datum, K extends string> {
  status: "screened";
  plan: ProposalPlan<Datum, K>;
  child: Candidate<K>;
  proposed: ComponentPatch<K>;
  parentScore: number;
  childScore: number;
  /** Total score gained over the parent on its own minibatch. */
  improvement: number;
  accepted: boolean;
}

type ProposalOutcome<Datum, K extends string> =
  | ScreenedProposal<Datum, K>
  | { status: "skipped" }
  | { status: "budgetExhausted" }
  | { status: "reflectionExhausted" };

const DEFAULT_MINIBATCH_SIZE = 3;
const DEFAULT_REJECTED_PROPOSAL_MEMORY = 3;
const DEFAULT_MAX_MERGES = 5;
const MERGE_SUBSAMPLE_SIZE = 5;

class ReflectionBudgetExhausted extends Error {}

/**
 * Reflective prompt evolution: propose, screen on a minibatch, promote what
 * survives, and track the Pareto frontier of everything promoted.
 *
 * One instance is a configured search that can be run against any number of
 * tasks. It holds no run state, so two runs never share a shuffle position, a
 * budget or a candidate pool.
 */
export class GepaOptimizer implements Optimizer<GepaStopReason> {
  readonly #config: GepaConfig;

  constructor(config: GepaConfig = {}) {
    assertGepaConfig(config);
    this.#config = config;
  }

  async optimize<
    Datum,
    Trajectory = unknown,
    Output = unknown,
    const K extends string = string,
  >(
    task: GepaTask<Datum, Trajectory, Output, K>,
  ): Promise<GepaResult<K, Output>> {
    try {
      return await runGepa({ config: this.#config, task });
    } finally {
      // In a finally rather than after the run: a reporter that buffers has
      // the most to say about a run that aborted or threw, and that is exactly
      // the run that never reaches its last line.
      await flushReporters(task.reporters ?? []);
    }
  }
}

async function runGepa<Datum, Trajectory, Output, K extends string>(args: {
  config: GepaConfig;
  task: GepaTask<Datum, Trajectory, Output, K>;
}): Promise<GepaResult<K, Output>> {
  const { config, task } = args;

  const {
    minibatchSize = DEFAULT_MINIBATCH_SIZE,
    maxIterations = Number.POSITIVE_INFINITY,
    seed = 0,
    candidateSelector = paretoSelector(),
    acceptance = improvementAcceptance(),
    merge,
    skipPerfectScore = true,
    perfectScore = 1,
    rejectedProposalMemory = DEFAULT_REJECTED_PROPOSAL_MEMORY,
    proposals,
    reflection,
    checkpointCache = true,
    trackBestOutputs = false,
    raiseOnError = true,
  } = config;

  // Every stateful default is built here rather than in the constructor: the
  // epoch-shuffled sampler carries a shuffle position and the memory cache
  // carries scores, so building either once per optimizer would make two runs
  // share the state of the first.
  const {
    seedCandidate,
    trainingSet,
    validationSet = trainingSet,
    testSet,
    adapter,
    reflect,
    maxMetricCalls,
    componentSelector = roundRobinComponentSelector<K>(),
    batchSampler = createEpochShuffledSampler<Datum>({ minibatchSize }),
    valEvaluationPolicy = fullEvaluationPolicy<Datum, K>(),
    cache,
    cacheNamespace,
    retry,
    maxCostUsd,
    maxWallClockMs,
    instanceId = defaultInstanceId,
    reporters = [],
    onCheckpoint,
    resumeFrom,
    signal,
  } = task;

  const deadline = createDeadline({ maxWallClockMs });
  const seedComponents = componentNames(seedCandidate);
  const mergeConfig = {
    enabled: merge?.enabled ?? seedComponents.length > 1,
    maxInvocations: merge?.maxInvocations ?? DEFAULT_MAX_MERGES,
    ...(merge?.valOverlapFloor === undefined
      ? {}
      : { valOverlapFloor: merge.valOverlapFloor }),
  };
  const proposalsPerIteration = proposals?.perIteration ?? 1;
  const proposalConcurrency = proposals?.concurrency ?? 1;
  const survivorsPerIteration = keepCount(proposals?.selection ?? "all");

  if (trainingSet.length === 0) {
    throw new Error("optimize requires a non-empty trainingSet");
  }
  if (validationSet.length === 0) {
    throw new Error(
      "optimize requires a non-empty validationSet; the Pareto frontier is tracked over validation instances",
    );
  }
  if (seedComponents.length === 0) {
    throw new Error(
      "optimize requires a seed candidate with at least one component",
    );
  }
  // Omitting the test set means "do not measure"; passing an empty one means a
  // split was computed and came out empty, which would report a mean over
  // nothing as a held-out score of 0.
  if (testSet !== undefined && testSet.length === 0) {
    throw new Error(
      "optimize requires a non-empty testSet when one is given; omit it to skip held-out evaluation",
    );
  }

  const evaluationCache =
    cache === false ? undefined : (cache ?? createMemoryCache());
  const propose =
    adapter.proposeNewTexts?.bind(adapter) ??
    createDefaultProposer<K>({
      ...(reflection?.buildPrompt === undefined
        ? {}
        : { buildPrompt: reflection.buildPrompt }),
      ...(reflection?.strategies === undefined
        ? {}
        : { strategies: reflection.strategies }),
      limits: {
        ...(reflection?.maxRecords === undefined
          ? {}
          : { maxRecords: reflection.maxRecords }),
        ...(reflection?.maxCharacters === undefined
          ? {}
          : { maxCharacters: reflection.maxCharacters }),
      },
    });

  const trainingIds = trainingSet.map((datum, index) =>
    instanceId({ datum, index }),
  );
  const validationIds = validationSet.map((datum, index) =>
    instanceId({ datum, index }),
  );
  const testIds =
    testSet?.map((datum, index) => instanceId({ datum, index })) ?? [];

  const fingerprint = runFingerprint({
    seedCandidate,
    trainingIds,
    validationIds,
    seed,
    ...(cacheNamespace === undefined ? {} : { cacheNamespace }),
  });
  assertResumable({
    fingerprint,
    ...(resumeFrom === undefined ? {} : { snapshot: resumeFrom }),
  });

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
  const countedReflect: TextModel = async (args) => {
    if (
      reflection?.maxCalls !== undefined &&
      reflectionCalls >= reflection.maxCalls
    ) {
      throw new ReflectionBudgetExhausted();
    }
    reflectionCalls += 1;
    return reflect(args);
  };

  // Copied on the way in as well as on the way out: the caller owns the object
  // it persisted, and the run writes component cursors and rejections back into
  // exactly these structures.
  const records: CandidateRecord<K>[] = restoreRecords({
    records: resumeFrom?.records ?? [],
    seedCandidate,
  });
  const seenCandidates = new Set(
    records.map((record) => candidateFingerprint(record.candidate)),
  );
  const outputsByCandidate = new Map<number, (Output | undefined)[]>();
  const rejectedProposals = restoreRejections({
    rejections: resumeFrom?.rejectedProposals ?? {},
    components: seedComponents,
  });
  let iteration = resumeFrom?.iteration ?? 0;

  const mergeAttempts = new Set<string>(resumeFrom?.merge.attempts);
  const mergeDescriptions = new Set<string>(resumeFrom?.merge.descriptions);
  let mergesDue = resumeFrom?.merge.due ?? 0;
  let totalMergesTested = resumeFrom?.merge.tested ?? 0;
  let lastIterationAccepted = resumeFrom?.merge.lastIterationAccepted ?? false;

  const emit = createEmitter<GepaEvent<K>>(reporters);

  /**
   * Everything an acceptance means, in one event: the text, the aggregate, and
   * the row it put on the frontier. Emitted from one place because the merge
   * path and the mutation path accept candidates separately, and a payload
   * assembled twice is a payload that drifts.
   */
  function emitAccepted(record: CandidateRecord<K>): void {
    const outputs = outputsByCandidate.get(record.id);

    emit({
      type: "candidateAccepted",
      iteration,
      candidateId: record.id,
      parentIds: record.parentIds,
      aggregateScore: record.aggregateScore,
      source: record.source,
      candidate: record.candidate,
      instanceScores: record.instanceScores,
      ...(outputs === undefined ? {} : { outputs }),
    });
  }

  /**
   * Copies everything mutable: a snapshot handed to `onCheckpoint` is a record
   * of that moment, and would otherwise keep growing as the run continues.
   */
  function takeSnapshot(): GepaSnapshot {
    const cached = checkpointCache ? evaluationCache?.entries?.() : undefined;
    const samplerState = batchSampler.state?.();

    return {
      version: 1,
      fingerprint,
      records: copyRecords(records),
      iteration,
      metricCalls: budget.spent(),
      reflectionCalls,
      cacheHits: evaluator.cacheHits(),
      usage: evaluator.usage(),
      ...(samplerState === undefined ? {} : { sampler: samplerState }),
      rejectedProposals: snapshotRejections({
        rejections: rejectedProposals,
        components: seedComponents,
      }),
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

  const evaluator = createEvaluator<Datum, Trajectory, Output, K>({
    adapter,
    budget,
    ...(retry === undefined ? {} : { retry }),
    ...(cacheNamespace === undefined ? {} : { cacheNamespace }),
    ...(evaluationCache === undefined ? {} : { cache: evaluationCache }),
    trackOutputs: trackBestOutputs,
    cacheHits: resumeFrom?.cacheHits ?? 0,
    ...(resumeFrom?.usage === undefined ? {} : { usage: resumeFrom.usage }),
    ...(signal === undefined ? {} : { signal }),
    onEvaluation: (event) => emit({ type: "evaluation", ...event }),
  });

  evaluator.restore(resumeFrom?.cache ?? []);

  /**
   * Evaluates a candidate on a batch at the current iteration. The cache, the
   * budget and the transient-failure rules live in the shared evaluator; what
   * belongs to GEPA is only which batch, and when.
   */
  async function evaluateCached(args: {
    candidate: Candidate<K>;
    batch: readonly Datum[];
    ids: readonly string[];
    split: EvaluationSplit;
    phase: EvaluationPhase;
    candidateId: number | null;
    charge?: boolean;
  }): Promise<ScoredBatch<Output>> {
    return evaluator.evaluate({ ...args, iteration });
  }

  /**
   * Scores a candidate on the validation instances the policy selects, and
   * spreads the result back over the full validation set — instances the
   * policy skipped stay `undefined`, which every consumer reads as unknown
   * rather than as a zero.
   */
  async function evaluateValidation(args: {
    candidate: Candidate<K>;
    instances: readonly number[];
    phase: EvaluationPhase;
    candidateId: number | null;
  }): Promise<EvaluatedBatch<Output>> {
    const { candidate, instances, phase, candidateId } = args;

    const dense = await evaluateCached({
      candidate,
      batch: instances.map((index) => validationSet[index] as Datum),
      ids: instances.map((index) => validationIds[index] as string),
      split: "val",
      phase,
      candidateId,
    });

    const scores = new Array<number | undefined>(validationSet.length).fill(
      undefined,
    );
    const objectiveScores = new Array<Record<string, number> | undefined>(
      validationSet.length,
    ).fill(undefined);
    const outputs = new Array<Output | undefined>(validationSet.length).fill(
      undefined,
    );

    instances.forEach((instance, position) => {
      // A transient row measured the infrastructure, not the candidate, and
      // nothing ever re-measures a promoted candidate. Left unknown it costs
      // one instance of coverage; recorded it is a permanent zero on the
      // frontier.
      if (dense.transient[position] === true) {
        return;
      }
      scores[instance] = dense.scores[position];
      objectiveScores[instance] = dense.objectiveScores[position];
      outputs[instance] = dense.outputs[position];
    });

    return { scores, objectiveScores, outputs };
  }

  /** The validation instances this candidate should be scored on. */
  function selectValInstances(candidate: Candidate<K>): number[] {
    const selected = valEvaluationPolicy.selectInstances({
      validationSet,
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

  function addCandidate(args: {
    candidate: Candidate<K>;
    parentIds: number[];
    evaluation: EvaluatedBatch<Output>;
    source: CandidateRecord["source"];
    updatedComponents: K[];
  }): CandidateRecord<K> {
    const objectiveScores = meanObjectives({
      rows: args.evaluation.objectiveScores,
      scores: args.evaluation.scores,
    });
    const record: CandidateRecord<K> = {
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

    // A new frontier member is what makes a merge worth attempting, so an
    // accepted *mutation* schedules one. An accepted merge does not: it
    // recombines components two lineages already held, so there is no new
    // material for a second merge to find, and the reference clears the flag
    // on the merge path and returns to mutation rather than chaining.
    if (args.source !== "merge") {
      lastIterationAccepted = true;
      if (
        mergeConfig.enabled &&
        totalMergesTested < mergeConfig.maxInvocations
      ) {
        mergesDue += 1;
      }
    }

    return record;
  }

  /** Keeps the most recent rejections per component, oldest dropped first. */
  function rememberRejection(args: {
    proposed: ComponentPatch<K>;
    parentScore: number;
    childScore: number;
  }): void {
    const { proposed, parentScore, childScore } = args;

    if (rejectedProposalMemory <= 0) {
      return;
    }
    for (const component of componentNames(proposed)) {
      const text = proposed[component];
      if (text === undefined) {
        continue;
      }
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
    components: seedComponents,
    validationSetSize: validationSet.length,
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
    // Announced like any other candidate: it is the baseline every later
    // acceptance is read against, and a reporter that never hears about it has
    // a run whose first row is missing.
    emitAccepted(
      addCandidate({
        candidate: seedCandidate,
        parentIds: [],
        evaluation: seedEvaluation,
        source: "seed",
        updatedComponents: [],
      }),
    );
    lastIterationAccepted = false;
    mergesDue = 0;
    await checkpoint();
  }

  let stopReason: GepaStopReason = "budgetExhausted";

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
      ...(mergeConfig.valOverlapFloor === undefined
        ? {}
        : { valOverlapFloor: mergeConfig.valOverlapFloor }),
    });
    if (proposal === null) {
      return "none";
    }

    const [leftId, rightId] = proposal.parentIds;
    const left = records[leftId] as CandidateRecord<K>;
    const right = records[rightId] as CandidateRecord<K>;

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
    const uniqueIds = unique.map((index) => validationIds[index] as string);
    if (
      !budget.canAfford(
        evaluator.countUncached({
          candidate: proposal.candidate,
          ids: uniqueIds,
          split: "val",
        }),
      )
    ) {
      return "none";
    }

    // Past this point the merge is being tested, so the iteration belongs to
    // it rather than to the mutation branch, and is announced here for the
    // same reason a mutation iteration is: the evaluations that follow have to
    // be attributable to something.
    emit({
      type: "iterationStart",
      iteration,
      parentIds: [...proposal.parentIds],
    });

    // Recorded before scoring: a triplet that was tested and lost must not be
    // proposed again, or the run relitigates the same merge forever.
    mergeAttempts.add(proposal.attemptKey);
    mergeDescriptions.add(proposal.descriptionKey);

    const uniqueEvaluation = await evaluateCached({
      candidate: proposal.candidate,
      batch: unique.map((index) => validationSet[index] as Datum),
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

    // A rejected merge consumes neither `mergesDue` nor `totalMergesTested`,
    // matching the reference — `engine.py` marks the branch "REJECTED: do NOT
    // consume merges_due or total_merges_tested". The cap counts merges that
    // landed, not merges that were priced, so the run keeps looking while any
    // untried triplet remains. It is bounded regardless: `mergeAttempts` is
    // recorded before scoring, so no triplet is tried twice.
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
        evaluator.countUncached({
          candidate: proposal.candidate,
          ids: mergeInstances.map((index) => validationIds[index] as string),
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

    const ancestor = records[proposal.ancestorId] as CandidateRecord<K>;
    const record = addCandidate({
      candidate: proposal.candidate,
      parentIds: [...proposal.parentIds],
      evaluation,
      source: "merge",
      updatedComponents: componentNames(proposal.candidate).filter(
        (name) => proposal.candidate[name] !== ancestor.candidate[name],
      ),
    });
    mergesDue -= 1;
    totalMergesTested += 1;

    emitAccepted(record);
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
  function planProposals(): ProposalPlan<Datum, K>[] {
    const state: SelectionState = {
      scoreMatrix: records.map((record) => record.instanceScores),
      aggregateScores: records.map((record) => record.aggregateScore),
      objectiveScores: records.map((record) => record.objectiveScores),
    };
    const plans: ProposalPlan<Datum, K>[] = [];

    for (let slot = 0; slot < proposalsPerIteration; slot += 1) {
      const parent = records[
        candidateSelector({ state, rng })
      ] as CandidateRecord<K>;
      const batchIndices = batchSampler({
        trainingSet,
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
      assertComponents({
        names: componentsToUpdate,
        candidate: parent.candidate,
        source: "componentSelector",
      });
      // Advanced here, when the component is chosen. The reference chooses
      // after its skip checks, so a parent whose minibatch came back perfect
      // leaves its cursor where it was and offers the same component again;
      // here that iteration still costs the parent its turn. Moving the
      // advance past the skip would put it inside the concurrent phase, where
      // several proposals can share a parent and the order they finish in
      // would decide the cursor — trading a component's turn for a run that no
      // longer reproduces at a fixed seed.
      parent.componentCursor =
        (parent.componentCursor + 1) %
        Math.max(1, componentNames(parent.candidate).length);

      plans.push({
        parent,
        batch: batchIndices.map((index) => trainingSet[index] as Datum),
        batchIds: batchIndices.map((index) => trainingIds[index] as string),
        componentsToUpdate,
        attempt: iteration * proposalsPerIteration + slot,
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
    plan: ProposalPlan<Datum, K>,
  ): Promise<ProposalOutcome<Datum, K>> {
    const { parent, batch, batchIds, componentsToUpdate, attempt } = plan;

    const parentEvaluation = await evaluator.evaluateTraced({
      batch,
      candidate: parent.candidate,
      split: "train",
      phase: "minibatch",
      candidateId: parent.id,
      iteration,
    });
    if (parentEvaluation === null) {
      return { status: "budgetExhausted" };
    }

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

    let proposed: ComponentPatch<K>;
    try {
      proposed = await propose({
        candidate: parent.candidate,
        reflectiveDataset,
        componentsToUpdate,
        rejectedProposals,
        attempt,
        reflect: countedReflect,
        signal,
      });
    } catch (err) {
      if (err instanceof ReflectionBudgetExhausted) {
        return { status: "reflectionExhausted" };
      }
      throw err;
    }

    assertComponents({
      names: componentNames(proposed),
      candidate: parent.candidate,
      source: "proposeNewTexts",
    });

    const child: Candidate<K> = { ...parent.candidate, ...proposed };
    // Only the already-recorded pool is consulted here: it does not change
    // while proposals are in flight, whereas deduplicating siblings against
    // each other would hand the run to whichever reflection returned first.
    // That is settled in plan order when the outcomes are committed.
    const changed =
      componentNames(proposed).length > 0 &&
      !seenCandidates.has(candidateFingerprint(child));

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

    let childEvaluation: ScoredBatch<Output>;
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

    const screened = pairMeasured({
      parent: parentEvaluation,
      child: childEvaluation,
    });

    // Every instance in the minibatch failed on one side or the other, so the
    // batch holds no comparison to make. Screening it anyway would decide the
    // proposal on an outage.
    if (screened.parentScores.length === 0) {
      return { status: "skipped" };
    }

    return {
      status: "screened",
      plan,
      child,
      proposed,
      parentScore: mean(screened.parentScores),
      childScore: mean(screened.childScores),
      improvement: sum(screened.childScores) - sum(screened.parentScores),
      accepted: acceptance(screened),
    };
  }

  /**
   * Turns screened proposals into candidates: rejections first, then the
   * survivors the selection policy keeps, each paying for its own validation
   * sweep. Returns a stop reason when the iteration ran the run out of budget.
   */
  async function commitProposals(
    outcomes: readonly ProposalOutcome<Datum, K>[],
  ): Promise<GepaStopReason | undefined> {
    let stop: GepaStopReason | undefined;
    const improved: ScreenedProposal<Datum, K>[] = [];
    /**
     * Children two siblings converged on. Resolved here rather than while the
     * proposals were in flight: the first one in plan order keeps the child,
     * whichever of them finished first.
     */
    const claimed = new Set<string>();

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
      const fingerprint = candidateFingerprint(outcome.child);
      if (claimed.has(fingerprint)) {
        continue;
      }
      claimed.add(fingerprint);

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
    const scheduled: {
      outcome: ScreenedProposal<Datum, K>;
      candidateId: number;
      instances: number[];
    }[] = [];
    let owed = 0;

    for (const outcome of survivors) {
      const instances = selectValInstances(outcome.child);
      const uncached = evaluator.countUncached({
        candidate: outcome.child,
        ids: instances.map((index) => validationIds[index] as string),
        split: "val",
      });

      // Every sweep is priced against the whole schedule before any of them
      // runs. Discovering the shortfall mid-fan-out instead would either waste
      // a sweep that was paid for, or record a candidate under an id the
      // adapter was never told.
      if (!budget.canAfford(owed + uncached)) {
        stop ??= "budgetExhausted";
        break;
      }
      owed += uncached;
      scheduled.push({
        outcome,
        candidateId: baseId + scheduled.length,
        instances,
      });
    }

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
      // Unreachable while the schedule is priced up front, and left in place
      // because recording a candidate under an id that shifted would be worse
      // than stopping.
      if (evaluation === undefined) {
        stop ??= "budgetExhausted";
        break;
      }
      const record = addCandidate({
        candidate: item.outcome.child,
        parentIds: [item.outcome.plan.parent.id],
        evaluation,
        source: "mutation",
        updatedComponents: componentNames(item.outcome.proposed),
      });
      emitAccepted(record);
    }

    return stop;
  }

  /** The improving proposals an iteration keeps, in the order they were made. */
  function selectSurvivors(
    improved: readonly ScreenedProposal<Datum, K>[],
  ): ScreenedProposal<Datum, K>[] {
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
    if (costExhausted({ usage: evaluator.usage(), maxCostUsd })) {
      stopReason = "costExhausted";
      break;
    }
    if (deadline.exceeded()) {
      stopReason = "deadlineReached";
      break;
    }
    if (iteration >= maxIterations) {
      stopReason = "maxIterations";
      break;
    }
    // What it takes to finish an iteration rather than to start one: every
    // proposal screens its parent and its child on a minibatch, and a child
    // that passes still needs a validation sweep to be promoted. Reserving
    // less means paying for reflection on proposals that can never be
    // screened, and discarding a child that already earned its place.
    if (
      !budget.canAfford(
        proposalsPerIteration * minibatchSize * 2 + validationSet.length,
      )
    ) {
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
    let pendingStop: GepaStopReason | undefined;

    try {
      // The cap is checked here as well as where merges are scheduled: a run
      // that keeps improving builds a backlog of due merges, and the backlog
      // outlives the acceptances that created it.
      const mergeScheduled =
        mergeConfig.enabled &&
        mergesDue > 0 &&
        lastIterationAccepted &&
        totalMergesTested < mergeConfig.maxInvocations;
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

    // Checkpointed before the stop is honoured: an iteration that ran out of
    // budget partway still consumed its plans, and a snapshot that stopped
    // short of it would replay work this run already paid for.
    iteration += 1;
    await checkpoint();

    if (pendingStop !== undefined) {
      stopReason = pendingStop;
      break;
    }
  }

  const bestCandidateId = valEvaluationPolicy.bestCandidate(records);
  const best = records[bestCandidateId] as CandidateRecord<K>;

  // Run after the winner is chosen, never before: an evaluation the selection
  // could read would make the held-out set another validation set.
  const heldOut =
    testSet === undefined
      ? undefined
      : await evaluateCached({
          candidate: best.candidate,
          batch: testSet,
          ids: testIds,
          split: "test",
          phase: "test",
          candidateId: bestCandidateId,
          charge: false,
        });
  const testScore = heldOut === undefined ? undefined : measuredMean(heldOut);

  emit({
    type: "finish",
    reason: stopReason,
    bestCandidateId,
    bestScore: best.aggregateScore,
    metricCalls: budget.spent(),
    ...(testScore === undefined ? {} : { testScore }),
    ...(heldOut === undefined
      ? {}
      : { testInstanceScores: instanceRow(heldOut) }),
    ...(heldOut === undefined || !trackBestOutputs
      ? {}
      : { testOutputs: heldOut.outputs }),
  });

  const perObjectiveBest = collectPerObjectiveBest(records);
  const bestOutputs = outputsByCandidate.get(bestCandidateId);

  return {
    bestCandidate: best.candidate,
    bestScore: best.aggregateScore,
    usage: evaluator.usage(),
    bestCandidateId,
    ...(testScore === undefined
      ? {}
      : {
          testScore,
          testMetricCalls: evaluator.unchargedCalls(),
          testUsage: evaluator.unchargedUsage(),
        }),
    ...(bestOutputs === undefined ? {} : { bestOutputs }),
    candidates: records,
    paretoFrontier: collectDominatorIds(records).map(
      (id) => records[id] as CandidateRecord<K>,
    ),
    ...(perObjectiveBest === undefined ? {} : { perObjectiveBest }),
    scoreMatrix: records.map((record) => [...record.instanceScores]),
    metricCalls: budget.spent(),
    reflectionCalls,
    cacheHits: evaluator.cacheHits(),
    iterations: iteration,
    stopReason,
    snapshot: takeSnapshot(),
  };
}

/**
 * Range checks on the search knobs, run at construction so a configuration
 * that could never terminate is refused before a task is ever handed to it.
 * Task-shaped checks stay in `runGepa`, where the data is.
 */
function assertGepaConfig(config: GepaConfig): void {
  if (
    config.reflection?.buildPrompt !== undefined &&
    config.reflection.strategies !== undefined
  ) {
    throw new Error("reflection takes buildPrompt or strategies, not both");
  }
  if (config.reflection?.strategies?.length === 0) {
    throw new Error("reflection.strategies must not be empty");
  }

  const {
    minibatchSize = DEFAULT_MINIBATCH_SIZE,
    maxIterations = Number.POSITIVE_INFINITY,
    perfectScore = 1,
    rejectedProposalMemory = DEFAULT_REJECTED_PROPOSAL_MEMORY,
    proposals,
  } = config;

  const proposalsPerIteration = proposals?.perIteration ?? 1;
  const proposalConcurrency = proposals?.concurrency ?? 1;

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
  keepCount(proposals?.selection ?? "all");

  // An empty minibatch is vacuously perfect, which skips the iteration body
  // and charges nothing: uncaught, the default iteration ceiling turns that
  // into a run that neither spends nor terminates.
  if (!Number.isInteger(minibatchSize) || minibatchSize < 1) {
    throw new Error(
      `minibatchSize must be a positive integer, received ${minibatchSize}`,
    );
  }
  if (!Number.isFinite(perfectScore)) {
    throw new Error(
      `perfectScore must be a finite number, received ${perfectScore}`,
    );
  }
  if (!Number.isInteger(rejectedProposalMemory) || rejectedProposalMemory < 0) {
    throw new Error(
      `rejectedProposalMemory must be a non-negative integer, received ${rejectedProposalMemory}`,
    );
  }
  if (
    maxIterations !== Number.POSITIVE_INFINITY &&
    (!Number.isInteger(maxIterations) || maxIterations < 0)
  ) {
    throw new Error(
      `maxIterations must be a non-negative integer or Infinity, received ${maxIterations}`,
    );
  }
}

/**
 * The single narrowing point for a snapshot's candidate pool. A snapshot is
 * JSON that left the process and came back with plain string keys, so every
 * record is checked against the seed's components before it is read as one of
 * them — behind the fingerprint check, which has already established that the
 * snapshot belongs to this run.
 */
function restoreRecords<K extends string>(args: {
  records: readonly CandidateRecord[];
  seedCandidate: Candidate<K>;
}): CandidateRecord<K>[] {
  const { records, seedCandidate } = args;

  const known = new Set<string>(componentNames(seedCandidate));
  for (const record of records) {
    const named = [
      ...Object.keys(record.candidate),
      ...record.updatedComponents,
    ];
    for (const name of named) {
      if (!known.has(name)) {
        throw new Error(
          `checkpoint names the component "${name}", which the seed candidate does not have (${[...known].join(", ")})`,
        );
      }
    }
  }

  return copyRecords(records) as CandidateRecord<K>[];
}

/**
 * Rejections arrive from a snapshot keyed by plain strings. Reading them
 * through the seed's own component names narrows them without an assertion,
 * and drops anything the seed no longer has.
 */
function restoreRejections<K extends string>(args: {
  rejections: Readonly<Record<string, RejectedProposal[]>>;
  components: readonly K[];
}): Partial<Record<K, RejectedProposal[]>> {
  const { rejections, components } = args;

  const restored: Partial<Record<K, RejectedProposal[]>> = {};
  for (const component of components) {
    const history = rejections[component];
    if (history !== undefined) {
      restored[component] = history.map((entry) => ({ ...entry }));
    }
  }
  return restored;
}

/** The inverse: back to the plain string keys a snapshot is written with. */
function snapshotRejections<K extends string>(args: {
  rejections: Readonly<Partial<Record<K, RejectedProposal[]>>>;
  components: readonly K[];
}): Record<string, RejectedProposal[]> {
  const { rejections, components } = args;

  const copy: Record<string, RejectedProposal[]> = {};
  for (const component of components) {
    const history = rejections[component];
    if (history !== undefined) {
      copy[component] = history.map((entry) => ({ ...entry }));
    }
  }
  return copy;
}

/**
 * Patches are merged over the parent, so a name the candidate does not have is
 * added rather than refused: the run would go on optimizing text the system
 * under optimization never reads, and every descendant would carry it.
 */
function assertComponents(args: {
  names: readonly string[];
  candidate: Candidate;
  source: string;
}): void {
  const { names, candidate, source } = args;

  for (const name of names) {
    if (!Object.hasOwn(candidate, name)) {
      throw new Error(
        `${source} named "${name}", which is not a component of the candidate (${Object.keys(candidate).join(", ")})`,
      );
    }
  }
}

/** Copies everything a run mutates in place, so a snapshot never aliases one. */
function copyRecords<K extends string>(
  records: readonly CandidateRecord<K>[],
): CandidateRecord<K>[] {
  return records.map((record) => ({
    ...record,
    parentIds: [...record.parentIds],
    instanceScores: [...record.instanceScores],
    updatedComponents: [...record.updatedComponents],
    ...(record.objectiveScores === undefined
      ? {}
      : { objectiveScores: { ...record.objectiveScores } }),
  }));
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

/**
 * Mean of each objective over the instances this candidate was scored on.
 *
 * An objective only some of those instances reported is left out rather than
 * averaged over the ones that did: candidates are compared objective by
 * objective on the frontier, and a mean over one instance is not the same
 * measurement as a mean over forty.
 */
function meanObjectives(args: {
  rows: readonly (Record<string, number> | undefined)[];
  scores: readonly (number | undefined)[];
}): Record<string, number> | undefined {
  const { rows, scores } = args;

  const measured = rows.filter((_, index) => scores[index] !== undefined);
  const totals = new Map<string, { total: number; count: number }>();

  for (const row of measured) {
    for (const [objective, value] of Object.entries(row ?? {})) {
      const running = totals.get(objective) ?? { total: 0, count: 0 };
      running.total += value;
      running.count += 1;
      totals.set(objective, running);
    }
  }

  const complete = [...totals].filter(
    ([, { count }]) => count === measured.length,
  );
  if (complete.length === 0) {
    return undefined;
  }
  return Object.fromEntries(
    complete.map(([objective, { total, count }]) => [objective, total / count]),
  );
}

function collectPerObjectiveBest(
  records: readonly CandidateRecord[],
): GepaResult["perObjectiveBest"] {
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
 * The two rollout sets restricted to the instances both of them measured.
 *
 * Screening is a paired comparison over one minibatch: a transient row is a
 * rollout that never happened, and leaving it in scores the candidate that ran
 * against the infrastructure failure of the one that did not.
 */
function pairMeasured(args: {
  parent: { scores: readonly number[]; transient?: readonly boolean[] };
  child: { scores: readonly number[]; transient?: readonly boolean[] };
}): { parentScores: number[]; childScores: number[] } {
  const { parent, child } = args;

  const parentScores: number[] = [];
  const childScores: number[] = [];

  for (let index = 0; index < parent.scores.length; index += 1) {
    if (
      parent.transient?.[index] === true ||
      child.transient?.[index] === true
    ) {
      continue;
    }
    parentScores.push(parent.scores[index] as number);
    childScores.push(child.scores[index] as number);
  }
  return { parentScores, childScores };
}
