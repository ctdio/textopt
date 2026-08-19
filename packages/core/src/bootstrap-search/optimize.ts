import { createBudget } from "../budget.js";
import { createMemoryCache } from "../cache.js";
import type { CachedScore, EvaluationCache } from "../cache.js";
import { assertResumable, runFingerprint } from "../checkpoint.js";
import { createDeadline } from "../deadline.js";
import { bootstrapDemos, formatDemos } from "../demos.js";
import type { Demo, DemoRenderer } from "../demos.js";
import {
  BudgetExhausted,
  costExhausted,
  createEvaluator,
  measuredMean,
  requireMeasuredMean,
} from "../evaluation.js";
import type { EvaluationEvent } from "../evaluation.js";
import type {
  Optimizer,
  OptimizerResult,
  OptimizerTask,
} from "../optimizer.js";
import { createSeededRng } from "../rng.js";
import { componentNames } from "../types.js";
import type { Adapter, Candidate } from "../types.js";

/**
 * Where a candidate's demo block came from. `zeroShot` holds no demos at all,
 * `labeled` is built from gold outputs and costs no rollouts, `unshuffled`
 * takes the training set in order, and `bootstrapped` is one shuffled harvest.
 */
export type DemoSource = "zeroShot" | "labeled" | "unshuffled" | "bootstrapped";

export interface BootstrapCandidate<K extends string = string> {
  candidate: Candidate<K>;
  source: DemoSource;
  /** Demos in the block, so a win can be read against how much it carried. */
  demos: number;
  score: number;
}

export interface BootstrapSearchConfig {
  /**
   * Shuffled harvests attempted, beyond the fixed candidates every run tries.
   * Default 16, DSPy's `num_candidate_programs`.
   */
  candidates?: number;
  /** Most demos a harvested set may hold. Default 4. */
  maxDemos?: number;
  /** Fewest demos a shuffled harvest may ask for. Default 1. */
  minDemos?: number;
  /** Most demos the labels-only candidate may hold. Default 16. */
  maxLabeledDemos?: number;
  /** Score a rollout must reach to be kept as a demo. */
  demoMinScore?: number;
  /**
   * Stop as soon as a candidate reaches this validation score. Unset evaluates
   * every candidate, which is the reliable reading and the expensive one.
   */
  stopAtScore?: number;
  seed?: number;
  trackBestOutputs?: boolean;
  /**
   * Include cached instance scores in every checkpoint. Leaving them out keeps
   * snapshots small at the cost of a resumed run re-paying for rollouts it
   * cannot look up. Default true.
   */
  checkpointCache?: boolean;
}

/**
 * Everything needed to continue a run: which candidates have been evaluated,
 * what they scored, and the budget already spent. Plain JSON — persist it and
 * hand it back as `resumeFrom`.
 */
export interface BootstrapSearchSnapshot {
  version: 1;
  fingerprint: string;
  candidates: BootstrapCandidate[];
  best: Candidate;
  bestScore: number;
  seedScore: number;
  /** How many candidates have been drawn, including the fixed ones. */
  drawn: number;
  metricCalls: number;
  bootstrapMetricCalls: number;
  cacheHits: number;
  rngState: number;
  cache?: [string, CachedScore][];
}

export interface BootstrapSearchTask<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> extends OptimizerTask<Datum, Trajectory, Output, K> {
  /**
   * The base adapter, and no reflection model: this search writes no text at
   * all, so there is nothing for a model to propose.
   */
  adapter: Adapter<Datum, Trajectory, Output, NoInfer<K>>;
  /**
   * Components holding few-shot demo blocks. Every one of them receives the
   * same harvest, because a harvest is a pass over the training set and one
   * pass per component would multiply the cost of the search by the number of
   * components for evidence that came from the same rollouts.
   */
  demoComponents: readonly NoInfer<K>[];
  /** Renders a harvested rollout as demo text. Defaults to JSON. */
  renderDemo?: DemoRenderer<NoInfer<Datum>, NoInfer<Output>>;
  /**
   * The gold output for a training datum, where the caller has labels. Supply
   * it and the run tries a labels-only candidate, which costs no rollouts to
   * build and is the only candidate available at all to a system too weak to
   * bootstrap from.
   */
  goldOutput?: (datum: NoInfer<Datum>) => NoInfer<Output> | undefined;
  instanceId?: (args: { datum: NoInfer<Datum>; index: number }) => string;
  /** Pass `false` to disable caching entirely. */
  cache?: EvaluationCache | false;
  onEvent?: (event: BootstrapSearchEvent<NoInfer<K>>) => void;
  /** Called with a resumable snapshot after every candidate is scored. */
  onCheckpoint?: (snapshot: BootstrapSearchSnapshot) => void | Promise<void>;
  /** Snapshot to continue from. */
  resumeFrom?: BootstrapSearchSnapshot;
}

export type BootstrapSearchStopReason =
  | "budgetExhausted"
  | "costExhausted"
  | "deadlineReached"
  | "scoreReached"
  | "candidatesExhausted"
  | "aborted";

export type BootstrapSearchEvent<K extends string = string> =
  | { type: "start"; components: K[]; validationSetSize: number }
  | ({ type: "evaluation" } & EvaluationEvent)
  | {
      type: "candidate";
      index: number;
      source: DemoSource;
      demos: number;
      score: number;
      accepted: boolean;
    }
  | {
      type: "finish";
      reason: BootstrapSearchStopReason;
      bestScore: number;
      metricCalls: number;
      testScore?: number;
    };

export interface BootstrapSearchResult<
  K extends string = string,
  Output = unknown,
> extends OptimizerResult<K, BootstrapSearchStopReason, Output> {
  /** The seed's score, so the lift the demos bought is readable directly. */
  seedScore: number;
  /** Every candidate evaluated, in the order it was tried. */
  candidates: BootstrapCandidate<K>[];
  /** Rollouts spent harvesting demos, included in `metricCalls`. */
  bootstrapMetricCalls: number;
  cacheHits: number;
  snapshot: BootstrapSearchSnapshot;
}

const DEFAULT_CANDIDATES = 16;
const DEFAULT_MAX_DEMOS = 4;
const DEFAULT_MIN_DEMOS = 1;
const DEFAULT_MAX_LABELED_DEMOS = 16;

/**
 * Bootstrapped few-shot search: harvest demonstrations from rollouts the metric
 * already rewarded, and pick the set that scores best.
 *
 * DSPy's `BootstrapFewShotWithRandomSearch`, which is what the literature
 * usually means by "random search" over prompts. It is the only optimizer here
 * that calls no model to write text: every candidate is assembled from outputs
 * the system itself produced, so the search costs rollouts and nothing else.
 * That makes it the right first thing to try — it is cheap, it needs no
 * frontier model, and on tasks where the instruction is already adequate and
 * the format is not, it is often the whole win.
 *
 * The fixed candidates come first and in DSPy's order: zero-shot (seed -3),
 * labels-only when gold outputs exist (seed -2), and one unshuffled harvest at
 * full size (seed -1). Shuffled harvests of random size follow. Keeping
 * zero-shot in the running is not a formality — demonstrations can hurt, and a
 * search that cannot return "no demos" has no baseline to report against.
 *
 * One deviation, deliberate: DSPy bootstraps each predictor separately from the
 * traces of one pass. This library's adapter runs the whole system, so a
 * harvest is a set of end-to-end rollouts and every demo component is given the
 * same block. For per-module demos, use `createPipelineAdapter` with GEPA,
 * which sees each module's own inputs and outputs.
 */
export class BootstrapSearchOptimizer implements Optimizer<BootstrapSearchStopReason> {
  readonly #config: BootstrapSearchConfig;

  constructor(config: BootstrapSearchConfig = {}) {
    this.#config = config;
  }

  optimize<
    Datum,
    Trajectory = unknown,
    Output = unknown,
    const K extends string = string,
  >(
    task: BootstrapSearchTask<Datum, Trajectory, Output, K>,
  ): Promise<BootstrapSearchResult<K, Output>> {
    return run({ config: this.#config, task });
  }
}

async function run<Datum, Trajectory, Output, K extends string>(args: {
  config: BootstrapSearchConfig;
  task: BootstrapSearchTask<Datum, Trajectory, Output, K>;
}): Promise<BootstrapSearchResult<K, Output>> {
  const { config, task } = args;

  const {
    candidates: shuffledHarvests = DEFAULT_CANDIDATES,
    maxDemos = DEFAULT_MAX_DEMOS,
    minDemos = DEFAULT_MIN_DEMOS,
    maxLabeledDemos = DEFAULT_MAX_LABELED_DEMOS,
    demoMinScore,
    stopAtScore,
    seed = 0,
    trackBestOutputs = false,
    checkpointCache = true,
  } = config;

  const {
    seedCandidate,
    trainingSet,
    validationSet = trainingSet,
    testSet,
    adapter,
    demoComponents,
    renderDemo,
    goldOutput,
    maxMetricCalls,
    cache,
    cacheNamespace,
    retry,
    maxCostUsd,
    maxWallClockMs,
    instanceId = defaultInstanceId,
    onEvent,
    onCheckpoint,
    resumeFrom,
    signal,
  } = task;

  const deadline = createDeadline({ maxWallClockMs });
  const components = componentNames(seedCandidate);

  if (trainingSet.length === 0) {
    throw new Error("optimize requires a non-empty trainingSet");
  }
  if (validationSet.length === 0) {
    throw new Error("optimize requires a non-empty validationSet");
  }
  if (demoComponents.length === 0) {
    throw new Error(
      "optimize requires at least one demoComponent: this search has nothing to put demonstrations in otherwise",
    );
  }
  if (testSet !== undefined && testSet.length === 0) {
    throw new Error(
      "optimize requires a non-empty testSet when one is given; omit it to skip held-out evaluation",
    );
  }

  const validationIds = validationSet.map((datum, index) =>
    instanceId({ datum, index }),
  );
  const testIds =
    testSet?.map((datum, index) => instanceId({ datum, index })) ?? [];

  const fingerprint = runFingerprint({
    seedCandidate,
    trainingIds: trainingSet.map((datum, index) =>
      instanceId({ datum, index }),
    ),
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
  const evaluationCache =
    cache === false ? undefined : (cache ?? createMemoryCache());
  const evaluator = createEvaluator<Datum, Trajectory, Output, K>({
    adapter,
    budget,
    ...(retry === undefined ? {} : { retry }),
    ...(cacheNamespace === undefined ? {} : { cacheNamespace }),
    ...(evaluationCache === undefined ? {} : { cache: evaluationCache }),
    trackOutputs: trackBestOutputs,
    cacheHits: resumeFrom?.cacheHits ?? 0,
    ...(signal === undefined ? {} : { signal }),
    onEvaluation: (event) => onEvent?.({ type: "evaluation", ...event }),
  });

  evaluator.restore(resumeFrom?.cache ?? []);

  // Copied, never aliased: a snapshot is the caller's to keep, and a resumed
  // run that pushed into it would leave them holding a checkpoint of a run
  // that had already moved past it.
  const evaluated: BootstrapCandidate<K>[] = [
    ...((resumeFrom?.candidates ?? []) as BootstrapCandidate<K>[]),
  ];
  let drawn = resumeFrom?.drawn ?? 0;
  let bootstrapMetricCalls = resumeFrom?.bootstrapMetricCalls ?? 0;
  let stopReason: BootstrapSearchStopReason = "candidatesExhausted";

  onEvent?.({
    type: "start",
    components,
    validationSetSize: validationSet.length,
  });

  async function sweep(candidate: Candidate<K>, phase: "seed" | "validation") {
    return evaluator.evaluate({
      candidate,
      batch: validationSet,
      ids: validationIds,
      split: "val",
      phase,
      candidateId: null,
      iteration: evaluated.length,
    });
  }

  // The seed is the zero-shot candidate's own score when the demo components
  // start empty, which they usually do; scoring it separately would buy the
  // same number twice.
  const seedScore =
    resumeFrom?.seedScore ??
    requireMeasuredMean({
      batch: await sweep(seedCandidate, "seed"),
      phase: "seed",
    });

  let best = (resumeFrom?.best as Candidate<K> | undefined) ?? seedCandidate;
  let bestScore = resumeFrom?.bestScore ?? seedScore;
  let bestOutputs: (Output | undefined)[] | undefined;

  function takeSnapshot(): BootstrapSearchSnapshot {
    const cached = checkpointCache ? evaluationCache?.entries?.() : undefined;

    return {
      version: 1,
      fingerprint,
      candidates: [...evaluated],
      best,
      bestScore,
      seedScore,
      drawn,
      metricCalls: budget.spent(),
      bootstrapMetricCalls,
      cacheHits: evaluator.cacheHits(),
      rngState: rng.state(),
      ...(cached === undefined ? {} : { cache: cached }),
    };
  }

  async function checkpoint(): Promise<void> {
    if (onCheckpoint === undefined) {
      return;
    }
    await onCheckpoint(takeSnapshot());
  }

  const plan = candidatePlan({
    shuffledHarvests,
    labeled: goldOutput !== undefined,
  });

  for (; drawn < plan.length; drawn += 1) {
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
    // A harvest and its sweep are one purchase: harvesting demos the run
    // cannot afford to score buys nothing at all.
    if (!budget.canAfford(validationSet.length + 1)) {
      stopReason = "budgetExhausted";
      break;
    }

    const source = plan[drawn] as DemoSource;
    const block = await buildBlock(source);
    const candidate = withDemos(block);

    let evaluation: Awaited<ReturnType<typeof sweep>>;
    try {
      evaluation = await sweep(candidate, "validation");
    } catch (err) {
      if (err instanceof BudgetExhausted) {
        stopReason = "budgetExhausted";
        break;
      }
      if (signal?.aborted) {
        stopReason = "aborted";
        break;
      }
      throw err;
    }

    const score = measuredMean(evaluation);
    // The sweep measured the provider rather than the demos, so there is
    // nothing to compare the incumbent against.
    if (score === undefined) {
      continue;
    }

    const accepted = score > bestScore;
    evaluated.push({
      candidate,
      source,
      demos: countDemos(block),
      score,
    });
    onEvent?.({
      type: "candidate",
      index: evaluated.length - 1,
      source,
      demos: countDemos(block),
      score,
      accepted,
    });

    if (accepted) {
      best = candidate;
      bestScore = score;
      bestOutputs = evaluation.outputs;
    }

    await checkpoint();

    if (stopAtScore !== undefined && score >= stopAtScore) {
      stopReason = "scoreReached";
      drawn += 1;
      break;
    }
  }

  if (signal?.aborted) {
    stopReason = "aborted";
  }

  const testScore =
    testSet === undefined
      ? undefined
      : measuredMean(
          await evaluator.evaluate({
            candidate: best,
            batch: testSet,
            ids: testIds,
            split: "test",
            phase: "test",
            candidateId: null,
            iteration: evaluated.length,
            charge: false,
          }),
        );

  onEvent?.({
    type: "finish",
    reason: stopReason,
    bestScore,
    metricCalls: budget.spent(),
    ...(testScore === undefined ? {} : { testScore }),
  });

  return {
    bestCandidate: best,
    bestScore,
    usage: evaluator.usage(),
    seedScore,
    candidates: evaluated,
    bootstrapMetricCalls,
    cacheHits: evaluator.cacheHits(),
    metricCalls: budget.spent(),
    snapshot: takeSnapshot(),
    ...(trackBestOutputs && bestOutputs !== undefined ? { bestOutputs } : {}),
    ...(testScore === undefined
      ? {}
      : { testScore, testMetricCalls: testSet?.length ?? 0 }),
    stopReason,
  };

  function withDemos(block: string): Candidate<K> {
    const candidate = { ...seedCandidate };
    for (const name of demoComponents) {
      candidate[name] = block;
    }
    return candidate;
  }

  async function buildBlock(source: DemoSource): Promise<string> {
    if (source === "zeroShot") {
      return "";
    }
    if (source === "labeled") {
      return labeledBlock();
    }

    // Sizes vary across the shuffled harvests because more demos is not
    // monotonically better: a long block crowds out the instruction, and which
    // length wins is exactly what this search settles.
    const requested =
      source === "unshuffled"
        ? maxDemos
        : minDemos + rng.nextInt(Math.max(1, maxDemos - minDemos + 1));
    const affordable = Math.min(
      trainingSet.length,
      budget.remaining() - validationSet.length,
    );
    if (affordable < 1) {
      return "";
    }

    const harvest = await bootstrapDemos<Datum, Trajectory, Output, K>({
      adapter,
      candidate: seedCandidate,
      trainingSet,
      ...(demoMinScore === undefined ? {} : { minScore: demoMinScore }),
      maxDemos: requested,
      maxMetricCalls: affordable,
      ...(source === "unshuffled" ? {} : { rng }),
      ...(renderDemo === undefined ? {} : { renderDemo }),
      ...(signal === undefined ? {} : { signal }),
    });

    bootstrapMetricCalls += harvest.metricCalls;
    budget.reserve(harvest.metricCalls);

    return harvest.block;
  }

  function labeledBlock(): string {
    const labelled = trainingSet
      .map((datum) => ({ input: datum, output: goldOutput?.(datum) }))
      .filter((demo) => demo.output !== undefined)
      .slice(0, maxLabeledDemos) as Demo<Datum, Output>[];

    return formatDemos(
      labelled,
      renderDemo === undefined ? {} : { render: renderDemo },
    );
  }
}

/**
 * The order candidates are tried in, following DSPy's special seeds: zero-shot
 * first, then labels-only, then one unshuffled full-size harvest, then the
 * shuffled ones. Cheapest and most reliable first, so a run cut short by its
 * budget still has the baseline it needs to report against.
 */
function candidatePlan(args: {
  shuffledHarvests: number;
  labeled: boolean;
}): DemoSource[] {
  const { shuffledHarvests, labeled } = args;

  return [
    "zeroShot" as const,
    ...(labeled ? (["labeled"] as const) : []),
    "unshuffled" as const,
    ...Array.from({ length: shuffledHarvests }, () => "bootstrapped" as const),
  ];
}

function countDemos(block: string): number {
  return block.split("<demo>").length - 1;
}

function defaultInstanceId(args: { datum: unknown; index: number }): string {
  return String(args.index);
}
