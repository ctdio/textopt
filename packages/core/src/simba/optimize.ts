import { createBudget } from "../budget.js";
import { candidateHash, createMemoryCache } from "../cache.js";
import type { CachedScore, EvaluationCache } from "../cache.js";
import { assertResumable, runFingerprint } from "../checkpoint.js";
import { mapWithConcurrency } from "../concurrency.js";
import { createDeadline } from "../deadline.js";
import { formatDemos, parseDemos } from "../demos.js";
import type { Demo, DemoRenderer } from "../demos.js";
import {
  BudgetExhausted,
  costExhausted,
  createEvaluator,
  measuredMean,
} from "../evaluation.js";
import type { EvaluationEvent } from "../evaluation.js";
import type {
  Optimizer,
  OptimizerResult,
  OptimizerTask,
} from "../optimizer.js";
import { createSeededRng } from "../rng.js";
import { createEpochShuffledSampler } from "../sampling.js";
import type { BatchSampler } from "../sampling.js";
import { componentNames } from "../types.js";
import type { Adapter, Candidate, TextModel } from "../types.js";
import { buildAdvicePrompt, parseAdvice } from "./advice.js";
import type { AdvicePromptBuilder } from "./advice.js";
import {
  buildBuckets,
  evenlySpacedIndices,
  percentile,
  samplePoisson,
  softmaxWeights,
  topKPlusBaseline,
} from "./strategies.js";
import type { SimbaBucket } from "./strategies.js";

/**
 * How a candidate is mutated. `appendDemo` adds a rollout the metric already
 * rewarded; `appendRule` asks a model to say what the winning run did
 * differently and writes that into the instruction.
 */
export type SimbaStrategy = "appendDemo" | "appendRule";

export interface SimbaFinalist<K extends string = string> {
  candidate: Candidate<K>;
  /** Mean over the full validation set, which is what selection is decided on. */
  score: number;
  /** Which step produced it; 0 is the seed. */
  step: number;
}

export interface SimbaConfig {
  /** Instances per step. Default 32. */
  minibatchSize?: number;
  /** Programs sampled per step, and candidates built from them. Default 6. */
  candidates?: number;
  /**
   * How many evaluations may be in flight at once. Default 1.
   *
   * Covers the two places a step's work is independent — scoring the
   * candidates a step built, and sweeping the finalists at the end of the run
   * — and nothing else. The trajectory samples and the mutations that read
   * them are deliberately left in sequence: each of those reads state the one
   * before it wrote, so overlapping them would make a seeded run depend on
   * which call returned first.
   */
  concurrency?: number;
  /** Steps to run. Default 8. */
  maxSteps?: number;
  /** Demos a candidate may hold before the loop starts dropping them. Default 4. */
  maxDemos?: number;
  /** Sharpness of the pick between programs when sampling trajectories. */
  samplingTemperature?: number;
  /** Sharpness of the pick between programs when choosing what to mutate. */
  candidateTemperature?: number;
  /** Mutations to draw from. Defaults to both, or to rules alone with no demo component. */
  strategies?: readonly SimbaStrategy[];
  /**
   * Advice calls the run may make. Unset lets the rollout budget bound them,
   * which it does only loosely: a step spends a bounded number of rollouts and
   * up to one advice call per instance in its minibatch, and the advice model
   * is usually the expensive one. Once spent, `appendDemo` carries the run on
   * alone, or the run stops if it was the only mutation enabled.
   */
  maxReflectionCalls?: number;
  seed?: number;
  trackBestOutputs?: boolean;
  checkpointCache?: boolean;
}

/**
 * Everything needed to continue a run: the program pool with its observed
 * scores, the step winners, and the budget already spent. Plain JSON — persist
 * it and hand it back as `resumeFrom`.
 */
export interface SimbaSnapshot {
  version: 1;
  fingerprint: string;
  programs: Candidate[];
  /** Every minibatch mean observed for each program, in the order observed. */
  programScores: number[][];
  winners: { candidate: Candidate; step: number }[];
  step: number;
  metricCalls: number;
  reflectionCalls: number;
  cacheHits: number;
  rngState: number;
  sampler?: unknown;
  cache?: [string, CachedScore][];
}

export interface SimbaTask<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> extends OptimizerTask<Datum, Trajectory, Output, K> {
  /**
   * The base adapter. SIMBA reads outputs, scores and feedback and builds its
   * own evidence, so unlike GEPA it needs no `makeReflectiveDataset`.
   */
  adapter: Adapter<Datum, Trajectory, Output, NoInfer<K>>;
  /** Writes the advice `appendRule` appends. Unused when only demos are appended. */
  reflect?: TextModel;
  /** Components holding few-shot demo blocks. Empty disables `appendDemo`. */
  demoComponents?: readonly NoInfer<K>[];
  /** Components advice is appended to. Defaults to every non-demo component. */
  instructionComponents?: readonly NoInfer<K>[];
  renderDemo?: DemoRenderer<NoInfer<Datum>, NoInfer<Output>>;
  buildAdvicePrompt?: AdvicePromptBuilder<NoInfer<Datum>, NoInfer<Output>>;
  sampler?: BatchSampler<NoInfer<Datum>>;
  instanceId?: (args: { datum: NoInfer<Datum>; index: number }) => string;
  cache?: EvaluationCache | false;
  onEvent?: (event: SimbaEvent<NoInfer<K>>) => void;
  onCheckpoint?: (snapshot: SimbaSnapshot) => void | Promise<void>;
  resumeFrom?: SimbaSnapshot;
}

export type SimbaStopReason =
  | "budgetExhausted"
  | "costExhausted"
  | "deadlineReached"
  | "reflectionBudgetExhausted"
  | "maxSteps"
  | "aborted";

export type SimbaEvent<K extends string = string> =
  | { type: "start"; components: K[]; validationSetSize: number }
  | ({ type: "evaluation" } & EvaluationEvent)
  | { type: "stepStart"; step: number; poolSize: number }
  | {
      type: "candidate";
      step: number;
      strategy: SimbaStrategy;
      /** The program it was mutated from; 0 is the seed. */
      sourceProgram: number;
      minibatchScore: number;
    }
  | { type: "error"; step: number; err: unknown }
  | {
      type: "finish";
      reason: SimbaStopReason;
      bestScore: number;
      metricCalls: number;
      testScore?: number;
    };

export interface SimbaResult<
  K extends string = string,
  Output = unknown,
> extends OptimizerResult<K, SimbaStopReason, Output> {
  /** The seed's own validation score, so the lift the run bought is readable. */
  seedScore: number;
  steps: number;
  /** The step winners scored on the full validation set, best first. */
  finalists: SimbaFinalist<K>[];
  reflectionCalls: number;
  cacheHits: number;
  snapshot: SimbaSnapshot;
}

/**
 * What one evaluation measured, or that the run stopped before it could.
 * Carried rather than thrown so the step, or the finalist ranking, still
 * commits everything ahead of it exactly as the serial version did.
 */
type ScoreOutcome =
  | { score: number | undefined; stop?: undefined }
  | { score?: undefined; stop: true };

const DEFAULT_MINIBATCH_SIZE = 32;
const DEFAULT_CANDIDATES = 6;
const DEFAULT_MAX_STEPS = 8;
const DEFAULT_MAX_DEMOS = 4;
const DEFAULT_TEMPERATURE = 0.2;
/** What `maxDemos: 0` stands in as, so the drop rate stays defined. */
const DEMO_DROP_SCALE = 3;

/**
 * SIMBA — stochastic introspective mini-batch ascent.
 *
 * Run several programs over the same minibatch, find the instances they
 * disagree about most, and mutate toward whatever the winning run did. The
 * disagreement is the point: an instance one program solved and another failed
 * is a controlled experiment with the input held fixed, so the difference in
 * reward is attributable to behaviour rather than to difficulty. GEPA reflects
 * on failures; SIMBA reflects on the *contrast* between a success and a failure
 * of the same input, which is a strictly stronger signal when it exists — and
 * costs a pool of programs to produce.
 *
 * Two mutations, drawn at random per instance: append a demonstration the
 * metric already rewarded, or ask a model what the better run did differently
 * and append that as a rule. Neither replaces text, so a candidate accumulates;
 * demos are dropped at a Poisson rate to keep the block from crowding out
 * everything else.
 *
 * Ported from DSPy's SIMBA, with two deliberate changes. First, a trajectory
 * sample runs one program across the whole minibatch rather than resampling a
 * program per instance: the adapter here owns decoding, so there is no
 * temperature knob to vary, and the variability comes from the program pool
 * instead. Second, the percentile guards are strict rather than inclusive — on
 * a step where every rollout ties, an inclusive guard blocks every mutation and
 * the run does nothing at all, which is the one case where the guard's own
 * premise does not hold.
 */
export class SimbaOptimizer implements Optimizer<SimbaStopReason> {
  readonly #config: SimbaConfig;

  constructor(config: SimbaConfig = {}) {
    assertSimbaConfig(config);
    this.#config = config;
  }

  optimize<
    Datum,
    Trajectory = unknown,
    Output = unknown,
    const K extends string = string,
  >(
    task: SimbaTask<Datum, Trajectory, Output, K>,
  ): Promise<SimbaResult<K, Output>> {
    return run({ config: this.#config, task });
  }
}

async function run<Datum, Trajectory, Output, K extends string>(args: {
  config: SimbaConfig;
  task: SimbaTask<Datum, Trajectory, Output, K>;
}): Promise<SimbaResult<K, Output>> {
  const { config, task } = args;

  const {
    minibatchSize = DEFAULT_MINIBATCH_SIZE,
    candidates: candidateCount = DEFAULT_CANDIDATES,
    concurrency = 1,
    maxSteps = DEFAULT_MAX_STEPS,
    maxDemos = DEFAULT_MAX_DEMOS,
    samplingTemperature = DEFAULT_TEMPERATURE,
    candidateTemperature = DEFAULT_TEMPERATURE,
    strategies,
    maxReflectionCalls,
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
    reflect,
    demoComponents = [],
    instructionComponents,
    renderDemo,
    buildAdvicePrompt: buildPrompt = buildAdvicePrompt,
    sampler = createEpochShuffledSampler<Datum>({ minibatchSize }),
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

  const components = componentNames(seedCandidate);
  const ruleComponents =
    instructionComponents ??
    components.filter((name) => !demoComponents.includes(name));
  const enabled =
    strategies ??
    (demoComponents.length > 0
      ? (["appendDemo", "appendRule"] as const)
      : (["appendRule"] as const));

  if (trainingSet.length === 0) {
    throw new Error("optimize requires a non-empty trainingSet");
  }
  if (validationSet.length === 0) {
    throw new Error("optimize requires a non-empty validationSet");
  }
  if (minibatchSize > trainingSet.length) {
    throw new Error(
      `optimize requires a minibatchSize no larger than the trainingSet: ${minibatchSize} > ${trainingSet.length}`,
    );
  }
  if (enabled.length === 0) {
    throw new Error("optimize requires at least one strategy");
  }
  if (enabled.includes("appendDemo") && demoComponents.length === 0) {
    throw new Error(
      "the appendDemo strategy requires at least one entry in demoComponents",
    );
  }
  if (enabled.includes("appendRule") && ruleComponents.length === 0) {
    throw new Error(
      "the appendRule strategy requires at least one instruction component to write into",
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

  const deadline = createDeadline({ maxWallClockMs });
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
    trackOutputs: true,
    cacheHits: resumeFrom?.cacheHits ?? 0,
    ...(signal === undefined ? {} : { signal }),
    onEvaluation: (event) => onEvent?.({ type: "evaluation", ...event }),
  });

  evaluator.restore(resumeFrom?.cache ?? []);
  if (resumeFrom?.sampler !== undefined) {
    sampler.restore?.(resumeFrom.sampler);
  }

  // Copied, never aliased: a snapshot is the caller's to keep, and a resumed
  // run that pushed into it would leave them holding a checkpoint of a run
  // that had already moved past it.
  const programs: Candidate<K>[] = [
    ...((resumeFrom?.programs as Candidate<K>[] | undefined) ?? [
      seedCandidate,
    ]),
  ];
  const programScores: number[][] = (resumeFrom?.programScores ?? [[]]).map(
    (scores) => [...scores],
  );
  const winners: { candidate: Candidate<K>; step: number }[] = [
    ...((resumeFrom?.winners as
      { candidate: Candidate<K>; step: number }[] | undefined) ?? [
      { candidate: seedCandidate, step: 0 },
    ]),
  ];

  let step = resumeFrom?.step ?? 0;
  let reflectionCalls = resumeFrom?.reflectionCalls ?? 0;
  let stopReason: SimbaStopReason = "maxSteps";

  onEvent?.({
    type: "start",
    components,
    validationSetSize: validationSet.length,
  });

  // Held back so the run can always afford to score its finalists against each
  // other. Selection on minibatch means alone would pick whichever candidate
  // drew the easiest instances, which is the failure mode the final sweep exists
  // to prevent — so the sweep is reserved before the search, not after it.
  const finalistReserve =
    Math.min(candidateCount + 1, maxSteps + 1) * validationSet.length;
  const stepCost = candidateCount * minibatchSize + minibatchSize;

  for (; step < maxSteps; step += 1) {
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
    if (budget.remaining() - finalistReserve < stepCost) {
      stopReason = "budgetExhausted";
      break;
    }
    if (available().length === 0) {
      stopReason = "reflectionBudgetExhausted";
      break;
    }

    const pool = topKPlusBaseline({
      scores: programs.map((_, index) => averageScore(index)),
      k: candidateCount,
    });
    onEvent?.({ type: "stepStart", step, poolSize: pool.length });

    const batch = sampler({ trainingSet, iteration: step, rng }).map(
      (index) => trainingSet[index] as Datum,
    );

    const samples = [];
    for (let slot = 0; slot < candidateCount; slot += 1) {
      const programIndex = softmaxSample({
        pool,
        temperature: samplingTemperature,
      });
      const evaluation = await evaluator.evaluateTraced({
        candidate: programs[programIndex] as Candidate<K>,
        batch,
        split: "train",
        phase: "minibatch",
        candidateId: programIndex,
        iteration: step,
      });
      if (evaluation === null) {
        break;
      }

      samples.push({
        programIndex,
        scores: evaluation.scores,
        outputs: evaluation.outputs,
        ...(evaluation.feedback === undefined
          ? {}
          : { feedback: evaluation.feedback }),
      });
      // The sample is a real minibatch mean for that program, so it sharpens
      // the estimate the next step's softmax is drawn from at no extra cost.
      (programScores[programIndex] as number[]).push(
        measuredMean(evaluation) ?? 0,
      );
    }

    if (samples.length === 0) {
      stopReason = "budgetExhausted";
      break;
    }

    const buckets = buildBuckets<Datum, Output>({ batch, samples });
    const allScores = samples.flatMap((sample) => [...sample.scores]);
    const low = percentile(allScores, 10);
    const high = percentile(allScores, 90);

    const built: {
      candidate: Candidate<K>;
      strategy: SimbaStrategy;
      source: number;
    }[] = [];

    for (const bucket of buckets) {
      if (built.length >= candidateCount + 1) {
        break;
      }

      const drawable = available();
      if (drawable.length === 0) {
        break;
      }

      const source = softmaxSample({
        pool: topKPlusBaseline({
          scores: programs.map((_, index) => averageScore(index)),
          k: candidateCount,
        }),
        temperature: candidateTemperature,
      });
      const strategy = rng.pick(drawable);
      const dropped = dropDemos(programs[source] as Candidate<K>);

      let mutated: Candidate<K> | null;
      try {
        mutated =
          strategy === "appendDemo"
            ? appendDemo({ candidate: dropped, bucket, low })
            : await appendRule({ candidate: dropped, bucket, low, high });
      } catch (err) {
        onEvent?.({ type: "error", step, err });
        continue;
      }

      // An unchanged candidate would cost a full minibatch to learn nothing:
      // the strategy declined, and dropping demos happened to change nothing.
      if (
        mutated === null ||
        sameText(mutated, programs[source] as Candidate<K>)
      ) {
        continue;
      }
      built.push({ candidate: mutated, strategy, source });
    }

    let stepBest: { candidate: Candidate<K>; score: number } | undefined;

    const batchIds = batch.map((datum, index) => instanceId({ datum, index }));

    // Every candidate the step built is priced against the allowance before
    // any of them runs. Reserving mid-fan-out instead would decide which of
    // them the step scores by which one reached the budget first.
    const scheduled: typeof built = [];
    let owed = 0;

    for (const entry of built) {
      if (!budget.canAfford(owed + batch.length)) {
        stopReason = "budgetExhausted";
        break;
      }
      owed += evaluator.countUncached({
        candidate: entry.candidate,
        ids: batchIds,
        split: "train",
      });
      scheduled.push(entry);
    }

    // Ids are assigned before the evaluations so concurrent scorers can report
    // the index their candidate will enter the pool at.
    const poolBase = programs.length;
    const scored = await mapDistinct({
      items: scheduled,
      limit: concurrency,
      key: (entry) => candidateHash(entry.candidate),
      task: async (entry, index): Promise<ScoreOutcome> => {
        try {
          return {
            score: measuredMean(
              await evaluator.evaluate({
                candidate: entry.candidate,
                batch,
                ids: batchIds,
                split: "train",
                phase: "minibatch",
                candidateId: poolBase + index,
                iteration: step,
              }),
            ),
          };
        } catch (err) {
          if (err instanceof BudgetExhausted) {
            return { stop: true };
          }
          throw err;
        }
      },
    });

    // Committed in the order the step proposed them: a candidate enters the
    // pool at the index the next step's softmax will sample it by, so letting
    // completion order decide it would renumber the whole search.
    for (const [index, outcome] of scored.entries()) {
      if (outcome.stop === true) {
        stopReason = "budgetExhausted";
        break;
      }
      const { score } = outcome;
      if (score === undefined) {
        continue;
      }
      const entry = scheduled[index] as (typeof built)[number];

      programs.push(entry.candidate);
      programScores.push([score]);
      onEvent?.({
        type: "candidate",
        step,
        strategy: entry.strategy,
        sourceProgram: entry.source,
        minibatchScore: score,
      });

      if (stepBest === undefined || score > stepBest.score) {
        stepBest = { candidate: entry.candidate, score };
      }
    }

    if (stepBest !== undefined) {
      winners.push({ candidate: stepBest.candidate, step: step + 1 });
    }

    await checkpoint(step + 1);

    if (stopReason === "budgetExhausted") {
      step += 1;
      break;
    }
  }

  if (signal?.aborted) {
    stopReason = "aborted";
  }

  // The sweeps that decide the winner are priced as one schedule before any of
  // them runs, so which winners get measured is settled by the allowance
  // rather than by which sweep reserved it first.
  const contenders: { index: number; candidate: Candidate<K>; step: number }[] =
    [];
  let owedForFinalists = 0;

  for (const index of evenlySpacedIndices({
    length: winners.length,
    count: candidateCount + 1,
  })) {
    const winner = winners[index] as { candidate: Candidate<K>; step: number };
    const uncached = evaluator.countUncached({
      candidate: winner.candidate,
      ids: validationIds,
      split: "val",
    });

    if (!budget.canAfford(owedForFinalists + uncached)) {
      break;
    }
    owedForFinalists += uncached;
    contenders.push({ index, candidate: winner.candidate, step: winner.step });
  }

  const sweeps = await mapDistinct({
    items: contenders,
    limit: concurrency,
    key: (contender) => candidateHash(contender.candidate),
    task: async (contender): Promise<ScoreOutcome> => {
      if (signal?.aborted) {
        return { stop: true };
      }
      try {
        return {
          score: measuredMean(
            await evaluator.evaluate({
              candidate: contender.candidate,
              batch: validationSet,
              ids: validationIds,
              split: "val",
              // The seed is always the first winner, and its sweep is the
              // baseline the whole result is reported against.
              phase: contender.index === 0 ? "seed" : "validation",
              candidateId: contender.index,
              iteration: step,
            }),
          ),
        };
      } catch (err) {
        if (err instanceof BudgetExhausted || signal?.aborted) {
          return { stop: true };
        }
        throw err;
      }
    },
  });

  const finalists: SimbaFinalist<K>[] = [];
  // Collected in winner order, because the seed's score is read off the head
  // of this list before it is sorted.
  for (const [position, outcome] of sweeps.entries()) {
    if (outcome.stop === true) {
      break;
    }
    if (outcome.score === undefined) {
      continue;
    }
    const contender = contenders[position] as (typeof contenders)[number];
    finalists.push({
      candidate: contender.candidate,
      score: outcome.score,
      step: contender.step,
    });
  }

  const seedScore = finalists[0]?.score ?? 0;
  finalists.sort((a, b) => b.score - a.score);

  const best = finalists[0] ?? { candidate: seedCandidate, score: seedScore };
  let bestOutputs: (Output | undefined)[] | undefined;

  if (trackBestOutputs) {
    const sweep = await evaluator.evaluateTraced({
      candidate: best.candidate,
      batch: validationSet,
      split: "val",
      phase: "validation",
      candidateId: null,
      iteration: step,
    });
    bestOutputs = sweep?.outputs;
  }

  const testScore =
    testSet === undefined
      ? undefined
      : measuredMean(
          await evaluator.evaluate({
            candidate: best.candidate,
            batch: testSet,
            ids: testIds,
            split: "test",
            phase: "test",
            candidateId: null,
            iteration: step,
            charge: false,
          }),
        );

  onEvent?.({
    type: "finish",
    reason: stopReason,
    bestScore: best.score,
    metricCalls: budget.spent(),
    ...(testScore === undefined ? {} : { testScore }),
  });

  return {
    bestCandidate: best.candidate,
    bestScore: best.score,
    usage: evaluator.usage(),
    seedScore,
    steps: step,
    finalists,
    reflectionCalls,
    cacheHits: evaluator.cacheHits(),
    metricCalls: budget.spent(),
    snapshot: takeSnapshot(step),
    ...(bestOutputs === undefined ? {} : { bestOutputs }),
    ...(testScore === undefined
      ? {}
      : { testScore, testMetricCalls: testSet?.length ?? 0 }),
    stopReason,
  };

  /**
   * The mutations still open. `appendRule` closes once the advice budget is
   * spent; `appendDemo` never closes, because harvesting costs no model call.
   */
  function available(): readonly SimbaStrategy[] {
    if (
      maxReflectionCalls === undefined ||
      reflectionCalls < maxReflectionCalls
    ) {
      return enabled;
    }
    return enabled.filter((strategy) => strategy !== "appendRule");
  }

  function averageScore(index: number): number {
    const observed = programScores[index] ?? [];
    if (observed.length === 0) {
      return 0;
    }
    return (
      observed.reduce((total, score) => total + score, 0) / observed.length
    );
  }

  function softmaxSample(args: {
    pool: readonly number[];
    temperature: number;
  }): number {
    const { pool, temperature } = args;
    return rng.weighted(
      pool,
      softmaxWeights(
        pool.map((index) => averageScore(index)),
        temperature,
      ),
    );
  }

  /**
   * Drops a Poisson-distributed number of demos, always at least one once the
   * block is at its ceiling. Sampled with replacement, matching the reference:
   * a repeated draw drops fewer than the count suggests, which biases the loop
   * toward keeping demos rather than shedding them.
   */
  function dropDemos(candidate: Candidate<K>): Candidate<K> {
    if (demoComponents.length === 0) {
      return { ...candidate };
    }

    const blocks = new Map<K, Demo[]>(
      demoComponents.map((name) => [name, parseDemos(candidate[name] ?? "")]),
    );
    const held = Math.max(...[...blocks.values()].map((demos) => demos.length));
    if (held === 0) {
      return { ...candidate };
    }

    const scale = maxDemos > 0 ? maxDemos : DEMO_DROP_SCALE;
    const wanted = Math.max(
      samplePoisson(rng, held / scale),
      held >= scale ? 1 : 0,
    );
    const drops = new Set<number>();
    for (let draw = 0; draw < Math.min(wanted, held); draw += 1) {
      drops.add(rng.nextInt(held));
    }

    const next = { ...candidate };
    for (const [name, demos] of blocks) {
      next[name] = formatDemos(
        demos.filter((_, index) => !drops.has(index)),
        renderDemo === undefined
          ? {}
          : { render: renderDemo as DemoRenderer<unknown, unknown> },
      );
    }
    return next;
  }

  /**
   * Keep the winning rollout of this instance as a demonstration.
   *
   * Declines when the winner is below the batch's tenth percentile: a demo is
   * an assertion that this is what good looks like, and the worst rollouts of a
   * bad step are not that.
   */
  function appendDemo(args: {
    candidate: Candidate<K>;
    bucket: SimbaBucket<Datum, Output>;
    low: number;
  }): Candidate<K> | null {
    const { candidate, bucket, low } = args;

    const winner = bucket.rollouts[0];
    if (
      winner === undefined ||
      winner.score < low ||
      winner.output === undefined
    ) {
      return null;
    }

    const demo: Demo<Datum, Output> = {
      input: bucket.datum,
      output: winner.output,
      score: winner.score,
    };

    const next = { ...candidate };
    for (const name of demoComponents) {
      const kept = parseDemos(next[name] ?? "");
      next[name] = formatDemos(
        [...kept, demo as Demo],
        renderDemo === undefined
          ? {}
          : { render: renderDemo as DemoRenderer<unknown, unknown> },
      );
    }
    return next;
  }

  /**
   * Ask the reflection model what the better run did differently, and append
   * its answer to each instruction component.
   *
   * Declines when the contrast is not informative: a winner below the tenth
   * percentile is not a success to imitate, and a loser above the ninetieth is
   * not a failure to avoid. When the two runs tied, the uninformative side is
   * withheld and the model advises from one trajectory — a tie at a low score
   * is shown as a failure, a tie at a high score as a success.
   */
  async function appendRule(args: {
    candidate: Candidate<K>;
    bucket: SimbaBucket<Datum, Output>;
    low: number;
    high: number;
  }): Promise<Candidate<K> | null> {
    const { candidate, bucket, low, high } = args;

    if (reflect === undefined) {
      throw new Error("the appendRule strategy requires a reflect model");
    }

    const good = bucket.rollouts[0];
    const bad = bucket.rollouts[bucket.rollouts.length - 1];
    if (good === undefined || bad === undefined) {
      return null;
    }
    if (good.score < low || bad.score > high) {
      return null;
    }

    const tied = good.score <= bad.score;
    const better = tied && good.score <= high ? undefined : trajectory(good);
    const worse = tied && good.score > high ? undefined : trajectory(bad);
    if (better === undefined && worse === undefined) {
      return null;
    }

    const response = await reflect({
      prompt: buildPrompt({
        components: ruleComponents,
        input: bucket.datum,
        ...(better === undefined ? {} : { better }),
        ...(worse === undefined ? {} : { worse }),
      }),
      ...(signal === undefined ? {} : { signal }),
    });
    reflectionCalls += 1;

    const advice = parseAdvice(response);
    const next = { ...candidate };
    let changed = false;

    for (const name of ruleComponents) {
      const rule = advice[name];
      if (rule === undefined) {
        continue;
      }
      next[name] = [next[name] ?? "", rule].filter(Boolean).join("\n\n");
      changed = true;
    }
    return changed ? next : null;
  }

  function trajectory(rollout: {
    output?: Output;
    score: number;
    feedback?: string;
  }) {
    return {
      output: rollout.output as Output,
      score: rollout.score,
      ...(rollout.feedback === undefined ? {} : { feedback: rollout.feedback }),
    };
  }

  function sameText(a: Candidate<K>, b: Candidate<K>): boolean {
    return components.every((name) => a[name] === b[name]);
  }

  function takeSnapshot(completed: number): SimbaSnapshot {
    const cached = checkpointCache ? evaluationCache?.entries?.() : undefined;

    return {
      version: 1,
      fingerprint,
      programs: [...programs],
      programScores: programScores.map((scores) => [...scores]),
      winners: [...winners],
      step: completed,
      metricCalls: budget.spent(),
      reflectionCalls,
      cacheHits: evaluator.cacheHits(),
      rngState: rng.state(),
      ...(sampler.state === undefined ? {} : { sampler: sampler.state() }),
      ...(cached === undefined ? {} : { cache: cached }),
    };
  }

  async function checkpoint(completed: number): Promise<void> {
    if (onCheckpoint === undefined) {
      return;
    }
    await onCheckpoint(takeSnapshot(completed));
  }
}

/**
 * Runs `items` concurrently, except that items sharing a key run one after the
 * other.
 *
 * Two identical candidates cost one evaluation and one cache hit when they are
 * scored in sequence, and two evaluations when they overlap: the cache is only
 * written once a rollout returns. Serializing the duplicates is what keeps a
 * fan-out from buying a second copy of a score the run has already paid for.
 */
async function mapDistinct<Item, Result>(args: {
  items: readonly Item[];
  limit: number;
  key: (item: Item) => string;
  task: (item: Item, index: number) => Promise<Result>;
}): Promise<Result[]> {
  const { items, limit, key, task } = args;

  const groups = new Map<string, number[]>();
  items.forEach((item, index) => {
    const group = groups.get(key(item));
    if (group === undefined) {
      groups.set(key(item), [index]);
      return;
    }
    group.push(index);
  });

  const results = new Array<Result>(items.length);
  await mapWithConcurrency({
    items: [...groups.values()],
    limit,
    task: async (indices) => {
      for (const index of indices) {
        results[index] = await task(items[index] as Item, index);
      }
    },
  });
  return results;
}

/**
 * Range checks on the search knobs, run at construction so a configuration
 * that could never terminate is refused before a task is ever handed to it.
 */
function assertSimbaConfig(config: SimbaConfig): void {
  const { concurrency = 1 } = config;

  if (!Number.isInteger(concurrency) || concurrency < 1) {
    throw new Error(
      `concurrency must be a positive integer, received ${concurrency}`,
    );
  }
}

function defaultInstanceId(args: { datum: unknown; index: number }): string {
  return String(args.index);
}
