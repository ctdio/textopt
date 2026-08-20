import { createDeadline } from "../deadline.js";
import { createBudget } from "../budget.js";
import { createMemoryCache, defaultInstanceId } from "../cache.js";
import type { CachedScore, EvaluationCache } from "../cache.js";
import { assertResumable, runFingerprint } from "../checkpoint.js";
import { mapWithConcurrency } from "../concurrency.js";
import {
  BudgetExhausted,
  costExhausted,
  createEvaluator,
  measuredMean,
  requireMeasuredMean,
} from "../evaluation.js";
import type { EvaluationEvent, ScoredBatch } from "../evaluation.js";
import type {
  Optimizer,
  OptimizerResult,
  OptimizerTask,
} from "../optimizer.js";
import { createEmitter, flushReporters, instanceRow } from "../reporting.js";
import type { CandidateAccepted, Reporter, RunFinished } from "../reporting.js";
import { parseProposedText } from "../text.js";
import { componentNames } from "../types.js";
import type { Adapter, Candidate, TextModel, UsageTotals } from "../types.js";

/** Builds the prompt one variant is drawn from. */
export type ParaphrasePromptBuilder = (args: {
  componentName: string;
  currentText: string;
  /** Which of this round's variants is being drawn, from 0. */
  attempt: number;
}) => string;

/**
 * How the baseline searches. Deliberately small: every knob GEPA has that this
 * one lacks is a knob whose value GEPA has to earn.
 */
export interface RandomSearchConfig {
  /** Variants drawn per round, each evaluated in full. Default 4. */
  variants?: number;
  /** How many of them may be in flight at once. Default 1. */
  concurrency?: number;
  maxRounds?: number;
  /** Replaces the default paraphrase template. */
  buildPrompt?: ParaphrasePromptBuilder;
  /** Keep what the winner produced on each validation instance. */
  trackBestOutputs?: boolean;
  /**
   * Include cached instance scores in every checkpoint. Leaving them out keeps
   * snapshots small at the cost of a resumed run re-paying for rollouts it
   * cannot look up. Default true.
   */
  checkpointCache?: boolean;
}

/**
 * Everything needed to continue a run: the incumbent and its score, the budget
 * already spent, and the bookkeeping that stops the seed from being re-scored.
 * Plain JSON — persist it and hand it back as `resumeFrom`.
 *
 * There is no random stream here because this search has none: its variants
 * come from the proposer alone.
 */
export interface RandomSearchSnapshot {
  version: 1;
  /**
   * Identifies the run this came from — seed candidate, instance ids, cache
   * namespace. Resuming against a different setup is refused rather than
   * silently scoring an old incumbent against new data.
   */
  fingerprint: string;
  best: Candidate;
  bestScore: number;
  seedScore: number;
  round: number;
  variantsEvaluated: number;
  reflectionCalls: number;
  metricCalls: number;
  cacheHits: number;
  /** Usage already spent, so a resumed run reports totals and honours ceilings. */
  usage?: UsageTotals;
  /**
   * Candidates accepted so far. Reporters key rows by this id, so restarting it
   * at zero makes a resumed run collide with the run it continues.
   */
  acceptedCandidates?: number;
  /** Cached instance scores, when the cache can enumerate them. */
  cache?: [string, CachedScore][];
}

export interface RandomSearchTask<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> extends OptimizerTask<Datum, Trajectory, Output, K> {
  /**
   * The base adapter, not `GepaAdapter`: this search never reflects, so it has
   * no use for a reflective dataset and does not ask for one.
   */
  adapter: Adapter<Datum, Trajectory, Output, NoInfer<K>>;
  /** Rewrites a component's text. Sees the text and nothing else. */
  reflect: TextModel;
  instanceId?: (args: { datum: NoInfer<Datum>; index: number }) => string;
  /** Pass `false` to disable caching entirely. */
  cache?: EvaluationCache | false;
  /** Observers of the run. Every one sees every event; none can fail it. */
  reporters?: readonly Reporter<RandomSearchEvent<NoInfer<K>>>[];
  /**
   * Called with a resumable snapshot after the seed is scored and after every
   * round. Persist it and a killed run costs the last round, not all of them.
   */
  onCheckpoint?: (snapshot: RandomSearchSnapshot) => void | Promise<void>;
  /** Snapshot to continue from, instead of starting at the seed candidate. */
  resumeFrom?: RandomSearchSnapshot;
}

export type RandomSearchStopReason =
  | "budgetExhausted"
  | "costExhausted"
  | "deadlineReached"
  | "maxRounds"
  | "proposerStalled"
  | "aborted";

export type RandomSearchEvent<K extends string = string> =
  | { type: "start"; components: K[]; validationSetSize: number }
  | { type: "roundStart"; round: number; component: K }
  | ({ type: "evaluation" } & EvaluationEvent)
  | ({ type: "candidateAccepted"; round: number } & CandidateAccepted<K>)
  | ({ type: "finish"; reason: RandomSearchStopReason } & RunFinished);

export interface RandomSearchResult<
  K extends string = string,
  Output = unknown,
> extends OptimizerResult<K, RandomSearchStopReason, Output> {
  /** The seed's score, so the lift the search bought is readable directly. */
  seedScore: number;
  rounds: number;
  /** Variants drawn and evaluated, including the ones that lost. */
  variantsEvaluated: number;
  reflectionCalls: number;
  cacheHits: number;
  /** State as of the last round, ready to hand back as `resumeFrom`. */
  snapshot: RandomSearchSnapshot;
}

/**
 * What one variant's sweep produced, or why it never ran. Carried rather than
 * thrown so a round that ran out of allowance or was cancelled still commits
 * the variants ahead of it, exactly as the serial round did.
 */
type VariantOutcome<Output> =
  | { evaluation: ScoredBatch<Output>; stop?: undefined }
  | { evaluation?: undefined; stop: RandomSearchStopReason };

const DEFAULT_VARIANTS = 4;

/**
 * The ablation baseline: propose blind, evaluate in full, keep what wins.
 *
 * It exists to be beaten. Reflective search costs a frontier-model call per
 * proposal on top of its rollouts, and running the same budget through a
 * search that cannot read feedback at all puts a floor under what that call
 * has to buy. A GEPA run that does not clear this one is paying for machinery
 * it is not using.
 *
 * What the gap measures is the whole of that machinery, not reflection alone:
 * this drops feedback, Pareto parent selection, and minibatch screening
 * together, and GEPA's own ablations put candidate selection at several points
 * by itself. For reflection in isolation, keep GEPA and hand it a
 * `reflection.buildPrompt` that withholds the evidence — same frontier, same
 * screening, one variable.
 *
 * Not DSPy's `BootstrapFewShotWithRandomSearch`, which the literature usually
 * means by "random search" and which searches bootstrapped demo sets; nor
 * random search in the Bergstra–Bengio sense, since proposals are paraphrases
 * of the incumbent rather than independent draws. No reference implements this
 * baseline — it is original here, and the GEPA paper has no equivalent.
 */
export class RandomSearchOptimizer implements Optimizer<RandomSearchStopReason> {
  readonly #config: RandomSearchConfig;

  constructor(config: RandomSearchConfig = {}) {
    assertConfig(config);
    this.#config = config;
  }

  async optimize<
    Datum,
    Trajectory = unknown,
    Output = unknown,
    const K extends string = string,
  >(
    task: RandomSearchTask<Datum, Trajectory, Output, K>,
  ): Promise<RandomSearchResult<K, Output>> {
    try {
      return await runRandomSearch({ config: this.#config, task });
    } finally {
      await flushReporters(task.reporters ?? []);
    }
  }
}

/**
 * Adapted from the GEPA reflection prompt with every source of evidence
 * removed. What is left is the ablation: the same model, the same component,
 * the same output format, and nothing to reason from.
 */
export function buildParaphrasePrompt(args: {
  componentName: string;
  currentText: string;
  attempt: number;
}): string {
  const { componentName, currentText, attempt } = args;

  return [
    `Here is the current instruction for the "${componentName}" component of a larger system:`,
    "",
    "<current_instruction>",
    currentText,
    "</current_instruction>",
    "",
    "Write a different instruction for this component.",
    "You have no information about how the current one has performed, so do not guess at its weaknesses — vary it instead.",
    `Make variation ${attempt + 1} distinct from the others: change the wording, the level of detail, or the strategy it describes.`,
    "",
    "Return only the new instruction, inside a ``` block.",
  ].join("\n");
}

async function runRandomSearch<
  Datum,
  Trajectory,
  Output,
  K extends string,
>(args: {
  config: RandomSearchConfig;
  task: RandomSearchTask<Datum, Trajectory, Output, K>;
}): Promise<RandomSearchResult<K, Output>> {
  const { config, task } = args;

  const {
    variants = DEFAULT_VARIANTS,
    concurrency = 1,
    maxRounds = Number.POSITIVE_INFINITY,
    buildPrompt = buildParaphrasePrompt,
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
    maxMetricCalls,
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

  const emit = createEmitter<RandomSearchEvent<K>>(reporters);

  const deadline = createDeadline({ maxWallClockMs });
  const components = componentNames(seedCandidate);

  if (trainingSet.length === 0) {
    throw new Error("optimize requires a non-empty trainingSet");
  }
  if (validationSet.length === 0) {
    throw new Error("optimize requires a non-empty validationSet");
  }
  if (components.length === 0) {
    throw new Error(
      "optimize requires a seed candidate with at least one component",
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
    ...(cacheNamespace === undefined ? {} : { cacheNamespace }),
  });
  assertResumable({
    fingerprint,
    ...(resumeFrom === undefined ? {} : { snapshot: resumeFrom }),
  });

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
    ...(resumeFrom?.usage === undefined ? {} : { usage: resumeFrom.usage }),
    ...(signal === undefined ? {} : { signal }),
    onEvaluation: (event) => emit({ type: "evaluation", ...event }),
  });

  evaluator.restore(resumeFrom?.cache ?? []);

  let round = resumeFrom?.round ?? 0;
  let variantsEvaluated = resumeFrom?.variantsEvaluated ?? 0;
  let reflectionCalls = resumeFrom?.reflectionCalls ?? 0;
  let stopReason: RandomSearchStopReason = "maxRounds";
  /**
   * Consecutive rounds that neither spent a rollout nor improved on the
   * incumbent. A proposer stuck on texts that are already cached costs
   * nothing, so neither the metric budget nor the cost ceiling can end the
   * run — without this the loop spins forever, burning reflection calls that
   * no budget here bounds.
   */
  let stalledRounds = 0;

  emit({
    type: "start",
    components,
    validationSetSize: validationSet.length,
  });

  async function sweep(args: {
    candidate: Candidate<K>;
    phase: "seed" | "validation";
  }) {
    return evaluator.evaluate({
      candidate: args.candidate,
      batch: validationSet,
      ids: validationIds,
      split: "val",
      phase: args.phase,
      candidateId: null,
      iteration: round,
    });
  }

  function takeSnapshot(): RandomSearchSnapshot {
    const cached = checkpointCache ? evaluationCache?.entries?.() : undefined;

    return {
      version: 1,
      fingerprint,
      best,
      bestScore,
      seedScore,
      round,
      variantsEvaluated,
      reflectionCalls,
      metricCalls: budget.spent(),
      cacheHits: evaluator.cacheHits(),
      usage: evaluator.usage(),
      acceptedCandidates,
      ...(cached === undefined ? {} : { cache: cached }),
    };
  }

  async function checkpoint(): Promise<void> {
    if (onCheckpoint === undefined) {
      return;
    }
    await onCheckpoint(takeSnapshot());
  }

  // A resumed run already knows what the seed scored. Re-sweeping it would
  // charge the budget a second time for a number the checkpoint carries.
  const seedEvaluation =
    resumeFrom === undefined
      ? await sweep({ candidate: seedCandidate, phase: "seed" })
      : undefined;
  const seedScore =
    seedEvaluation === undefined
      ? (resumeFrom as RandomSearchSnapshot).seedScore
      : requireMeasuredMean({ batch: seedEvaluation, phase: "seed" });

  // The seed is the baseline every later candidate is read against, and its
  // sweep is a full measurement like any other. A report that starts at the
  // first improvement has nothing to compare the improvement to. A resumed run
  // does not re-emit it: the run that swept it already did.
  if (seedEvaluation !== undefined) {
    emit({
      type: "candidateAccepted",
      round: 0,
      candidateId: 0,
      candidate: seedCandidate,
      aggregateScore: seedScore,
      instanceScores: instanceRow(seedEvaluation),
      ...(trackBestOutputs ? { outputs: seedEvaluation.outputs } : {}),
    });
  }

  let best = (resumeFrom?.best as Candidate<K> | undefined) ?? seedCandidate;
  let bestScore = resumeFrom?.bestScore ?? seedScore;
  /** Absent on a resumed run until a variant wins: outputs are not checkpointed. */
  let bestOutputs = seedEvaluation?.outputs;
  // Numbers acceptances rather than proposals: the seed is 0, and every id
  // after it names a candidate a full validation sweep actually preferred.
  let acceptedCandidates = resumeFrom?.acceptedCandidates ?? 0;

  await checkpoint();

  while (round < maxRounds) {
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
    // A round is only worth starting if every variant in it can be both
    // proposed and scored: a half-funded round spends rollouts on variants
    // that can never be compared against the rest.
    if (!budget.canAfford(variants * validationSet.length)) {
      stopReason = "budgetExhausted";
      break;
    }

    const component = components[round % components.length] as K;
    emit({ type: "roundStart", round, component });
    const spentBefore = budget.spent();
    const scoreBefore = bestScore;

    const currentText = best[component];
    const drawn = await mapWithConcurrency({
      items: Array.from({ length: variants }, (_, attempt) => attempt),
      limit: concurrency,
      signal,
      task: async (attempt) => {
        reflectionCalls += 1;
        const response = await reflect({
          prompt: buildPrompt({
            componentName: component,
            currentText,
            attempt,
          }),
          ...(signal === undefined ? {} : { signal }),
        });
        return parseProposedText(response);
      },
    });

    // Duplicates of each other, and of the incumbent, are free to discard and
    // expensive to score: this search has no memory of what it has tried, so
    // without this a stuck proposer re-buys the same sweep every round.
    const unique = [...new Set(drawn)].filter(
      (text) => text.length > 0 && text !== currentText,
    );

    let roundStop: RandomSearchStopReason | undefined;

    // Every sweep in the round is priced against the allowance before any of
    // them runs. Reserving mid-fan-out instead would hand the round to
    // whichever variant reached the budget first, so which variants a round
    // scores would stop being a property of the search.
    const scheduled: Candidate<K>[] = [];
    let owed = 0;

    for (const text of unique) {
      // Every variant replaces the same component of the same incumbent, so a
      // sibling that wins mid-round changes nothing about what the rest are:
      // building them all here is the candidate the serial round would reach.
      const candidate = { ...best, [component]: text } as Candidate<K>;
      const uncached = evaluator.countUncached({
        candidate,
        ids: validationIds,
        split: "val",
      });

      if (!budget.canAfford(owed + uncached)) {
        roundStop = "budgetExhausted";
        break;
      }
      owed += uncached;
      scheduled.push(candidate);
    }

    const swept = await mapWithConcurrency({
      items: scheduled,
      limit: concurrency,
      // Cancellation is reported per variant rather than thrown out of the
      // fan-out, so an aborted round still commits the sweeps it paid for.
      task: async (candidate): Promise<VariantOutcome<Output>> => {
        if (signal?.aborted) {
          return { stop: "aborted" };
        }
        try {
          return {
            evaluation: await sweep({ candidate, phase: "validation" }),
          };
        } catch (err) {
          if (err instanceof BudgetExhausted) {
            return { stop: "budgetExhausted" };
          }
          if (signal?.aborted) {
            return { stop: "aborted" };
          }
          throw err;
        }
      },
    });

    // Committed in draw order, not completion order: a variant is accepted
    // only if it beats every variant drawn before it, and deciding that on
    // whichever sweep returned first would make the run's lineage a property
    // of the network.
    for (const [index, outcome] of swept.entries()) {
      if (outcome.stop !== undefined) {
        roundStop = outcome.stop;
        break;
      }
      variantsEvaluated += 1;

      const score = measuredMean(outcome.evaluation);
      // The variant measured the provider rather than its own text, so there
      // is nothing here to compare the incumbent against.
      if (score !== undefined && score > bestScore) {
        const candidate = scheduled[index] as Candidate<K>;
        acceptedCandidates += 1;

        emit({
          type: "candidateAccepted",
          round,
          candidateId: acceptedCandidates,
          candidate,
          aggregateScore: score,
          instanceScores: instanceRow(outcome.evaluation),
          ...(trackBestOutputs ? { outputs: outcome.evaluation.outputs } : {}),
        });

        best = candidate;
        bestScore = score;
        bestOutputs = outcome.evaluation.outputs;
      }
    }

    round += 1;
    stalledRounds =
      budget.spent() === spentBefore && bestScore === scoreBefore
        ? stalledRounds + 1
        : 0;
    await checkpoint();

    if (stalledRounds >= components.length) {
      stopReason = "proposerStalled";
      break;
    }

    if (roundStop !== undefined) {
      stopReason = roundStop;
      break;
    }
  }

  if (signal?.aborted) {
    stopReason = "aborted";
  }

  const heldOut =
    testSet === undefined
      ? undefined
      : await evaluator.evaluate({
          candidate: best,
          batch: testSet,
          ids: testIds,
          split: "test",
          phase: "test",
          candidateId: null,
          iteration: round,
          charge: false,
        });
  const testScore = heldOut === undefined ? undefined : measuredMean(heldOut);

  emit({
    type: "finish",
    reason: stopReason,
    bestCandidateId: acceptedCandidates,
    bestScore,
    metricCalls: budget.spent(),
    ...(testScore === undefined ? {} : { testScore }),
    ...(heldOut === undefined
      ? {}
      : { testInstanceScores: instanceRow(heldOut) }),
    ...(heldOut === undefined || !trackBestOutputs
      ? {}
      : { testOutputs: heldOut.outputs }),
  });

  return {
    bestCandidate: best,
    bestScore,
    usage: evaluator.usage(),
    seedScore,
    snapshot: takeSnapshot(),
    ...(trackBestOutputs ? { bestOutputs } : {}),
    ...(testScore === undefined
      ? {}
      : { testScore, testMetricCalls: evaluator.unchargedCalls() }),
    rounds: round,
    variantsEvaluated,
    metricCalls: budget.spent(),
    reflectionCalls,
    cacheHits: evaluator.cacheHits(),
    stopReason,
  };
}

function assertConfig(config: RandomSearchConfig): void {
  const { variants = DEFAULT_VARIANTS, concurrency = 1, maxRounds } = config;

  if (!Number.isInteger(variants) || variants < 1) {
    throw new Error(
      `variants must be a positive integer, received ${variants}`,
    );
  }
  if (!Number.isInteger(concurrency) || concurrency < 1) {
    throw new Error(
      `concurrency must be a positive integer, received ${concurrency}`,
    );
  }
  if (
    maxRounds !== undefined &&
    (!Number.isInteger(maxRounds) || maxRounds < 1)
  ) {
    throw new Error(
      `maxRounds must be a positive integer, received ${maxRounds}`,
    );
  }
}
