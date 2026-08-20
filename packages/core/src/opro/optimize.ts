import { createDeadline } from "../deadline.js";
import { createBudget } from "../budget.js";
import { createMemoryCache, defaultInstanceId, stableHash } from "../cache.js";
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
import { createSeededRng } from "../rng.js";
import { parseProposedText } from "../text.js";
import { componentNames } from "../types.js";
import type { Adapter, Candidate, TextModel, UsageTotals } from "../types.js";

/** One instruction that was tried, and what it scored. */
/** A history entry plus the system state its score was measured in. */
export interface RecordedAttempt extends ScoredAttempt {
  context: string;
}

export interface ScoredAttempt {
  text: string;
  score: number;
}

export type OproPromptBuilder = (args: {
  componentName: string;
  /** Attempts so far, ascending by score — the weakest first. */
  history: readonly ScoredAttempt[];
  /** Rendered task inputs, for grounding. Empty when none were requested. */
  exemplars: readonly string[];
}) => string;

export interface OproConfig {
  /**
   * Instructions drawn per round. Default 8, as in the paper.
   *
   * All eight come from one prompt, so they differ only by the sampling
   * temperature of `reflect`. The paper runs its optimizer at 1.0 and finds
   * below 0.5 explores too little to escape a plateau. At temperature 0 the
   * eight drafts are identical, dedup collapses them to one, and the round
   * costs eight reflection calls to try a single instruction — set a
   * temperature on the model before raising this.
   */
  proposalsPerRound?: number;
  /** How many of them may be in flight at once. Default 1. */
  concurrency?: number;
  maxRounds?: number;
  /**
   * Reflection is the expensive half of this search and no metric budget
   * covers it, so it is bounded separately.
   */
  maxReflectionCalls?: number;
  seed?: number;
  /**
   * Scored attempts the prompt carries, strongest kept. Default 20, matching
   * the paper's `max_num_instructions`.
   *
   * The reference also drops attempts below an absolute score threshold
   * (`old_instruction_score_threshold`, 0.3 for its GPT scorers) before this
   * cut. Nothing here does: the weak tail is what marks the bottom of the
   * range the model is reading a gradient across, and once twenty decent
   * attempts exist the strongest-kept rule has retired it anyway.
   */
  historySize?: number;
  /**
   * Task inputs shown for grounding, redrawn each round. Default 3, the
   * reference's `num_few_shot_questions_for_instruction_refinement`, which
   * resamples them per step under its default `random` selection.
   */
  exemplars?: number;
  /**
   * Instances drawn once from the training set to screen proposals on. Unset means
   * every proposal is measured on the whole validation set, which is the reliable
   * reading and the expensive one: a round of eight proposals against a
   * 500-instance validation set costs 4000 rollouts before anything is learned.
   *
   * The paper screens on a small fixed slice of the training set and checks the
   * full set periodically, which is what lets the validation set be large enough to
   * trust. The slice is drawn once and never resampled — the meta-prompt ranks
   * attempts against each other, so they have to be measured on the same
   * instances or the ranking is noise.
   */
  scoringSetSize?: number;
  /**
   * Rounds between full validation set sweeps of the incumbent. Default 3, the paper's
   * `eval_interval`. Only used when `scoringSetSize` is set; without it every
   * proposal is already a full sweep.
   */
  fullEvalInterval?: number;
  /**
   * What scores are multiplied by before being shown. Models discriminate
   * between 41 and 68 far more reliably than between 0.41 and 0.68, which is
   * the whole mechanism this search runs on. Default 100.
   */
  scoreScale?: number;
  buildPrompt?: OproPromptBuilder;
  trackBestOutputs?: boolean;
  /**
   * Include cached instance scores in every checkpoint. Leaving them out keeps
   * snapshots small at the cost of a resumed run re-paying for rollouts it
   * cannot look up. Default true.
   */
  checkpointCache?: boolean;
}

/**
 * Everything needed to continue a run: the per-component score histories the
 * meta-prompt is written from, the incumbent, the budget already spent, the
 * position of the random stream, and the screening slice — which is drawn once
 * and must survive a resume, since attempts screened on different instances
 * are not the gradient this search reads.
 */
export interface OproSnapshot {
  version: 1;
  fingerprint: string;
  best: Candidate;
  reported: Candidate;
  /** Whether the incumbent has already been confirmed by a full sweep. */
  incumbentSwept: boolean;
  bestScore: number;
  bestSearchScore: number;
  seedScore: number;
  round: number;
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
  rngState: number;
  /** Component name -> every text tried for it, with what it scored. */
  histories: Record<string, RecordedAttempt[]>;
  /** Training set positions the screening slice was drawn from. */
  scoringIndices?: number[];
  cache?: [string, CachedScore][];
}

export interface OproTask<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> extends OptimizerTask<Datum, Trajectory, Output, K> {
  /**
   * The base adapter, not `GepaAdapter`: this search reads scores only, so it
   * never asks for traces or a reflective dataset.
   */
  adapter: Adapter<Datum, Trajectory, Output, NoInfer<K>>;
  reflect: TextModel;
  /** Renders a task input for the prompt. Defaults to JSON. */
  renderDatum?: (datum: NoInfer<Datum>) => string;
  instanceId?: (args: { datum: NoInfer<Datum>; index: number }) => string;
  /** Pass `false` to disable caching entirely. */
  cache?: EvaluationCache | false;
  /** Observers of the run. Every one sees every event; none can fail it. */
  reporters?: readonly Reporter<OproEvent<NoInfer<K>>>[];
  /**
   * Called with a resumable snapshot after the seed is scored and after every
   * round. Persist it and a killed run costs the last round, not all of them.
   */
  onCheckpoint?: (snapshot: OproSnapshot) => void | Promise<void>;
  /** Snapshot to continue from, instead of starting at the seed candidate. */
  resumeFrom?: OproSnapshot;
}

export type OproStopReason =
  | "budgetExhausted"
  | "costExhausted"
  | "deadlineReached"
  | "reflectionBudgetExhausted"
  | "proposalsExhausted"
  | "maxRounds"
  | "aborted";

export type OproEvent<K extends string = string> =
  | { type: "start"; components: K[]; validationSetSize: number }
  | { type: "roundStart"; round: number; component: K; historySize: number }
  | ({ type: "evaluation" } & EvaluationEvent)
  | {
      type: "attempt";
      round: number;
      component: K;
      score: number;
      /** True when this attempt became the new incumbent. */
      accepted: boolean;
    }
  | ({ type: "candidateAccepted"; round: number } & CandidateAccepted<K>)
  | ({ type: "finish"; reason: OproStopReason } & RunFinished);

export interface OproAttempt<K extends string = string> {
  round: number;
  component: K;
  candidate: Candidate<K>;
  score: number;
}

export interface OproResult<
  K extends string = string,
  Output = unknown,
> extends OptimizerResult<K, OproStopReason, Output> {
  /** The seed's score, so the lift the search bought is readable directly. */
  seedScore: number;
  rounds: number;
  /** Every candidate scored, in the order it was tried. */
  trajectory: OproAttempt<K>[];
  reflectionCalls: number;
  cacheHits: number;
  /** State as of the last round, ready to hand back as `resumeFrom`. */
  snapshot: OproSnapshot;
}

/**
 * What one proposal's screen produced, or why it never ran. Carried rather
 * than thrown so a round that ran out of allowance or was cancelled still
 * commits the proposals ahead of it, exactly as the serial round did.
 */
type AttemptOutcome<Output> =
  | { evaluation: ScoredBatch<Output>; stop?: undefined }
  | { evaluation?: undefined; stop: OproStopReason };

/**
 * Rounds that may pass without a proposal nobody has tried before, per
 * component, before the run gives up.
 *
 * Such a round spends no rollouts, so the budget guard — the only thing
 * bounding a run at the default round limit — never fires, and a proposal
 * model that has settled on one answer would loop forever. Waiting a few
 * rounds costs one prompt each and leaves room for a stochastic model to find
 * something new; waiting indefinitely is a hang.
 */
const BARREN_ROUNDS = 3;

const DEFAULT_PROPOSALS_PER_ROUND = 8;
const DEFAULT_HISTORY_SIZE = 20;
const DEFAULT_EXEMPLARS = 3;
const DEFAULT_SCORE_SCALE = 100;
const DEFAULT_FULL_EVAL_INTERVAL = 3;

/**
 * Optimization by prompting: show the model what has been tried and what each
 * attempt scored, and ask for something better.
 *
 * The regime this is for is the one GEPA cannot serve. GEPA's advantage over
 * blind search is the per-instance feedback string — a diagnosis of *why* a
 * rollout failed. Plenty of metrics cannot produce one: a reward model, a
 * preference score, a classifier's accuracy over a closed label set. Handed
 * those, reflection is asked to diagnose a failure it has no evidence about,
 * and pays a frontier-model call to guess.
 *
 * This search asks for less and needs less. One scalar per candidate is the
 * entire signal, and the model reasons over the *trajectory* of scores rather
 * than the anatomy of a single failure. Where feedback does exist, use GEPA:
 * a score history is a much thinner channel than a paragraph saying what broke.
 */
export class OproOptimizer implements Optimizer<OproStopReason> {
  readonly #config: OproConfig;

  constructor(config: OproConfig = {}) {
    assertConfig(config);
    this.#config = config;
  }

  async optimize<
    Datum,
    Trajectory = unknown,
    Output = unknown,
    const K extends string = string,
  >(
    task: OproTask<Datum, Trajectory, Output, K>,
  ): Promise<OproResult<K, Output>> {
    try {
      return await runOpro({ config: this.#config, task });
    } finally {
      await flushReporters(task.reporters ?? []);
    }
  }
}

/**
 * Adapted from the meta-prompt in *Large Language Models as Optimizers* (Yang
 * et al., 2023). Ascending order is load-bearing: the paper finds the model
 * attends most to what is nearest the end of the prompt, so the best attempt
 * has to be last.
 *
 * Exemplars carry an `<INS>` marker for the same reason the paper's do: an
 * instruction written against the task in the abstract reads differently from
 * one written to sit in a particular slot. The marker is placed ahead of the
 * input, which is where a component's text usually goes but which this
 * optimizer cannot actually know — only the adapter composes the real prompt.
 * The paper also shows each exemplar's gold answer. Nothing here needs to know
 * what that is: `renderDatum` is handed the whole datum, so a caller holding
 * labels can render them alongside the input.
 */
export function buildOproPrompt(args: {
  componentName: string;
  history: readonly ScoredAttempt[];
  exemplars: readonly string[];
}): string {
  const { componentName, history, exemplars } = args;

  return [
    `I am tuning the "${componentName}" component of a larger system. Below are the instructions I have tried, each with the score it achieved. Higher scores are better.`,
    "",
    "<attempts>",
    history
      .map(
        (attempt) =>
          `score: ${attempt.score}\n<instruction>\n${attempt.text}\n</instruction>`,
      )
      .join("\n\n"),
    "</attempts>",
    ...(exemplars.length === 0
      ? []
      : [
          "",
          "Here are examples of the inputs this component receives. `<INS>` marks where your instruction is read:",
          "",
          "<inputs>",
          exemplars
            .map((exemplar) => `input:\n<INS>\n${exemplar}`)
            .join("\n\n"),
          "</inputs>",
        ]),
    "",
    "Write a new instruction that will score higher than every instruction above.",
    "Work out what the higher-scoring instructions do that the lower-scoring ones do not, and push further in that direction.",
    "Do not repeat an instruction that has already been tried.",
    "",
    "Return only the new instruction, inside a ``` block.",
  ].join("\n");
}

async function runOpro<Datum, Trajectory, Output, K extends string>(args: {
  config: OproConfig;
  task: OproTask<Datum, Trajectory, Output, K>;
}): Promise<OproResult<K, Output>> {
  const { config, task } = args;

  const {
    proposalsPerRound = DEFAULT_PROPOSALS_PER_ROUND,
    concurrency = 1,
    maxRounds = Number.POSITIVE_INFINITY,
    maxReflectionCalls = Number.POSITIVE_INFINITY,
    seed = 0,
    historySize = DEFAULT_HISTORY_SIZE,
    exemplars = DEFAULT_EXEMPLARS,
    scoringSetSize,
    fullEvalInterval = DEFAULT_FULL_EVAL_INTERVAL,
    scoreScale = DEFAULT_SCORE_SCALE,
    buildPrompt = buildOproPrompt,
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
    renderDatum = renderDefault,
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

  const emit = createEmitter<OproEvent<K>>(reporters);

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
    ...(resumeFrom?.usage === undefined ? {} : { usage: resumeFrom.usage }),
    ...(signal === undefined ? {} : { signal }),
    onEvaluation: (event) => emit({ type: "evaluation", ...event }),
  });

  evaluator.restore(resumeFrom?.cache ?? []);

  // Redrawn every round, as the reference does with its `random` few-shot
  // selection. A slice held fixed for the whole run lets the search tune its
  // instruction to those particular inputs and call the result a gain.
  function drawExemplars(): string[] {
    return rng
      .sample(trainingSet, Math.min(exemplars, trainingSet.length))
      .map(renderDatum);
  }

  // Drawn once. Every attempt is screened on these same instances, because the
  // meta-prompt asks the model to read a gradient across scores and a gradient
  // across different instances is not one.
  const scoringIndices =
    resumeFrom?.scoringIndices ??
    (scoringSetSize === undefined
      ? undefined
      : rng.sample(
          trainingSet.map((_, index) => index),
          Math.min(scoringSetSize, trainingSet.length),
        ));
  const scoringSet = scoringIndices?.map(
    (index) => trainingSet[index] as Datum,
  );
  const scoringIds = scoringIndices?.map((index) =>
    instanceId({ datum: trainingSet[index] as Datum, index }),
  );

  // Per component: every text tried for it and what it scored. Kept apart
  // because a score earned while rewriting one component says nothing about
  // what the text of another should be.
  //
  // Each attempt also carries the context it was measured in — the text of
  // every other component at the time. The paper optimizes a single string,
  // where that context is constant and the score history is a clean gradient.
  // Generalizing to several components breaks that: an attempt scored before
  // another component moved was measured in a system that no longer exists,
  // and listing it beside current scores asks the model to read a trend across
  // measurements that were never comparable.
  const histories = new Map<K, RecordedAttempt[]>(
    components.map((name) => [name, [...(resumeFrom?.histories[name] ?? [])]]),
  );

  function contextOf(candidate: Candidate<K>, component: K): string {
    const rest: Record<string, string> = {};
    for (const name of components) {
      if (name !== component) {
        rest[name] = candidate[name];
      }
    }
    return stableHash(rest);
  }
  const trajectory: OproAttempt<K>[] = [];

  let round = resumeFrom?.round ?? 0;
  let reflectionCalls = resumeFrom?.reflectionCalls ?? 0;
  let stopReason: OproStopReason = "maxRounds";

  emit({
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
      iteration: round,
    });
  }

  /**
   * Screens a candidate. With a scoring set that is a slice of the training set;
   * without one it is the full validation set, and screening and reporting are the
   * same measurement.
   */
  async function screen(candidate: Candidate<K>, phase: "seed" | "validation") {
    if (scoringSet === undefined) {
      return sweep(candidate, phase);
    }
    return evaluator.evaluate({
      candidate,
      batch: scoringSet,
      ids: scoringIds as string[],
      split: "train",
      phase: "minibatch",
      candidateId: null,
      iteration: round,
    });
  }

  /** What screening a candidate would cost, in rollouts nothing has cached. */
  function screenCost(candidate: Candidate<K>): number {
    return scoringSet === undefined
      ? evaluator.countUncached({ candidate, ids: validationIds, split: "val" })
      : evaluator.countUncached({
          candidate,
          ids: scoringIds as string[],
          split: "train",
        });
  }

  // A resumed run already knows what the seed scored. Re-sweeping it would
  // charge the budget a second time for a number the checkpoint carries.
  const seedEvaluation =
    resumeFrom === undefined ? await sweep(seedCandidate, "seed") : undefined;
  const seedScore =
    seedEvaluation === undefined
      ? (resumeFrom as OproSnapshot).seedScore
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
  /** Absent on a resumed run until a sweep wins: outputs are not checkpointed. */
  let acceptedCandidates = resumeFrom?.acceptedCandidates ?? 0;
  let bestOutputs = seedEvaluation?.outputs;
  // The best candidate a full sweep has actually seen, and the incumbent most
  // recently swept. They are not the same thing: sweeping a candidate that
  // turns out worse than the seed confirms it, and reports the seed.
  let reported =
    (resumeFrom?.reported as Candidate<K> | undefined) ?? seedCandidate;
  // Identity, not equality: on a resume the flag says whether the incumbent
  // was already confirmed, and a fresh object stands in for "not this one".
  let lastSwept =
    resumeFrom === undefined
      ? seedCandidate
      : resumeFrom.incumbentSwept
        ? best
        : ({} as Candidate<K>);

  // What the search compares against. The same number as `bestScore` until a
  // scoring set splits the two apart: the search then runs on the subset while
  // the reported result stays a full-validation set measurement.
  let bestSearchScore = resumeFrom?.bestSearchScore ?? seedScore;
  if (resumeFrom === undefined && scoringSet !== undefined) {
    bestSearchScore = requireMeasuredMean({
      batch: await screen(seedCandidate, "seed"),
      phase: "seed",
    });
  }

  /**
   * Measures the incumbent on the whole validation set. The search chooses by subset
   * score and can therefore chase something that only works on the subset, so
   * what gets reported is the best candidate a full sweep has actually seen —
   * never a subset number wearing a validation set label.
   */
  /**
   * The incumbent moved and a full sweep measured it. Emitted from the two
   * places that can be true — a screening run with no scoring set, where the
   * attempt's own evaluation is the sweep, and the cadence that confirms an
   * incumbent later — because a payload assembled twice is one that drifts.
   */
  function emitAccepted(args: {
    candidate: Candidate<K>;
    evaluation: ScoredBatch<Output>;
    score: number;
  }): void {
    acceptedCandidates += 1;
    emit({
      type: "candidateAccepted",
      round,
      candidateId: acceptedCandidates,
      candidate: args.candidate,
      aggregateScore: args.score,
      instanceScores: instanceRow(args.evaluation),
      ...(trackBestOutputs ? { outputs: args.evaluation.outputs } : {}),
    });
  }

  async function refreshIncumbent(): Promise<"ok" | "stop"> {
    if (best === lastSwept || !budget.canAfford(validationSet.length)) {
      return "ok";
    }
    try {
      const evaluation = await sweep(best, "validation");
      const full = measuredMean(evaluation);
      lastSwept = best;
      // A sweep that measured nothing cannot confirm the incumbent, and
      // reporting its zero would replace a real number with an outage.
      if (full !== undefined && full > bestScore) {
        reported = best;
        bestScore = full;
        bestOutputs = evaluation.outputs;
        emitAccepted({ candidate: best, evaluation, score: full });
      }
    } catch (err) {
      if (err instanceof BudgetExhausted || signal?.aborted) {
        return "stop";
      }
      throw err;
    }
    return "ok";
  }

  if (resumeFrom === undefined) {
    for (const name of components) {
      histories.get(name)?.push({
        text: seedCandidate[name],
        score: scaleScore(bestSearchScore, scoreScale),
        context: contextOf(seedCandidate, name),
      });
    }
    trajectory.push({
      round: 0,
      component: components[0] as K,
      candidate: seedCandidate,
      score: seedScore,
    });
  }

  function takeSnapshot(): OproSnapshot {
    const cached = checkpointCache ? evaluationCache?.entries?.() : undefined;

    return {
      version: 1,
      fingerprint,
      best,
      reported,
      incumbentSwept: best === lastSwept,
      bestScore,
      bestSearchScore,
      seedScore,
      round,
      reflectionCalls,
      metricCalls: budget.spent(),
      cacheHits: evaluator.cacheHits(),
      usage: evaluator.usage(),
      acceptedCandidates,
      rngState: rng.state(),
      histories: Object.fromEntries(
        [...histories].map(([name, attempts]) => [name, [...attempts]]),
      ),
      ...(scoringIndices === undefined ? {} : { scoringIndices }),
      ...(cached === undefined ? {} : { cache: cached }),
    };
  }

  async function checkpoint(): Promise<void> {
    if (onCheckpoint === undefined) {
      return;
    }
    await onCheckpoint(takeSnapshot());
  }

  await checkpoint();

  let barrenRounds = 0;

  while (round < maxRounds) {
    if (signal?.aborted) {
      stopReason = "aborted";
      break;
    }
    if (barrenRounds >= BARREN_ROUNDS * components.length) {
      stopReason = "proposalsExhausted";
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
    if (reflectionCalls >= maxReflectionCalls) {
      stopReason = "reflectionBudgetExhausted";
      break;
    }
    if (!budget.canAfford(validationSet.length)) {
      stopReason = "budgetExhausted";
      break;
    }

    const component = components[round % components.length] as K;
    const history = histories.get(component) as RecordedAttempt[];
    const context = contextOf(best, component);
    const comparable = history.filter((attempt) => attempt.context === context);
    emit({
      type: "roundStart",
      round,
      component,
      historySize: comparable.length,
    });

    const affordableProposals = Math.min(
      proposalsPerRound,
      maxReflectionCalls - reflectionCalls,
    );
    const prompt = buildPrompt({
      componentName: component,
      history: topAttempts({ history: comparable, keep: historySize }),
      exemplars: drawExemplars(),
    });

    reflectionCalls += affordableProposals;
    const drawn = await mapWithConcurrency({
      items: Array.from({ length: affordableProposals }, (_, index) => index),
      limit: concurrency,
      signal,
      // Every proposal in a round sees the same history: the round is one
      // question asked several times, not a chain in which each answer
      // changes what the next one is asked.
      task: async () =>
        parseProposedText(
          await reflect({
            prompt,
            ...(signal === undefined ? {} : { signal }),
          }),
        ),
    });

    // Scoped to this context, not the whole history: a candidate is the entire
    // assignment, so once another component moved, a text that failed before
    // names a candidate nobody has measured. Deduping across contexts would
    // retire it on the strength of a result that no longer describes it.
    const tried = new Set(comparable.map((attempt) => attempt.text));
    const unique = [...new Set(drawn)].filter(
      (text) => text.length > 0 && !tried.has(text),
    );
    barrenRounds = unique.length === 0 ? barrenRounds + 1 : 0;

    let roundStop: OproStopReason | undefined;

    // Every screen in the round is priced against the allowance before any of
    // them runs. Reserving mid-fan-out instead would hand the round to
    // whichever proposal reached the budget first, so which proposals a round
    // screens would stop being a property of the search.
    const scheduled: Candidate<K>[] = [];
    let owed = 0;

    for (const text of unique) {
      // Every proposal in a round replaces the same component of the same
      // incumbent, so a sibling accepted mid-round changes nothing about what
      // the rest are: building them all here is what the serial round reached.
      const candidate = { ...best, [component]: text } as Candidate<K>;
      const cost = screenCost(candidate);

      if (!budget.canAfford(owed + cost)) {
        roundStop = "budgetExhausted";
        break;
      }
      owed += cost;
      scheduled.push(candidate);
    }

    const screened = await mapWithConcurrency({
      items: scheduled,
      limit: concurrency,
      // Cancellation is reported per proposal rather than thrown out of the
      // fan-out, so an aborted round still records the screens it paid for.
      task: async (candidate): Promise<AttemptOutcome<Output>> => {
        if (signal?.aborted) {
          return { stop: "aborted" };
        }
        try {
          return { evaluation: await screen(candidate, "validation") };
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

    // Committed in draw order, not completion order: the history is what the
    // next round's meta-prompt reads as a gradient, and an attempt counts as
    // accepted only if it beat every attempt drawn before it.
    for (const [index, outcome] of screened.entries()) {
      if (outcome.stop !== undefined) {
        roundStop = outcome.stop;
        break;
      }
      const candidate = scheduled[index] as Candidate<K>;
      const text = candidate[component];
      const evaluation = outcome.evaluation;

      const score = measuredMean(evaluation);
      // The attempt measured the provider rather than the text. Recording it
      // would put a score in the history that no rollout produced, and the
      // next meta-prompt asks the model to read a gradient across that history.
      if (score === undefined) {
        continue;
      }
      const accepted = score > bestSearchScore;

      history.push({ text, score: scaleScore(score, scoreScale), context });
      trajectory.push({ round: round + 1, component, candidate, score });
      emit({ type: "attempt", round, component, score, accepted });

      if (accepted) {
        best = candidate;
        bestSearchScore = score;

        // Without a scoring set this evaluation already is the full sweep, so
        // there is nothing left to confirm.
        if (scoringSet === undefined) {
          reported = candidate;
          lastSwept = candidate;
          bestScore = score;
          bestOutputs = evaluation.outputs;
          emitAccepted({ candidate, evaluation, score });
        }

        // Accepting this candidate changed the context every other component's
        // history was measured in, which would leave their next prompt with
        // nothing to show. The sweep just run is itself a valid reading of
        // each of their current texts under the new context, so record it:
        // one anchor costs no rollouts and keeps the filtered history from
        // collapsing to empty every time a sibling moves.
        const scaled = scaleScore(score, scoreScale);
        for (const name of components) {
          if (name === component) {
            continue;
          }
          histories.get(name)?.push({
            text: candidate[name],
            score: scaled,
            context: contextOf(candidate, name),
          });
        }
      }
    }

    round += 1;

    // Before the checkpoint, not after: a snapshot names the round it was
    // taken at, and a resumed run schedules the next sweep an interval past
    // that round. Checkpointing first would describe half a round and the
    // resume would skip this sweep entirely.
    let cadenceStop: OproStopReason | undefined;
    if (
      roundStop === undefined &&
      scoringSet !== undefined &&
      round % fullEvalInterval === 0 &&
      (await refreshIncumbent()) === "stop"
    ) {
      cadenceStop = signal?.aborted ? "aborted" : "budgetExhausted";
    }

    await checkpoint();

    if (roundStop !== undefined) {
      stopReason = roundStop;
      break;
    }
    if (cadenceStop !== undefined) {
      stopReason = cadenceStop;
      break;
    }
  }

  // The cadence can leave the run's last acceptance unconfirmed, and an
  // incumbent nothing ever swept cannot be reported.
  if (scoringSet !== undefined && !signal?.aborted) {
    await refreshIncumbent();
  }

  if (signal?.aborted) {
    stopReason = "aborted";
  }

  const heldOut =
    testSet === undefined
      ? undefined
      : await evaluator.evaluate({
          // The returned candidate, not the search incumbent: a subset-scored
          // run can end on a `best` the closing sweep never confirmed, and a
          // held-out number naming a candidate the caller never sees is worse
          // than no number at all.
          candidate: reported,
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
    snapshot: takeSnapshot(),
    bestCandidate: reported,
    bestScore,
    usage: evaluator.usage(),
    seedScore,
    ...(trackBestOutputs ? { bestOutputs } : {}),
    ...(testScore === undefined
      ? {}
      : { testScore, testMetricCalls: evaluator.unchargedCalls() }),
    rounds: round,
    trajectory,
    metricCalls: budget.spent(),
    reflectionCalls,
    cacheHits: evaluator.cacheHits(),
    stopReason,
  };
}

/**
 * The strongest `keep` attempts, ordered weakest first. Dropping the weak tail
 * rather than the old one is deliberate: the prompt's job is to show a
 * gradient, and an attempt that scored badly a hundred rounds ago still marks
 * the bottom of it.
 */
function topAttempts(args: {
  history: readonly ScoredAttempt[];
  keep: number;
}): ScoredAttempt[] {
  const { history, keep } = args;

  return [...history]
    .sort((a, b) => b.score - a.score)
    .slice(0, keep)
    .sort((a, b) => a.score - b.score);
}

function scaleScore(score: number, scale: number): number {
  return Math.round(score * scale);
}

function renderDefault(datum: unknown): string {
  if (typeof datum === "string") {
    return datum;
  }
  try {
    return JSON.stringify(datum, null, 2) ?? String(datum);
  } catch {
    return String(datum);
  }
}

function assertConfig(config: OproConfig): void {
  const positive: [string, number | undefined][] = [
    ["proposalsPerRound", config.proposalsPerRound],
    ["concurrency", config.concurrency],
    ["maxRounds", config.maxRounds],
    ["maxReflectionCalls", config.maxReflectionCalls],
    ["historySize", config.historySize],
    ["scoringSetSize", config.scoringSetSize],
    ["fullEvalInterval", config.fullEvalInterval],
  ];
  for (const [name, value] of positive) {
    if (value !== undefined && (!Number.isInteger(value) || value < 1)) {
      throw new Error(`${name} must be a positive integer, received ${value}`);
    }
  }
  if (
    config.exemplars !== undefined &&
    (!Number.isInteger(config.exemplars) || config.exemplars < 0)
  ) {
    throw new Error(
      `exemplars must be a non-negative integer, received ${config.exemplars}`,
    );
  }
}
