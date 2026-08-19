import { createBudget } from "../budget.js";
import { createMemoryCache, stableHash } from "../cache.js";
import type { EvaluationCache } from "../cache.js";
import { mapWithConcurrency } from "../concurrency.js";
import { BudgetExhausted, createEvaluator } from "../evaluation.js";
import type { EvaluationEvent } from "../evaluation.js";
import { mean } from "../math.js";
import type {
  Optimizer,
  OptimizerResult,
  OptimizerTask,
} from "../optimizer.js";
import { createSeededRng } from "../rng.js";
import { parseProposedText } from "../text.js";
import { componentNames } from "../types.js";
import type { Adapter, Candidate, TextModel } from "../types.js";

/** One instruction that was tried, and what it scored. */
/** A history entry plus the system state its score was measured in. */
interface RecordedAttempt extends ScoredAttempt {
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
  onEvent?: (event: OproEvent<NoInfer<K>>) => void;
}

export type OproStopReason =
  "budgetExhausted" | "reflectionBudgetExhausted" | "maxRounds" | "aborted";

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
  | {
      type: "finish";
      reason: OproStopReason;
      bestScore: number;
      metricCalls: number;
      testScore?: number;
    };

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
}

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
    return runOpro({ config: this.#config, task });
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
    instanceId = defaultInstanceId,
    onEvent,
    signal,
  } = task;

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

  const rng = createSeededRng(seed);
  const budget = createBudget({ maxMetricCalls });
  const evaluator = createEvaluator<Datum, Trajectory, Output, K>({
    adapter,
    budget,
    ...(cache === false ? {} : { cache: cache ?? createMemoryCache() }),
    trackOutputs: trackBestOutputs,
    ...(signal === undefined ? {} : { signal }),
    onEvaluation: (event) => onEvent?.({ type: "evaluation", ...event }),
  });

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
    scoringSetSize === undefined
      ? undefined
      : rng.sample(
          trainingSet.map((_, index) => index),
          Math.min(scoringSetSize, trainingSet.length),
        );
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
    components.map((name) => [name, []]),
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

  let round = 0;
  let reflectionCalls = 0;
  let stopReason: OproStopReason = "maxRounds";

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

  const seedEvaluation = await sweep(seedCandidate, "seed");
  const seedScore = mean(seedEvaluation.scores);

  let best = seedCandidate;
  let bestScore = seedScore;
  let bestOutputs = seedEvaluation.outputs;
  // The best candidate a full sweep has actually seen, and the incumbent most
  // recently swept. They are not the same thing: sweeping a candidate that
  // turns out worse than the seed confirms it, and reports the seed.
  let reported = seedCandidate;
  let lastSwept = seedCandidate;

  // What the search compares against. The same number as `bestScore` until a
  // scoring set splits the two apart: the search then runs on the subset while
  // the reported result stays a full-validation set measurement.
  let bestSearchScore = seedScore;
  if (scoringSet !== undefined) {
    bestSearchScore = mean((await screen(seedCandidate, "seed")).scores);
  }

  /**
   * Measures the incumbent on the whole validation set. The search chooses by subset
   * score and can therefore chase something that only works on the subset, so
   * what gets reported is the best candidate a full sweep has actually seen —
   * never a subset number wearing a validation set label.
   */
  async function refreshIncumbent(): Promise<"ok" | "stop"> {
    if (best === lastSwept || !budget.canAfford(validationSet.length)) {
      return "ok";
    }
    try {
      const evaluation = await sweep(best, "validation");
      const full = mean(evaluation.scores);
      lastSwept = best;
      if (full > bestScore) {
        reported = best;
        bestScore = full;
        bestOutputs = evaluation.outputs;
      }
    } catch (err) {
      if (err instanceof BudgetExhausted || signal?.aborted) {
        return "stop";
      }
      throw err;
    }
    return "ok";
  }

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

  while (round < maxRounds) {
    if (signal?.aborted) {
      stopReason = "aborted";
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
    onEvent?.({
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

    let roundStop: OproStopReason | undefined;

    for (const text of unique) {
      const candidate = { ...best, [component]: text } as Candidate<K>;

      let evaluation: Awaited<ReturnType<typeof sweep>>;
      try {
        evaluation = await screen(candidate, "validation");
      } catch (err) {
        if (err instanceof BudgetExhausted) {
          roundStop = "budgetExhausted";
          break;
        }
        if (signal?.aborted) {
          roundStop = "aborted";
          break;
        }
        throw err;
      }

      const score = mean(evaluation.scores);
      const accepted = score > bestSearchScore;

      history.push({ text, score: scaleScore(score, scoreScale), context });
      trajectory.push({ round: round + 1, component, candidate, score });
      onEvent?.({ type: "attempt", round, component, score, accepted });

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

    if (roundStop !== undefined) {
      stopReason = roundStop;
      break;
    }

    if (scoringSet !== undefined && round % fullEvalInterval === 0) {
      if ((await refreshIncumbent()) === "stop") {
        stopReason = signal?.aborted ? "aborted" : "budgetExhausted";
        break;
      }
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

  const testScore =
    testSet === undefined
      ? undefined
      : mean(
          (
            await evaluator.evaluate({
              candidate: best,
              batch: testSet,
              ids: testIds,
              split: "test",
              phase: "test",
              candidateId: null,
              iteration: round,
              charge: false,
            })
          ).scores,
        );

  onEvent?.({
    type: "finish",
    reason: stopReason,
    bestScore,
    metricCalls: budget.spent(),
    ...(testScore === undefined ? {} : { testScore }),
  });

  return {
    bestCandidate: reported,
    bestScore,
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

function defaultInstanceId(args: { datum: unknown; index: number }): string {
  const hash = stableHash(args.datum);
  return hash === "" ? String(args.index) : hash;
}
