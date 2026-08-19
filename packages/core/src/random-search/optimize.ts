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
import { parseProposedText } from "../text.js";
import { componentNames } from "../types.js";
import type { Adapter, Candidate, TextModel } from "../types.js";

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
  onEvent?: (event: RandomSearchEvent<NoInfer<K>>) => void;
}

export type RandomSearchStopReason =
  "budgetExhausted" | "maxRounds" | "aborted";

export type RandomSearchEvent<K extends string = string> =
  | { type: "start"; components: K[]; validationSetSize: number }
  | { type: "roundStart"; round: number; component: K }
  | ({ type: "evaluation" } & EvaluationEvent)
  | {
      type: "candidateAccepted";
      round: number;
      component: K;
      score: number;
      previousScore: number;
    }
  | {
      type: "finish";
      reason: RandomSearchStopReason;
      bestScore: number;
      metricCalls: number;
      testScore?: number;
    };

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
}

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
    return runRandomSearch({ config: this.#config, task });
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

  const budget = createBudget({ maxMetricCalls });
  const evaluator = createEvaluator<Datum, Trajectory, Output, K>({
    adapter,
    budget,
    ...(cache === false ? {} : { cache: cache ?? createMemoryCache() }),
    trackOutputs: trackBestOutputs,
    ...(signal === undefined ? {} : { signal }),
    onEvaluation: (event) => onEvent?.({ type: "evaluation", ...event }),
  });

  let round = 0;
  let variantsEvaluated = 0;
  let reflectionCalls = 0;
  let stopReason: RandomSearchStopReason = "maxRounds";

  onEvent?.({
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

  const seedEvaluation = await sweep({
    candidate: seedCandidate,
    phase: "seed",
  });
  const seedScore = mean(seedEvaluation.scores);

  let best = seedCandidate;
  let bestScore = seedScore;
  let bestOutputs = seedEvaluation.outputs;

  while (round < maxRounds) {
    if (signal?.aborted) {
      stopReason = "aborted";
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
    onEvent?.({ type: "roundStart", round, component });

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

    for (const text of unique) {
      const candidate = { ...best, [component]: text } as Candidate<K>;

      let evaluation: Awaited<ReturnType<typeof sweep>>;
      try {
        evaluation = await sweep({ candidate, phase: "validation" });
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
      variantsEvaluated += 1;

      const score = mean(evaluation.scores);
      if (score > bestScore) {
        onEvent?.({
          type: "candidateAccepted",
          round,
          component,
          score,
          previousScore: bestScore,
        });
        best = candidate;
        bestScore = score;
        bestOutputs = evaluation.outputs;
      }
    }

    round += 1;

    if (roundStop !== undefined) {
      stopReason = roundStop;
      break;
    }
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
    bestCandidate: best,
    bestScore,
    usage: evaluator.usage(),
    seedScore,
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

function defaultInstanceId(args: { datum: unknown; index: number }): string {
  const hash = stableHash(args.datum);
  return hash === "" ? String(args.index) : hash;
}
