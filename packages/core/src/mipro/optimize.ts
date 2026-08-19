import { createDeadline } from "../deadline.js";
import { createBudget } from "../budget.js";
import { createMemoryCache, stableHash } from "../cache.js";
import type { CachedScore, EvaluationCache } from "../cache.js";
import { assertResumable, runFingerprint } from "../checkpoint.js";
import { mapWithConcurrency } from "../concurrency.js";
import { formatDemos, harvestFewShotExamples } from "../demos.js";
import type { Demo, DemoRenderer } from "../demos.js";
import {
  BudgetExhausted,
  costExhausted,
  createEvaluator,
  measuredMean,
  requireMeasuredMean,
} from "../evaluation.js";
import type { EvaluationEvent } from "../evaluation.js";
import { mean } from "../math.js";
import type {
  Optimizer,
  OptimizerResult,
  OptimizerTask,
} from "../optimizer.js";
import { createEmitter, flushReporters, instanceRow } from "../reporting.js";
import type { CandidateAccepted, Reporter, RunFinished } from "../reporting.js";
import { createSeededRng } from "../rng.js";
import { createEpochShuffledSampler } from "../sampling.js";
import type { BatchSampler } from "../sampling.js";
import { parseProposedText } from "../text.js";
import { componentNames } from "../types.js";
import type { Adapter, Candidate, TextModel } from "../types.js";
import { proposeConfiguration } from "./tpe.js";
import type { Observation } from "./tpe.js";

export type MiproPromptBuilder = (args: {
  componentName: string;
  seedText: string;
  /** Rendered task inputs, for grounding. Empty when none were requested. */
  exemplars: readonly string[];
  /** A style hint, varied per draw so the menu is not four of one idea. */
  tip: string;
}) => string;

export interface MiproConfig {
  /**
   * Instructions generated per component, beyond the seed. Ignored for a
   * component the caller supplied a menu for. Default 3.
   */
  instructionsPerComponent?: number;
  /**
   * Instances a trial is scored on. Default 35, MIPROv2's `minibatch_size`.
   *
   * The surrogate reads these means as evidence, so the size sets how much of
   * what it learns is signal. Shrinking it is the cheapest way to buy trials
   * and the fastest way to make them worthless: the good/bad split at thirty
   * trials is three or four observations, and a lucky small minibatch is
   * enough to put the wrong configuration among them.
   *
   * MIPROv2 also abandons minibatching altogether when the validation set is
   * 50 instances or fewer (`MIN_MINIBATCH_SIZE`), evaluating every trial in
   * full. textopt has no such mode; on a small validation set, set this to the
   * set's size to get the same behaviour.
   */
  minibatchSize?: number;
  /** Configurations evaluated. Default 30. */
  maxTrials?: number;
  /** Trials drawn uniformly before the surrogate takes over. Default 10. */
  startupTrials?: number;
  /**
   * Fraction of observations the surrogate treats as good. Defaults to
   * Optuna's rule — a tenth of them, capped at 25 — which narrows the good set
   * as observations accumulate rather than holding a fixed share.
   */
  gamma?: number;
  /** Configurations the surrogate draws per trial. Default 24. */
  surrogateSamples?: number;
  /**
   * Whether the surrogate models components jointly. Default true, matching
   * MIPROv2. Set false when the components do independent jobs — see
   * `proposeConfiguration` for the tradeoff.
   */
  multivariate?: boolean;
  /**
   * Trials between full validation sweeps. Every interval, the configuration
   * with the best *average* minibatch reading that has not been swept yet is
   * evaluated in full. Averaging is the point: a single minibatch is a noisy
   * reading, and promoting on one alone lets a lucky draw decide the run.
   * Default 5, MIPROv2's `minibatch_full_eval_steps`.
   *
   * MIPROv2 sweeps when `trial_num % (minibatch_full_eval_steps + 1) == 0`,
   * which reads like every sixth trial but is not: Optuna numbers the seed
   * baseline and each full evaluation as trials of their own, so the six slots
   * hold five minibatch trials and one sweep. Its own budgeting says the same
   * thing directly — `num_trials // minibatch_full_eval_steps + 1` sweeps for
   * `num_trials` trials. This counts only minibatch trials, so 5 here and 5
   * there describe one schedule.
   */
  fullEvalInterval?: number;
  /**
   * Bootstrapped demo sets generated per demo component, beyond the seed text
   * and the zero-shot option. Default 3.
   */
  demoSets?: number;
  /** Demos in the largest generated set. Default 4. */
  maxDemos?: number;
  /**
   * Score a rollout must reach to be kept as a demo. Unset keeps every rollout
   * the metric rewarded at all, as MIPROv2 does without a `metric_threshold`.
   * Set 1 to demand a perfect score, which suits a boolean metric and discards
   * most of a graded one.
   */
  demoMinScore?: number;
  /** Task inputs shown when generating instructions. Default 3. */
  exemplars?: number;
  /**
   * Summarize the training set with one reflection call and show that summary to
   * the proposer, as MIPROv2's grounded proposer does. Default true. The
   * summary reads more data than the exemplars can fit, so it describes the
   * task rather than a few instances of it. Costs one reflection call, and is
   * skipped entirely when every menu was supplied and nothing is proposed.
   */
  datasetSummary?: boolean;
  /** Trainset entries the summary is written from. Default 10. */
  summaryExamples?: number;
  /** How many instruction proposals may be in flight at once. Default 1. */
  concurrency?: number;
  seed?: number;
  buildPrompt?: MiproPromptBuilder;
  /** Replaces the built-in style hints. */
  tips?: readonly string[];
  trackBestOutputs?: boolean;
  /**
   * Include cached instance scores in every checkpoint. Leaving them out keeps
   * snapshots small at the cost of a resumed run re-paying for rollouts it
   * cannot look up. Default true.
   */
  checkpointCache?: boolean;
}

/**
 * Everything needed to continue a run.
 *
 * The menus matter most. Building them is the expensive half of a MIPRO run —
 * a reflection call per instruction and a harvesting pass per demo set — and
 * they are also what every trial's choice vector indexes into, so a resumed
 * run that rebuilt them would both pay twice and reinterpret every
 * observation it had already made.
 */
export interface MiproSnapshot {
  version: 1;
  fingerprint: string;
  /** Component name -> its option menu, in the order choices index it. */
  menu: Record<string, string[]>;
  best: Candidate;
  bestScore: number;
  seedScore: number;
  trial: number;
  fullEvaluations: number;
  reflectionCalls: number;
  bootstrapMetricCalls: number;
  metricCalls: number;
  cacheHits: number;
  rngState: number;
  observations: MiproObservation[];
  /** What the surrogate was fitted on: one entry per measured trial. */
  surrogateInput: { choices: number[]; score: number }[];
  /** Configuration key -> its minibatch readings so far. */
  readings: [string, number[]][];
  /** Configuration keys a full sweep has already bought. */
  swept: string[];
  /** Whatever the batch sampler reports from `state()`, when it has one. */
  sampler?: unknown;
  cache?: [string, CachedScore][];
}

export interface MiproTask<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> extends OptimizerTask<Datum, Trajectory, Output, K> {
  /**
   * The base adapter, not `GepaAdapter`: the surrogate reads scores only, so
   * this search never asks for traces or a reflective dataset.
   */
  adapter: Adapter<Datum, Trajectory, Output, NoInfer<K>>;
  /** Generates the menu for components no menu was supplied for. */
  reflect: TextModel;
  /**
   * Menu entries for a component, used verbatim and never rewritten. This is
   * where a bootstrapped demo block belongs: demos are harvested, not
   * authored, and handing them to a rewriting model destroys them.
   */
  componentOptions?: Partial<Record<NoInfer<K>, readonly string[]>>;
  /**
   * Components holding few-shot demo blocks. Their menus are bootstrapped from
   * the trainingSet rather than written by `reflect` — MIPROv2 searches
   * instructions and demonstrations together, and a demo is evidence a rollout
   * actually produced, which asking a model to author would destroy.
   */
  demoComponents?: readonly NoInfer<K>[];
  /** Renders a harvested rollout as demo text. Defaults to JSON. */
  renderDemo?: DemoRenderer<NoInfer<Datum>, NoInfer<Output>>;
  /**
   * The gold output for a training datum, where the caller has labels. Supply
   * it and every demo component keeps a labels-only set on its menu, as
   * MIPROv2 does — the one demo set that costs no rollouts at all, since the
   * output is known rather than produced. Return `undefined` for an unlabelled
   * datum. Nothing generic can infer this: only the caller knows which part of
   * a datum is the answer.
   */
  goldOutput?: (datum: NoInfer<Datum>) => NoInfer<Output> | undefined;
  /** Renders a task input for the proposal prompt. Defaults to JSON. */
  renderDatum?: (datum: NoInfer<Datum>) => string;
  batchSampler?: BatchSampler<NoInfer<Datum>>;
  instanceId?: (args: { datum: NoInfer<Datum>; index: number }) => string;
  /** Pass `false` to disable caching entirely. */
  cache?: EvaluationCache | false;
  /** Observers of the run. Every one sees every event; none can fail it. */
  reporters?: readonly Reporter<MiproEvent<NoInfer<K>>>[];
  /**
   * Called with a resumable snapshot once the menus are built and after every
   * trial. Persist it and a killed run costs the last trial, not the menus.
   */
  onCheckpoint?: (snapshot: MiproSnapshot) => void | Promise<void>;
  /** Snapshot to continue from, instead of starting at the seed candidate. */
  resumeFrom?: MiproSnapshot;
}

export type MiproStopReason =
  | "budgetExhausted"
  | "costExhausted"
  | "deadlineReached"
  | "maxTrials"
  | "aborted";

export type MiproEvent<K extends string = string> =
  | { type: "start"; components: K[]; validationSetSize: number }
  | { type: "menu"; menu: Record<K, string[]>; reflectionCalls: number }
  | ({ type: "evaluation" } & EvaluationEvent)
  | {
      type: "trial";
      trial: number;
      choices: number[];
      minibatchScore: number;
      /** True when the trial earned a full validation sweep. */
      promoted: boolean;
    }
  | ({ type: "candidateAccepted"; trial: number } & CandidateAccepted<K>)
  | ({ type: "finish"; reason: MiproStopReason } & RunFinished);

export interface MiproObservation {
  trial: number;
  choices: number[];
  minibatchScore: number;
  promoted: boolean;
  /** The full validation score, present only on promoted trials. */
  score?: number;
}

export interface MiproResult<
  K extends string = string,
  Output = unknown,
> extends OptimizerResult<K, MiproStopReason, Output> {
  /** The seed's full validation score, so the lift is readable directly. */
  seedScore: number;
  trials: number;
  /** The search space that was actually explored, per component. */
  menu: Record<K, string[]>;
  observations: MiproObservation[];
  /** Trials that earned a full sweep. The rest were minibatch readings only. */
  fullEvaluations: number;
  /** Rollouts spent harvesting demos, included in `metricCalls`. */
  bootstrapMetricCalls: number;
  reflectionCalls: number;
  cacheHits: number;
  /** State as of the last trial, ready to hand back as `resumeFrom`. */
  snapshot: MiproSnapshot;
}

const DEFAULT_INSTRUCTIONS = 3;
const DEFAULT_MINIBATCH_SIZE = 35;
const DEFAULT_MAX_TRIALS = 30;
const DEFAULT_FULL_EVAL_INTERVAL = 5;
const DEFAULT_DEMO_SETS = 3;
const DEFAULT_MAX_DEMOS = 4;
const DEFAULT_EXEMPLARS = 3;
const DEFAULT_SUMMARY_EXAMPLES = 10;

/**
 * Style hints, one per generated instruction. Drawing four instructions from
 * one prompt yields four rewordings of one idea; varying the hint is what
 * makes the menu a spread of approaches instead.
 */
const DEFAULT_TIPS = [
  "Be concise. Say only what changes the output.",
  "Be specific and detailed. Spell out the edge cases and the output format.",
  "Describe the reasoning the component should do before it answers.",
  "State the constraints as hard rules the component must never break.",
  "Write it as a role description: who the component is and what it cares about.",
];

/**
 * Joint search over a fixed menu, guided by a surrogate.
 *
 * The gap this fills is interaction between components. Reflective search
 * updates one component per iteration and screens it in isolation, so a pair
 * of components that only pay off together is invisible to it — a routing rule
 * and the prompt it routes to, an output format and the instruction that
 * assumes it. Merge recombines lineages after the fact but never proposes a
 * joint move.
 *
 * This search proposes a menu of options per component up front, then treats
 * the choice of one option per component as a single categorical
 * configuration and lets a TPE decide which to spend a trial on. Trials run on
 * minibatches; a configuration that beats the best minibatch reading earns a
 * full sweep before it can become the incumbent, so the number reported is
 * never a lucky minibatch.
 *
 * What it gives up is the ability to write text it did not think of at the
 * start. The menu is fixed at trial one — reflective search keeps writing new
 * text for the whole run. Neither dominates; they fail differently.
 */
export class MiproOptimizer implements Optimizer<MiproStopReason> {
  readonly #config: MiproConfig;

  constructor(config: MiproConfig = {}) {
    assertConfig(config);
    this.#config = config;
  }

  async optimize<
    Datum,
    Trajectory = unknown,
    Output = unknown,
    const K extends string = string,
  >(
    task: MiproTask<Datum, Trajectory, Output, K>,
  ): Promise<MiproResult<K, Output>> {
    try {
      return await runMipro({ config: this.#config, task });
    } finally {
      await flushReporters(task.reporters ?? []);
    }
  }
}

export function buildMiproPrompt(args: {
  componentName: string;
  seedText: string;
  exemplars: readonly string[];
  tip: string;
  /** The other components' current text, so the proposal fits the system. */
  siblings?: Readonly<Record<string, string>>;
  /** What the trainingSet looks like as a whole, beyond the exemplars. */
  datasetSummary?: string;
}): string {
  const {
    componentName,
    seedText,
    exemplars,
    tip,
    siblings = {},
    datasetSummary,
  } = args;
  const others = Object.entries(siblings).filter(([, text]) => text.length > 0);

  return [
    `I am writing the "${componentName}" component of a larger system. Here is the instruction it currently uses:`,
    "",
    "<current_instruction>",
    seedText,
    "</current_instruction>",
    ...(others.length === 0
      ? []
      : [
          "",
          "The rest of the system reads as follows. Write something that fits alongside it rather than repeating or contradicting it:",
          "",
          "<system>",
          others
            .map(([name, text]) => `<${name}>\n${text}\n</${name}>`)
            .join("\n"),
          "</system>",
        ]),
    ...(datasetSummary === undefined
      ? []
      : [
          "",
          "Here is what the data it runs on looks like:",
          "",
          "<dataset_summary>",
          datasetSummary,
          "</dataset_summary>",
        ]),
    ...(exemplars.length === 0
      ? []
      : [
          "",
          "Here are examples of the inputs this component receives:",
          "",
          "<inputs>",
          exemplars.join("\n\n"),
          "</inputs>",
        ]),
    "",
    "Write an alternative instruction for this component — a different approach to the same job, not an edit of the one above.",
    `Follow this style: ${tip}`,
    "",
    "Return only the new instruction, inside a ``` block.",
  ].join("\n");
}

function buildDatasetSummaryPrompt(examples: readonly string[]): string {
  return [
    "Below are entries from a dataset a system is being tuned against.",
    "",
    "<examples>",
    examples.join("\n\n"),
    "</examples>",
    "",
    "Describe what this dataset is: what the inputs are, what varies between them, and what answering one well requires. Two or three sentences, concrete rather than generic.",
    "",
    "Return only the description, inside a ``` block.",
  ].join("\n");
}

async function runMipro<Datum, Trajectory, Output, K extends string>(args: {
  config: MiproConfig;
  task: MiproTask<Datum, Trajectory, Output, K>;
}): Promise<MiproResult<K, Output>> {
  const { config, task } = args;

  const {
    instructionsPerComponent = DEFAULT_INSTRUCTIONS,
    minibatchSize = DEFAULT_MINIBATCH_SIZE,
    maxTrials = DEFAULT_MAX_TRIALS,
    startupTrials,
    gamma,
    surrogateSamples,
    multivariate,
    fullEvalInterval = DEFAULT_FULL_EVAL_INTERVAL,
    demoSets = DEFAULT_DEMO_SETS,
    maxDemos = DEFAULT_MAX_DEMOS,
    demoMinScore,
    exemplars = DEFAULT_EXEMPLARS,
    datasetSummary = true,
    summaryExamples = DEFAULT_SUMMARY_EXAMPLES,
    concurrency = 1,
    seed = 0,
    buildPrompt = buildMiproPrompt,
    tips = DEFAULT_TIPS,
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
    componentOptions,
    demoComponents,
    renderDemo,
    goldOutput,
    maxMetricCalls,
    renderDatum = renderDefault,
    batchSampler = createEpochShuffledSampler<Datum>({ minibatchSize }),
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

  const emit = createEmitter<MiproEvent<K>>(reporters);

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

  for (const name of demoComponents ?? []) {
    if (componentOptions?.[name] !== undefined) {
      throw new Error(
        `component "${name}" is listed in both demoComponents and componentOptions; pick one source for its menu`,
      );
    }
  }

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
    onEvaluation: (event) => emit({ type: "evaluation", ...event }),
  });

  evaluator.restore(resumeFrom?.cache ?? []);
  if (resumeFrom?.sampler !== undefined) {
    batchSampler.restore?.(resumeFrom.sampler);
  }

  emit({
    type: "start",
    components,
    validationSetSize: validationSet.length,
  });

  const shown = rng
    .sample(trainingSet, Math.min(exemplars, trainingSet.length))
    .map(renderDatum);

  let reflectionCalls = resumeFrom?.reflectionCalls ?? 0;
  const menu = {} as Record<K, string[]>;

  const demoNames = new Set<K>(demoComponents ?? []);
  // Only components whose menu has to be written cost a reflection call, and
  // the summary only earns its call if at least one of them exists.
  const proposing = components.some(
    (name) => !demoNames.has(name) && componentOptions?.[name] === undefined,
  );

  let summary: string | undefined;
  if (
    resumeFrom === undefined &&
    datasetSummary &&
    proposing &&
    trainingSet.length > 0
  ) {
    reflectionCalls += 1;
    summary = parseProposedText(
      await reflect({
        prompt: buildDatasetSummaryPrompt(
          rng
            .sample(trainingSet, Math.min(summaryExamples, trainingSet.length))
            .map(renderDatum),
        ),
        ...(signal === undefined ? {} : { signal }),
      }),
    );
  }

  let bootstrapMetricCalls = resumeFrom?.bootstrapMetricCalls ?? 0;

  /**
   * Builds a demo component's menu from rollouts the metric rewarded.
   *
   * Each set gets its own harvesting pass over a freshly shuffled training set, as
   * MIPROv2 does. Drawing every set from a single pool would be cheaper and
   * identical under deterministic scoring, but a system that answers at
   * temperature does not give the same verdict twice: a second pass can turn a
   * previously failing example into a demo, and one pass can never show it.
   * Sizes vary across the sets because more demos is not monotonically better
   * — a long block crowds out the instruction, and which length wins is
   * exactly what the search settles.
   *
   * Not covered here: MIPROv2 also builds label-only sets from gold outputs,
   * and pads bootstrapped sets with them. A gold output is something only the
   * adapter knows, so there is nothing generic for this to read.
   */
  async function bootstrapMenu(name: K): Promise<string[]> {
    const blocks: string[] = [];

    // Labels first, and free: a gold output is already the answer, so this set
    // needs no rollout and survives a system too weak to bootstrap from.
    if (goldOutput !== undefined) {
      const labelled = rng
        .shuffle(trainingSet)
        .map((datum) => ({ input: datum, output: goldOutput(datum) }))
        .filter((demo) => demo.output !== undefined)
        .slice(0, maxDemos) as Demo<Datum, Output>[];

      if (labelled.length > 0) {
        blocks.push(
          formatDemos(
            labelled,
            renderDemo === undefined ? {} : { render: renderDemo },
          ),
        );
      }
    }

    for (let index = 0; index < demoSets; index += 1) {
      // Demos are optional; scoring the seed is not. Harvesting is capped so it
      // can never eat the sweep that establishes the baseline every trial is
      // measured against — a run that bootstraps its way out of a starting
      // score has nothing to report. Re-checked per pass, so a run that can
      // afford two sets builds two rather than failing on the third.
      const affordable = Math.min(
        trainingSet.length,
        budget.remaining() - validationSet.length,
      );
      if (affordable < 1) {
        break;
      }

      // Sizes span 1..maxDemos so the largest set is always the full one, the
      // way MIPROv2 always keeps an unshuffled max-size set in the running.
      const requested =
        demoSets === 1
          ? maxDemos
          : Math.round(1 + (index * (maxDemos - 1)) / (demoSets - 1));

      const harvest = await harvestFewShotExamples<
        Datum,
        Trajectory,
        Output,
        K
      >({
        adapter,
        candidate: seedCandidate,
        trainingSet,
        ...(demoMinScore === undefined ? {} : { minScore: demoMinScore }),
        maxDemos: requested,
        maxMetricCalls: affordable,
        rng,
        ...(renderDemo === undefined ? {} : { renderDemo }),
        ...(signal === undefined ? {} : { signal }),
      });

      bootstrapMetricCalls += harvest.metricCalls;
      budget.reserve(harvest.metricCalls);

      if (harvest.demos.length > 0) {
        blocks.push(
          formatDemos(
            harvest.demos,
            renderDemo === undefined ? {} : { render: renderDemo },
          ),
        );
      }
    }

    // The zero-shot option is always on the menu: demos can hurt, and MIPROv2
    // keeps "no demonstrations" in the running for that reason.
    return [...new Set([seedCandidate[name], "", ...blocks])].filter(
      (text, index) =>
        index === 0 || text.length === 0 || text.includes("<demo>"),
    );
  }

  for (const name of components) {
    const supplied = componentOptions?.[name];

    // A resumed run reuses the menus it already paid for. Rebuilding them
    // would buy the same options a second time and, worse, reindex every
    // choice vector the surrogate has already been fitted on.
    const restored = resumeFrom?.menu[name];
    if (restored !== undefined) {
      menu[name] = [...restored];
      continue;
    }

    if (demoNames.has(name)) {
      menu[name] = await bootstrapMenu(name);
      continue;
    }

    if (supplied !== undefined) {
      // Verbatim: a supplied option is usually a harvested demo block, and
      // asking a model to "improve" one turns evidence back into prose.
      menu[name] = [seedCandidate[name], ...supplied];
      continue;
    }

    const drawn = await mapWithConcurrency({
      items: Array.from(
        { length: instructionsPerComponent },
        (_, index) => index,
      ),
      limit: concurrency,
      signal,
      task: async (index) => {
        reflectionCalls += 1;
        return parseProposedText(
          await reflect({
            prompt: buildPrompt({
              componentName: name,
              seedText: seedCandidate[name],
              exemplars: shown,
              tip: tips[index % tips.length] as string,
              ...(summary === undefined || summary.length === 0
                ? {}
                : { datasetSummary: summary }),
              siblings: Object.fromEntries(
                components
                  .filter((other) => other !== name)
                  .map((other) => [other, seedCandidate[other]]),
              ),
            }),
            ...(signal === undefined ? {} : { signal }),
          }),
        );
      },
    });

    menu[name] = [
      seedCandidate[name],
      ...drawn.filter((text) => text.length > 0),
    ];
  }

  emit({ type: "menu", menu, reflectionCalls });

  const menuSizes = components.map((name) => (menu[name] as string[]).length);

  function candidateFor(choices: readonly number[]): Candidate<K> {
    const candidate = {} as Candidate<K>;
    components.forEach((name, index) => {
      candidate[name] = (menu[name] as string[])[
        choices[index] as number
      ] as string;
    });
    return candidate;
  }

  let trial = resumeFrom?.trial ?? 0;
  let fullEvaluations = resumeFrom?.fullEvaluations ?? 0;
  const observations: MiproObservation[] = [
    ...(resumeFrom?.observations ?? []),
  ];
  const surrogateInput: Observation[] = [...(resumeFrom?.surrogateInput ?? [])];
  let stopReason: MiproStopReason = "maxTrials";

  async function fullSweep(
    candidate: Candidate<K>,
    phase: "seed" | "validation",
  ) {
    return evaluator.evaluate({
      candidate,
      batch: validationSet,
      ids: validationIds,
      split: "val",
      phase,
      candidateId: null,
      iteration: trial,
    });
  }

  // A resumed run already knows what the seed scored, and re-sweeping it would
  // charge the budget a second time for a number the checkpoint carries.
  const seedEvaluation =
    resumeFrom === undefined
      ? await fullSweep(seedCandidate, "seed")
      : undefined;
  const seedScore =
    seedEvaluation === undefined
      ? (resumeFrom as MiproSnapshot).seedScore
      : requireMeasuredMean({ batch: seedEvaluation, phase: "seed" });
  if (seedEvaluation !== undefined) {
    fullEvaluations += 1;
  }

  // The seed is the baseline every later candidate is read against, and its
  // sweep is a full measurement like any other. A report that starts at the
  // first improvement has nothing to compare the improvement to. A resumed run
  // does not re-emit it: the run that swept it already did.
  if (seedEvaluation !== undefined) {
    emit({
      type: "candidateAccepted",
      trial: 0,
      candidateId: 0,
      candidate: seedCandidate,
      aggregateScore: seedScore,
      instanceScores: instanceRow(seedEvaluation),
      ...(trackBestOutputs ? { outputs: seedEvaluation.outputs } : {}),
    });
  }

  // The seed is index 0 of every menu, and its sweep is the most reliable
  // measurement the run will ever make. Registering it as a trial is how the
  // surrogate starts with a reference point instead of having to buy one, and
  // is what dspy does with `study.add_trial` before its loop begins.
  if (resumeFrom === undefined) {
    surrogateInput.push({ choices: menuSizes.map(() => 0), score: seedScore });
  }

  let best = (resumeFrom?.best as Candidate<K> | undefined) ?? seedCandidate;
  let bestScore = resumeFrom?.bestScore ?? seedScore;
  /** Absent on a resumed run until a sweep wins: outputs are not checkpointed. */
  let acceptedCandidates = 0;
  let bestOutputs = seedEvaluation?.outputs;
  // Minibatch readings per configuration, not one running bar. A configuration
  // drawn more than once is measured on a different batch each time, so the
  // mean of its readings is a steadier estimate than any single one — and it
  // is what decides which configuration earns the expensive sweep.
  const readings = new Map<string, number[]>(resumeFrom?.readings ?? []);
  const swept = new Set<string>(resumeFrom?.swept ?? []);

  function takeSnapshot(): MiproSnapshot {
    const cached = checkpointCache ? evaluationCache?.entries?.() : undefined;
    const samplerState = batchSampler.state?.();

    return {
      version: 1,
      fingerprint,
      menu: { ...menu },
      best,
      bestScore,
      seedScore,
      trial,
      fullEvaluations,
      reflectionCalls,
      bootstrapMetricCalls,
      metricCalls: budget.spent(),
      cacheHits: evaluator.cacheHits(),
      rngState: rng.state(),
      observations: [...observations],
      surrogateInput: surrogateInput.map((entry) => ({
        choices: [...entry.choices],
        score: entry.score,
      })),
      readings: [...readings].map(([key, values]) => [key, [...values]]),
      swept: [...swept],
      ...(samplerState === undefined ? {} : { sampler: samplerState }),
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

  /**
   * Full-evaluates the strongest configuration by mean minibatch reading that
   * has not been swept yet. Only a full sweep can move the incumbent, so a
   * lucky minibatch buys a candidate a closer look and nothing more.
   */
  async function sweepBestUnswept(): Promise<
    "swept" | "none" | "unaffordable" | "budgetExhausted" | "aborted"
  > {
    let bestKey: string | undefined;
    let bestMean = Number.NEGATIVE_INFINITY;

    for (const [key, values] of readings) {
      if (swept.has(key)) {
        continue;
      }
      const value = mean(values);
      if (value > bestMean) {
        bestMean = value;
        bestKey = key;
      }
    }

    if (bestKey === undefined) {
      return "none";
    }
    if (!budget.canAfford(validationSet.length)) {
      return "unaffordable";
    }

    const choices = bestKey.split(",").map(Number);
    const candidate = candidateFor(choices);
    swept.add(bestKey);

    let evaluation: Awaited<ReturnType<typeof fullSweep>>;
    try {
      evaluation = await fullSweep(candidate, "validation");
    } catch (err) {
      if (err instanceof BudgetExhausted) {
        return "budgetExhausted";
      }
      if (signal?.aborted) {
        return "aborted";
      }
      throw err;
    }
    fullEvaluations += 1;

    const score = measuredMean(evaluation);
    // The sweep measured the provider, not the configuration. Feeding its
    // zero to the surrogate would teach it to avoid a configuration on the
    // strength of an outage.
    if (score === undefined) {
      return "swept";
    }
    // A sweep is a far better measurement of this configuration than the
    // minibatch readings that earned it one, so the surrogate hears it too.
    // Without this a lucky reading keeps pulling proposals toward a
    // configuration the full validation set has already ruled out.
    surrogateInput.push({ choices, score });
    for (const observation of observations) {
      if (observation.choices.join(",") === bestKey) {
        observation.promoted = true;
        observation.score = score;
      }
    }

    if (score > bestScore) {
      best = candidate;
      bestScore = score;
      bestOutputs = evaluation.outputs;
      acceptedCandidates += 1;
      emit({
        type: "candidateAccepted",
        trial,
        candidateId: acceptedCandidates,
        candidate,
        aggregateScore: score,
        instanceScores: instanceRow(evaluation),
        ...(trackBestOutputs ? { outputs: evaluation.outputs } : {}),
      });
    }
    return "swept";
  }

  while (trial < maxTrials) {
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
    if (!budget.canAfford(minibatchSize)) {
      stopReason = "budgetExhausted";
      break;
    }
    // A reading is only worth buying if it could still be acted on. Once the
    // allowance cannot cover a sweep, no configuration can be promoted and the
    // incumbent is settled, so further trials would spend the rest of the
    // budget on measurements that can change nothing.
    if (!budget.canAfford(validationSet.length)) {
      stopReason = "budgetExhausted";
      break;
    }

    const choices = proposeConfiguration({
      observations: surrogateInput,
      menuSizes,
      ...(gamma === undefined ? {} : { gamma }),
      ...(surrogateSamples === undefined ? {} : { samples: surrogateSamples }),
      ...(startupTrials === undefined ? {} : { startupTrials }),
      ...(multivariate === undefined ? {} : { multivariate }),
      rng,
    });
    const candidate = candidateFor(choices);
    const batchIndices = batchSampler({ trainingSet, iteration: trial, rng });

    let minibatchScore: number | undefined;
    try {
      const evaluation = await evaluator.evaluate({
        candidate,
        batch: batchIndices.map((index) => trainingSet[index] as Datum),
        ids: batchIndices.map((index) => trainingIds[index] as string),
        split: "train",
        phase: "minibatch",
        candidateId: null,
        iteration: trial,
      });
      minibatchScore = measuredMean(evaluation);
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

    // Nothing was measured, so the trial is spent but says nothing. Recording
    // it would fit the surrogate to an outage and, worse, average an
    // infrastructure zero into this configuration's readings.
    if (minibatchScore === undefined) {
      trial += 1;
      continue;
    }

    surrogateInput.push({ choices, score: minibatchScore });

    const key = choices.join(",");
    readings.set(key, [...(readings.get(key) ?? []), minibatchScore]);

    const observation: MiproObservation = {
      trial,
      choices,
      minibatchScore,
      promoted: false,
    };
    observations.push(observation);
    emit({
      type: "trial",
      trial,
      choices,
      minibatchScore,
      promoted: false,
    });
    trial += 1;
    await checkpoint();

    if (trial % fullEvalInterval === 0) {
      const outcome = await sweepBestUnswept();
      if (outcome === "budgetExhausted" || outcome === "aborted") {
        stopReason = outcome;
        break;
      }
      // Readings are still affordable but sweeps are not, so no configuration
      // found from here could ever be promoted. The incumbent is settled;
      // buying more readings would only spend the rest of the allowance.
      if (outcome === "unaffordable") {
        stopReason = "budgetExhausted";
        break;
      }
    }
  }

  // The schedule can leave the strongest configuration unswept — the run ends
  // mid-interval, or its best reading arrived on the last trial. Without this
  // the winner would be whatever the cadence happened to land on.
  if (stopReason === "maxTrials" && !signal?.aborted) {
    await sweepBestUnswept();
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
          iteration: trial,
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
    bestCandidate: best,
    bestScore,
    usage: evaluator.usage(),
    seedScore,
    ...(trackBestOutputs ? { bestOutputs } : {}),
    ...(testScore === undefined
      ? {}
      : { testScore, testMetricCalls: evaluator.unchargedCalls() }),
    trials: trial,
    menu,
    observations,
    fullEvaluations,
    bootstrapMetricCalls,
    metricCalls: budget.spent(),
    reflectionCalls,
    cacheHits: evaluator.cacheHits(),
    stopReason,
  };
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

function assertConfig(config: MiproConfig): void {
  const positive: [string, number | undefined][] = [
    ["minibatchSize", config.minibatchSize],
    ["maxTrials", config.maxTrials],
    ["surrogateSamples", config.surrogateSamples],
    ["concurrency", config.concurrency],
  ];
  for (const [name, value] of positive) {
    if (value !== undefined && (!Number.isInteger(value) || value < 1)) {
      throw new Error(`${name} must be a positive integer, received ${value}`);
    }
  }

  const nonNegative: [string, number | undefined][] = [
    ["instructionsPerComponent", config.instructionsPerComponent],
    ["startupTrials", config.startupTrials],
    ["exemplars", config.exemplars],
  ];
  for (const [name, value] of nonNegative) {
    if (value !== undefined && (!Number.isInteger(value) || value < 0)) {
      throw new Error(
        `${name} must be a non-negative integer, received ${value}`,
      );
    }
  }

  if (
    config.gamma !== undefined &&
    (!Number.isFinite(config.gamma) || config.gamma <= 0 || config.gamma > 1)
  ) {
    throw new Error(`gamma must be within (0, 1], received ${config.gamma}`);
  }
  if (config.tips !== undefined && config.tips.length === 0) {
    throw new Error("tips must not be empty");
  }
}

function defaultInstanceId(args: { datum: unknown; index: number }): string {
  const hash = stableHash(args.datum);
  return hash === "" ? String(args.index) : hash;
}
