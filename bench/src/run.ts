import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { compare } from "textopt";
import type {
  Candidate,
  Comparison,
  OptimizerResult,
  TextModel,
} from "textopt";
import { BootstrapSearchOptimizer } from "textopt/bootstrap-search";
import {
  GepaOptimizer,
  lowerBoundEvaluationPolicy,
  pairedPermutationAcceptance,
} from "textopt/gepa";
import { MiproOptimizer } from "textopt/mipro";
import { OproOptimizer } from "textopt/opro";
import { RandomSearchOptimizer } from "textopt/random-search";
import { SimbaOptimizer } from "textopt/simba";
import type { BenchDatum, BenchTask } from "./tasks.js";
import {
  benchTasks,
  bestUnconditionalCandidate,
  MAX_MINIBATCH,
  createBenchAdviser,
  createBenchReflector,
  policyCandidate,
  redactObservations,
  renderBenchDemo,
  shotgunCandidate,
} from "./tasks.js";

type Entrant = (args: {
  seed: number;
}) => Promise<OptimizerResult<string, string, string>>;

/** One entrant's tunable settings, and how to build it from them. */
interface Contender {
  grid: Record<string, number>[];
  build: (args: {
    task: BenchTask;
    seed: number;
    config: Record<string, number>;
    /** Runs the entrant with the metric's diagnosis redacted from its prompt. */
    blind: boolean;
  }) => Promise<OptimizerResult<string, string, string>>;
}

interface Reference {
  name: string;
  score: number;
  note: string;
}

/** One entrant's score with the metric's diagnosis redacted from its prompt. */
interface Blind {
  score: number;
  /** Proposals it actually bought, which is what the cap is trying to match. */
  reflectionCalls: number;
}

/** Twenty, not ten: at ten the closest comparison here cannot clear p = 0.05. */
const SEEDS = Array.from({ length: 20 }, (_, index) => index);

/**
 * Seeds every entrant is tuned on, disjoint from the ones it is scored on.
 * Choosing a setting is a fit like any other, so a setting chosen on the seeds
 * a table reports would be a number tuned on its own answer — which is exactly
 * what an earlier version of this file did for one entrant and not the others.
 */
const TUNING_SEEDS = Array.from({ length: 10 }, (_, index) => 100 + index);

/**
 * Proposals every entrant is held to when its prompt is redacted.
 *
 * Blind, a proposal is a draw from a fixed pool rather than an induction, so an
 * entrant that buys more draws scores better without searching better. Nine is
 * the fewest any entrant makes when it is left alone, so it is the most that
 * can be asked of all of them — and comparing blind rows is only about search
 * once the draws are the same. `randomSearch` is the exception and is reported
 * at the count it takes.
 */
const BLIND_REFLECTION_CALLS = 9;

const OUTPUT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../results/latest.json",
);

await main();

async function main(): Promise<void> {
  const results: {
    task: string;
    comparison: Comparison;
    tuning: Record<string, Record<string, number>>;
    references: Reference[];
    blind: Record<string, Blind>;
  }[] = [];

  for (const task of benchTasks()) {
    const tuning = await tune(task);
    const comparison = await compare({
      seeds: SEEDS,
      entrants: entrants({ task, tuning }),
    });
    const references = await referenceRows({ task });
    const blind = await blindRows({ task, tuning });

    results.push({ task: task.name, comparison, tuning, references, blind });
    process.stdout.write(`${task.name}: winner ${comparison.winner}\n`);
  }

  report(results);

  mkdirSync(dirname(OUTPUT), { recursive: true });
  writeFileSync(
    OUTPUT,
    `${JSON.stringify(
      {
        seeds: SEEDS.length,
        tuningSeeds: TUNING_SEEDS.length,
        node: process.version,
        results,
      },
      null,
      2,
    )}\n`,
  );
  process.stdout.write(`\nWrote ${OUTPUT}\n`);
}

/**
 * Every entrant, and the settings each one is allowed to be tuned over.
 *
 * The grids are comparable in size on purpose. An entrant swept over nine
 * settings against one pinned at its default is not a comparison of searches,
 * and reporting the swept one at its best is how a benchmark quietly becomes an
 * argument for whichever entrant its author spent the afternoon on.
 */
function contenders(): Record<string, Contender> {
  return {
    gepa: {
      grid: minibatchGrid(),
      build: ({ task, seed, config, blind }) =>
        new GepaOptimizer({
          seed,
          minibatchSize: config.minibatchSize as number,
          reflection: { maxCalls: proposalBudget({ task, blind }) },
        }).optimize({
          ...shared(task),
          reflect: proposer({ seed, blind }),
        }),
    },

    // The same search, judged differently: accept only when the minibatch
    // improvement survives a sign-flip test, and pick the winner by a lower
    // confidence bound rather than by its mean.
    gepaVarianceAware: {
      grid: minibatchGrid(),
      build: ({ task, seed, config, blind }) =>
        new GepaOptimizer({
          seed,
          minibatchSize: config.minibatchSize as number,
          reflection: { maxCalls: proposalBudget({ task, blind }) },
          acceptance: pairedPermutationAcceptance(),
        }).optimize({
          ...shared(task),
          reflect: proposer({ seed, blind }),
          valEvaluationPolicy: lowerBoundEvaluationPolicy(),
        }),
    },

    opro: {
      grid: [3, 6, MAX_MINIBATCH].map((scoringSetSize) => ({ scoringSetSize })),
      build: ({ task, seed, config, blind }) =>
        new OproOptimizer({
          seed,
          maxReflectionCalls: proposalBudget({ task, blind }),
          scoringSetSize: config.scoringSetSize as number,
        }).optimize({
          ...shared(task),
          reflect: proposer({ seed, blind }),
        }),
    },

    mipro: {
      grid: minibatchGrid(),
      build: ({ task, seed, config, blind }) =>
        new MiproOptimizer({
          seed,
          minibatchSize: config.minibatchSize as number,
          instructionsPerComponent: 8,
          maxTrials: task.maxReflectionCalls,
        }).optimize({
          ...shared(task),
          reflect: proposer({ seed, blind }),
        }),
    },

    // SIMBA is given the advice-shaped proposer instead: it appends per-component
    // rules rather than replacing an instruction, so it needs a model that
    // answers in that shape. Same induction threshold and same draw pool.
    // Which mutations to draw from is a setting like any other, so the tuner
    // chooses it. SIMBA has two, and they are worth different amounts here:
    // harvesting a rollout pays only where a rollout carries something its
    // prompt did not, and everywhere else it spends a draw a rule would have
    // used. Fixing it at the author's guess is the thing this bench is not
    // allowed to do.
    simba: {
      grid: minibatchGrid().flatMap((cell) =>
        [0, 1].map((demos) => ({ ...cell, demos })),
      ),
      build: ({ task, seed, config, blind }) =>
        new SimbaOptimizer({
          seed,
          minibatchSize: config.minibatchSize as number,
          candidates: 2,
          maxSteps: 40,
          maxReflectionCalls: proposalBudget({ task, blind }),
          strategies:
            config.demos === 1
              ? (["appendDemo", "appendRule"] as const)
              : (["appendRule"] as const),
        }).optimize({
          ...shared(task),
          reflect: adviser({ seed, blind }),
          demoComponents: task.demoComponents,
          renderDemo: renderBenchDemo,
        }),
    },

    // Random search takes no seed of its own: its variants come from `reflect`
    // alone. The seed still reaches it through the proposal model, which is
    // where every one of its draws comes from, so its spread across seeds is
    // the spread of the pool it happened to walk.
    randomSearch: {
      grid: [2, 4, 6, 8].map((variants) => ({ variants })),
      build: ({ task, seed, config, blind }) =>
        new RandomSearchOptimizer({
          variants: config.variants as number,
        }).optimize({
          ...shared(task),
          reflect: proposer({ seed, blind }),
        }),
    },

    // The one entrant that calls no proposal model: it harvests the rollouts
    // the metric rewarded and searches over which of them to show. On the three
    // tasks whose answer is a pure function of the candidate there is nothing
    // in a rollout to harvest, and the row it scores there is what that looks
    // like — the point of running it on all four rather than only where it wins.
    bootstrapSearch: {
      grid: [2, 4, 6, 8].map((maxDemos) => ({ maxDemos })),
      build: ({ task, seed, config }) =>
        new BootstrapSearchOptimizer({
          seed,
          maxDemos: config.maxDemos as number,
        }).optimize({
          ...shared(task),
          demoComponents: task.demoComponents,
          renderDemo: renderBenchDemo,
        }),
    },
  };
}

/**
 * The setting each entrant is scored at, chosen on the tuning seeds and on the
 * validation score alone. Never on `testScore`: the held-out number only means
 * something if nothing in the run — including the choice of how to run it —
 * was fitted to it.
 */
async function tune(
  task: BenchTask,
): Promise<Record<string, Record<string, number>>> {
  const chosen: Record<string, Record<string, number>> = {};

  for (const [name, contender] of Object.entries(contenders())) {
    let best: { config: Record<string, number>; score: number } | undefined;

    for (const config of contender.grid) {
      const scores = await Promise.all(
        TUNING_SEEDS.map(async (seed) => {
          const result = await contender.build({
            task,
            seed,
            config,
            blind: false,
          });
          return result.bestScore;
        }),
      );
      const score = mean(scores);
      if (best === undefined || score > best.score) {
        best = { config, score };
      }
    }
    chosen[name] = best?.config ?? {};
  }
  return chosen;
}

function entrants(args: {
  task: BenchTask;
  tuning: Record<string, Record<string, number>>;
}): Record<string, Entrant> {
  const { task, tuning } = args;

  return Object.fromEntries(
    Object.entries(contenders()).map(([name, contender]) => [
      name,
      ({ seed }: { seed: number }) =>
        contender.build({
          task,
          seed,
          config: tuning[name] ?? {},
          blind: false,
        }),
    ]),
  );
}

/**
 * Fixed candidates scored on the same held-out set, so a reader can place the
 * entrants between something that solved the task and something that did not
 * search at all.
 *
 * The floors are the load-bearing ones. A table of optimizer scores with no
 * floor cannot answer the only question that matters — whether the search found
 * anything a candidate written without one would have found too.
 */
async function referenceRows(args: { task: BenchTask }): Promise<Reference[]> {
  const { task } = args;

  return [
    {
      name: "policy",
      score: await scoreOnTest({ task, candidate: policyCandidate(task) }),
      note: "the rules the search is looking for; the ceiling",
    },
    {
      name: "shotgun",
      score: await scoreOnTest({ task, candidate: shotgunCandidate(task) }),
      note: "every action on every ticket; no search, no conditions",
    },
    {
      name: "bestFixed",
      score: await scoreOnTest({
        task,
        candidate: bestUnconditionalCandidate(task),
      }),
      note: "the best answer that ignores the ticket, chosen off the held-out set",
    },
    {
      name: "zeroShot",
      score: await scoreOnTest({ task, candidate: task.seedCandidate }),
      note: "the seed candidate, unoptimised",
    },
  ];
}

/**
 * Every entrant again, with the metric's per-instance diagnosis stripped out of
 * its prompt and its proposals capped at the same number.
 *
 * Published because the alternative is asserting that a reflective search beats
 * a score-only one on the strength of its search, when part of what separates
 * them is that one is handed evidence the other never sees. Run for one entrant
 * this answers only what that entrant loses; run for all of them at a matched
 * proposal count, the spread across this column is what search is worth once no
 * one can read the diagnosis, and the gap to the column above it is what the
 * channel is worth.
 */
async function blindRows(args: {
  task: BenchTask;
  tuning: Record<string, Record<string, number>>;
}): Promise<Record<string, Blind>> {
  const { task, tuning } = args;
  const rows: Record<string, Blind> = {};

  for (const [name, contender] of Object.entries(contenders())) {
    const runs = await Promise.all(
      SEEDS.map((seed) =>
        contender.build({
          task,
          seed,
          config: tuning[name] ?? {},
          blind: true,
        }),
      ),
    );
    rows[name] = {
      score: mean(runs.map((run) => run.testScore ?? 0)),
      reflectionCalls: mean(runs.map((run) => run.reflectionCalls ?? 0)),
    };
  }
  return rows;
}

/**
 * The proposal model with the metric's diagnosis stripped out of the prompt,
 * leaving it the blind draw a score-only search already gets.
 *
 * Published because the alternative is asserting that a reflective search beats
 * a score-only one on the strength of its search, when part of what separates
 * them is that one is handed evidence the other never sees. The gap between
 * this row and the `gepa` row is the part of the margin that is the channel.
 */
function withoutFeedback(seed: number): TextModel {
  const inner = createBenchReflector({ seed });
  return ({ prompt, signal }) =>
    inner({ prompt: redactObservations(prompt), signal });
}

function withoutAdviceFeedback(seed: number): TextModel {
  const inner = createBenchAdviser({ seed });
  return ({ prompt, signal }) =>
    inner({ prompt: redactObservations(prompt), signal });
}

function proposer(args: { seed: number; blind: boolean }): TextModel {
  const { seed, blind } = args;
  return blind ? withoutFeedback(seed) : createBenchReflector({ seed });
}

/** SIMBA's proposer answers in the advice shape, blind or not. */
function adviser(args: { seed: number; blind: boolean }): TextModel {
  const { seed, blind } = args;
  return blind ? withoutAdviceFeedback(seed) : createBenchAdviser({ seed });
}

function proposalBudget(args: { task: BenchTask; blind: boolean }): number {
  const { task, blind } = args;
  return blind ? BLIND_REFLECTION_CALLS : task.maxReflectionCalls;
}

async function scoreOnTest(args: {
  task: BenchTask;
  candidate: Candidate;
}): Promise<number> {
  const { task, candidate } = args;
  const evaluated = await task.adapter.evaluate({
    batch: task.testSet,
    candidate,
    captureTraces: false,
    run: { iteration: 0, phase: "test", split: "test", candidateId: null },
  });

  return mean(evaluated.scores);
}

/** The minibatch grid the four minibatch-shaped entrants share. */
function minibatchGrid(): Record<string, number>[] {
  return [3, 6, MAX_MINIBATCH].map((minibatchSize) => ({ minibatchSize }));
}

/**
 * The parts of a task every entrant is given identically. The proposal model is
 * not among them: it is built at each call site instead, because its rotation is
 * state and one shared across runs would make each result depend on the runs
 * before it — and because the demonstration search takes none at all.
 */
function shared(task: BenchTask): {
  seedCandidate: BenchTask["seedCandidate"];
  trainingSet: readonly BenchDatum[];
  validationSet: readonly BenchDatum[];
  testSet: readonly BenchDatum[];
  maxMetricCalls: number;
  adapter: BenchTask["adapter"];
} {
  return {
    seedCandidate: task.seedCandidate,
    trainingSet: task.trainingSet,
    validationSet: task.validationSet,
    testSet: task.testSet,
    maxMetricCalls: task.maxMetricCalls,
    adapter: task.adapter,
  };
}

function report(
  results: readonly {
    task: string;
    comparison: Comparison;
    tuning: Record<string, Record<string, number>>;
    references: Reference[];
    blind: Record<string, Blind>;
  }[],
): void {
  const header = [
    "task",
    "entrant",
    "test",
    "sd",
    "min",
    "max",
    "rollouts",
    "cached",
    "tuned",
    "p",
    "p(holm)",
  ];
  const rows = results.flatMap(({ task, comparison, tuning }) =>
    comparison.summaries.map((summary) => [
      task,
      summary.entrant,
      summary.meanScore.toFixed(3),
      summary.sdScore.toFixed(3),
      summary.minScore.toFixed(3),
      summary.maxScore.toFixed(3),
      summary.meanMetricCalls.toFixed(0),
      summary.meanCacheHits.toFixed(0),
      settings(tuning[summary.entrant] ?? {}),
      pValue({ summary, comparison }),
      summary.pValueVsWinnerHolm === undefined
        ? ""
        : summary.pValueVsWinnerHolm.toFixed(3),
    ]),
  );

  write([header, ...rows]);

  write([
    ["task", "reference", "test", "what it is"],
    ...results.flatMap(({ task, references }) =>
      references.map((reference) => [
        task,
        reference.name,
        reference.score.toFixed(3),
        reference.note,
      ]),
    ),
  ]);

  write([
    ["task", "entrant", "blind", "sighted", "lost", "proposals"],
    ...results.flatMap(({ task, comparison, blind }) =>
      comparison.summaries.map((summary) => {
        const row = blind[summary.entrant];
        return [
          task,
          summary.entrant,
          (row?.score ?? 0).toFixed(3),
          summary.meanScore.toFixed(3),
          (summary.meanScore - (row?.score ?? 0)).toFixed(3),
          (row?.reflectionCalls ?? 0).toFixed(0),
        ];
      }),
    ),
  ]);
}

/**
 * The winner's row says so; a row `compare` withheld a p-value from says why.
 * Printing a blank there would read as a missing number rather than as the
 * absence of anything to test.
 */
function pValue(args: {
  summary: Comparison["summaries"][number];
  comparison: Comparison;
}): string {
  const { summary, comparison } = args;

  if (summary.entrant === comparison.winner) {
    return "winner";
  }
  return summary.pValueVsWinner === undefined
    ? "no spread"
    : summary.pValueVsWinner.toFixed(3);
}

function settings(config: Record<string, number>): string {
  return Object.entries(config)
    .map(([name, value]) => `${name}=${value}`)
    .join(",");
}

function write(rows: readonly string[][]): void {
  const widths = (rows[0] ?? []).map((_, column) =>
    Math.max(...rows.map((row) => (row[column] ?? "").length)),
  );

  process.stdout.write("\n");
  for (const row of rows) {
    process.stdout.write(
      `${row.map((cell, column) => cell.padEnd(widths[column] ?? 0)).join("  ")}\n`,
    );
  }
}

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}
