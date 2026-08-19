import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { compare } from "textopt";
import type { Comparison, OptimizerResult } from "textopt";
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
  createBenchAdviser,
  createBenchReflector,
  renderBenchDemo,
} from "./tasks.js";

type Entrant = (args: {
  seed: number;
}) => Promise<OptimizerResult<string, string, string>>;

/** Twenty, not ten: at ten the closest comparison here cannot clear p = 0.05. */
const SEEDS = Array.from({ length: 20 }, (_, index) => index);

/**
 * Minibatch both GEPA variants screen on. Larger than the default 3 because a
 * sign-flip test over three instances cannot reach a p-value below 0.125 — at
 * the default the variance-aware variant would be measuring its own floor.
 */
const MINIBATCH = 6;

const OUTPUT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../results/latest.json",
);

await main();

async function main(): Promise<void> {
  const results: { task: string; comparison: Comparison }[] = [];

  for (const task of benchTasks()) {
    const comparison = await compare({
      seeds: SEEDS,
      entrants: entrants(task),
    });
    results.push({ task: task.name, comparison });
    process.stdout.write(`${task.name}: winner ${comparison.winner}\n`);
  }

  report(results);

  mkdirSync(dirname(OUTPUT), { recursive: true });
  writeFileSync(
    OUTPUT,
    `${JSON.stringify({ seeds: SEEDS.length, node: process.version, results }, null, 2)}\n`,
  );
  process.stdout.write(`\nWrote ${OUTPUT}\n`);
}

function entrants(task: BenchTask): Record<string, Entrant> {
  return {
    gepa: ({ seed }) =>
      new GepaOptimizer({
        seed,
        minibatchSize: MINIBATCH,
        reflection: { maxCalls: task.maxReflectionCalls },
      }).optimize({ ...shared(task), reflect: createBenchReflector() }),

    // The same search, judged differently: accept only when the minibatch
    // improvement survives a sign-flip test, and pick the winner by a lower
    // confidence bound rather than by its mean.
    gepaVarianceAware: ({ seed }) =>
      new GepaOptimizer({
        seed,
        minibatchSize: MINIBATCH,
        reflection: { maxCalls: task.maxReflectionCalls },
        acceptance: pairedPermutationAcceptance(),
      }).optimize({
        ...shared(task),
        reflect: createBenchReflector(),
        valEvaluationPolicy: lowerBoundEvaluationPolicy(),
      }),

    opro: ({ seed }) =>
      new OproOptimizer({
        seed,
        maxReflectionCalls: task.maxReflectionCalls,
        scoringSetSize: MINIBATCH,
      }).optimize({ ...shared(task), reflect: createBenchReflector() }),

    mipro: ({ seed }) =>
      new MiproOptimizer({
        seed,
        minibatchSize: MINIBATCH,
        instructionsPerComponent: 8,
        maxTrials: task.maxReflectionCalls,
      }).optimize({ ...shared(task), reflect: createBenchReflector() }),

    // SIMBA is given the advice-shaped proposer instead: it appends per-component
    // rules rather than replacing an instruction, so it needs a model that
    // answers in that shape. Same absorption rate and same draw pool.
    //
    // Two programs over four instances rather than the shared minibatch: a
    // trajectory sample is measurement, not search, and at these budgets every
    // wider setting spends more on measurement than it wins back. Swept over
    // candidates 2-6 and minibatches 4-8; this is the best of them on all three
    // tasks, so the number below is SIMBA at its best here rather than at a
    // default that happens to flatter the others.
    simba: ({ seed }) =>
      new SimbaOptimizer({
        seed,
        minibatchSize: 4,
        candidates: 2,
        maxSteps: 40,
        maxReflectionCalls: task.maxReflectionCalls,
      }).optimize({ ...shared(task), reflect: createBenchAdviser() }),

    // Random search takes no seed: its variants come from `reflect` alone, so
    // against a deterministic model every seed produces the same run. Its
    // reported spread is zero by construction, not by luck.
    randomSearch: () =>
      new RandomSearchOptimizer({}).optimize({
        ...shared(task),
        reflect: createBenchReflector(),
      }),

    // The one entrant that calls no proposal model: it harvests the rollouts
    // the metric rewarded and searches over which of them to show. On the three
    // tasks whose output is a pure function of the candidate there is nothing
    // in a rollout to harvest, and the row it scores there is what that looks
    // like — the point of running it on all four rather than only where it wins.
    bootstrapSearch: ({ seed }) =>
      new BootstrapSearchOptimizer({ seed, maxDemos: 4 }).optimize({
        ...shared(task),
        demoComponents: task.demoComponents,
        renderDemo: renderBenchDemo,
      }),
  };
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
  results: readonly { task: string; comparison: Comparison }[],
): void {
  const header = [
    "task",
    "entrant",
    "test",
    "sd",
    "min",
    "max",
    "rollouts",
    "p",
  ];
  const rows = results.flatMap(({ task, comparison }) =>
    comparison.summaries.map((summary) => [
      task,
      summary.entrant,
      summary.meanScore.toFixed(3),
      summary.sdScore.toFixed(3),
      summary.minScore.toFixed(3),
      summary.maxScore.toFixed(3),
      summary.meanMetricCalls.toFixed(0),
      summary.pValueVsWinner === undefined
        ? "winner"
        : summary.pValueVsWinner.toFixed(3),
    ]),
  );
  const widths = header.map((_, column) =>
    Math.max(...[header, ...rows].map((row) => (row[column] ?? "").length)),
  );

  process.stdout.write("\n");
  for (const row of [header, ...rows]) {
    process.stdout.write(
      `${row.map((cell, column) => cell.padEnd(widths[column] ?? 0)).join("  ")}\n`,
    );
  }
}
