import { mapWithConcurrency } from "./concurrency.js";
import { signFlipPValue } from "./math.js";
import type { OptimizerResult } from "./optimizer.js";

export interface ComparisonRun {
  entrant: string;
  seed: number;
  /** What the entrants are ranked on: the held-out score when there is one. */
  score: number;
  bestScore: number;
  testScore?: number;
  metricCalls: number;
  costUsd: number;
  stopReason: string;
}

export interface ComparisonSummary {
  entrant: string;
  runs: number;
  meanScore: number;
  sdScore: number;
  minScore: number;
  maxScore: number;
  meanMetricCalls: number;
  meanCostUsd: number;
  /**
   * How often the winner's margin over this entrant would arise if the two
   * were equally good and each seed's outcome were a coin flip. Absent for the
   * winner itself.
   */
  pValueVsWinner?: number;
}

export interface Comparison {
  /** Highest mean score. Read `pValueVsWinner` before believing it. */
  winner: string;
  summaries: ComparisonSummary[];
  runs: ComparisonRun[];
}

const EXACT_LIMIT = 16;

/**
 * Run several optimizers over the same seeds and report which one actually won.
 *
 * Two things make this worth a helper rather than a for-loop. The first is that
 * it ranks on `testScore` where a run reports one: the validation score is the
 * number the search selected against for its whole run, so an entrant that
 * overfits looks strongest on exactly the number it fitted. The second is that
 * a difference in means over a handful of seeds is usually noise, and the
 * paired sign-flip p-value against the winner is what says whether it is.
 *
 * Entrants are functions of a seed rather than optimizer instances, because the
 * seed is constructor config and every optimizer here is deterministic given
 * one — comparing two entrants at a single seed compares two anecdotes. It also
 * leaves the task where the caller builds it, which is the only place the
 * optimizer-specific parts of it (`reflect`, `cache`, `reporters`) are known.
 */
export async function compare<K extends string, Output = unknown>(args: {
  entrants: Record<
    string,
    (args: { seed: number }) => Promise<OptimizerResult<K, string, Output>>
  >;
  seeds: readonly number[];
  /** Runs in flight at once. Default 1. */
  concurrency?: number;
}): Promise<Comparison> {
  const { entrants, seeds, concurrency = 1 } = args;
  const names = Object.keys(entrants);

  if (names.length === 0) {
    throw new Error("compare requires at least one entrant");
  }
  if (seeds.length === 0) {
    throw new Error("compare requires at least one seed");
  }

  const grid = names.flatMap((entrant) =>
    seeds.map((seed) => ({ entrant, seed })),
  );

  const runs = await mapWithConcurrency({
    items: grid,
    limit: concurrency,
    task: async ({ entrant, seed }) => {
      const result = await (entrants[entrant] as (typeof entrants)[string])({
        seed,
      });

      return {
        entrant,
        seed,
        score: result.testScore ?? result.bestScore,
        bestScore: result.bestScore,
        ...(result.testScore === undefined
          ? {}
          : { testScore: result.testScore }),
        metricCalls: result.metricCalls,
        costUsd: result.usage.costUsd,
        stopReason: result.stopReason,
      } satisfies ComparisonRun;
    },
  });

  const summaries = names.map((entrant) =>
    summarize({ entrant, runs: runs.filter((run) => run.entrant === entrant) }),
  );
  const winner = summaries.reduce((best, summary) =>
    summary.meanScore > best.meanScore ? summary : best,
  );

  return {
    winner: winner.entrant,
    summaries: summaries.map((summary) =>
      summary.entrant === winner.entrant
        ? summary
        : {
            ...summary,
            pValueVsWinner: margin({
              winner: winner.entrant,
              entrant: summary.entrant,
              runs,
              seeds,
            }),
          },
    ),
    runs,
  };
}

function summarize(args: {
  entrant: string;
  runs: readonly ComparisonRun[];
}): ComparisonSummary {
  const { entrant, runs } = args;
  const scores = runs.map((run) => run.score);

  return {
    entrant,
    runs: runs.length,
    meanScore: mean(scores),
    sdScore: standardDeviation(scores),
    minScore: Math.min(...scores),
    maxScore: Math.max(...scores),
    meanMetricCalls: mean(runs.map((run) => run.metricCalls)),
    meanCostUsd: mean(runs.map((run) => run.costUsd)),
  };
}

/**
 * Paired across seeds rather than pooled: the same seed puts both entrants on
 * the same sampling order, so the difference at a seed is a comparison and the
 * spread between seeds is not.
 */
function margin(args: {
  winner: string;
  entrant: string;
  runs: readonly ComparisonRun[];
  seeds: readonly number[];
}): number {
  const { winner, entrant, runs, seeds } = args;

  const differences = seeds.map((seed) => {
    const won = scoreOf({ runs, entrant: winner, seed });
    const lost = scoreOf({ runs, entrant, seed });
    return won - lost;
  });

  return signFlipPValue({
    differences,
    observed: differences.reduce((total, value) => total + value, 0),
    maxExact: EXACT_LIMIT,
  });
}

function scoreOf(args: {
  runs: readonly ComparisonRun[];
  entrant: string;
  seed: number;
}): number {
  const { runs, entrant, seed } = args;
  const run = runs.find(
    (candidate) => candidate.entrant === entrant && candidate.seed === seed,
  );

  return run?.score ?? 0;
}

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function standardDeviation(values: readonly number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const average = mean(values);
  const variance =
    values.reduce((total, value) => total + (value - average) ** 2, 0) /
    (values.length - 1);

  return Math.sqrt(variance);
}
