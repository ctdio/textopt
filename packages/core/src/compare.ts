import { mapWithConcurrency } from "./concurrency.js";
import { holmAdjust, signFlipPValue } from "./math.js";
import type { OptimizerResult } from "./optimizer.js";

export interface ComparisonRun {
  entrant: string;
  seed: number;
  /** What the entrants are ranked on: the held-out score when there is one. */
  score: number;
  bestScore: number;
  testScore?: number;
  metricCalls: number;
  /** Rollouts this run got from the cache instead of paying for. */
  cacheHits: number;
  /** Calls to a proposal or reflection model, outside the metric budget. */
  reflectionCalls: number;
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
  meanCacheHits: number;
  meanReflectionCalls: number;
  /**
   * How many distinct values this entrant's score took across its seeds.
   * 1 means every seed landed on the same number — the seed changed nothing
   * about the outcome, whatever the search did internally with it.
   */
  distinctScores: number;
  /**
   * How often the winner's margin over this entrant would arise if the two
   * were equally good and each seed's outcome were a coin flip. Absent for
   * the winner itself, and also absent when every seed produced the exact
   * same margin: a sign-flip test over n seeds is answering a question about
   * n independent trials, and identical margins mean the seed never actually
   * put that to the test — there was one realization, repeated. Reporting a
   * p-value there would state a precision (as fine as 2^-n) that the run
   * never earned, so it is withheld rather than printed misleadingly small.
   */
  pValueVsWinner?: number;
  /**
   * `pValueVsWinner` after Holm-Bonferroni step-down across the other
   * entrants in this same `compare()` call — the family the raw p-value
   * would otherwise be read against in isolation. Absent wherever the raw
   * p-value is: a withheld comparison has nothing to adjust, but it still
   * occupies a slot in the family the other comparisons are corrected for.
   */
  pValueVsWinnerHolm?: number;
}

export interface Comparison {
  /** Highest mean score. Read `pValueVsWinner` before believing it. */
  winner: string;
  summaries: ComparisonSummary[];
  runs: ComparisonRun[];
}

const EXACT_LIMIT = 20;
/** Well above float subtraction noise (~1e-16), well below a real margin. */
const DEGENERACY_TOLERANCE = 1e-9;

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
        cacheHits: result.cacheHits,
        reflectionCalls: result.reflectionCalls ?? 0,
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

  const rawPValues = summaries.map((summary) =>
    summary.entrant === winner.entrant
      ? undefined
      : margin({
          winner: winner.entrant,
          entrant: summary.entrant,
          runs,
          seeds,
        }),
  );
  const holmAdjusted = holmAdjustSparse({
    pValues: rawPValues,
    familySize: names.length - 1,
  });

  return {
    winner: winner.entrant,
    summaries: summaries.map((summary, index) => ({
      ...summary,
      ...(rawPValues[index] === undefined
        ? {}
        : {
            pValueVsWinner: rawPValues[index],
            pValueVsWinnerHolm: holmAdjusted[index],
          }),
    })),
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
    meanCacheHits: mean(runs.map((run) => run.cacheHits)),
    meanReflectionCalls: mean(runs.map((run) => run.reflectionCalls)),
    distinctScores: new Set(scores).size,
  };
}

/**
 * Paired across seeds rather than pooled: the same seed puts both entrants on
 * the same sampling order, so the difference at a seed is a comparison and the
 * spread between seeds is not.
 *
 * Returns `undefined` when every seed produced the exact same nonzero margin.
 * That is not evidence of an n-seed-strong result — it is one realization the
 * seed never varied, and a sign-flip p-value would report a precision from n
 * independent trials that never happened. A margin of exactly zero every seed
 * is not this case: `signFlipPValue` already reports that honestly as 1, no
 * significance claimed either way, which is not a fabricated number.
 *
 * "Exact same" is judged within `DEGENERACY_TOLERANCE`, not `===`: subtracting
 * two scores that are equal in substance can still land a few ULPs apart
 * (0.95 - 0.55 and 0.9 - 0.5 differ at the 16th digit), and treating that as
 * n real trials would be the same fabrication this check exists to prevent.
 */
function margin(args: {
  winner: string;
  entrant: string;
  runs: readonly ComparisonRun[];
  seeds: readonly number[];
}): number | undefined {
  const { winner, entrant, runs, seeds } = args;

  const differences = seeds.map((seed) => {
    const won = scoreOf({ runs, entrant: winner, seed });
    const lost = scoreOf({ runs, entrant, seed });
    return won - lost;
  });

  const [first] = differences;
  if (first !== undefined) {
    const spread = differences.reduce(
      (widest, difference) => Math.max(widest, Math.abs(difference - first)),
      0,
    );
    if (
      spread < DEGENERACY_TOLERANCE &&
      Math.abs(first) > DEGENERACY_TOLERANCE
    ) {
      return undefined;
    }
  }

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

/**
 * `holmAdjust` over the raw p-values that exist, skipping the slots a
 * withheld comparison left `undefined` — those still count toward
 * `familySize`, they just have nothing of their own to adjust.
 */
function holmAdjustSparse(args: {
  pValues: readonly (number | undefined)[];
  familySize: number;
}): (number | undefined)[] {
  const { pValues, familySize } = args;

  const present = pValues
    .map((pValue, index) => ({ pValue, index }))
    .filter(
      (entry): entry is { pValue: number; index: number } =>
        entry.pValue !== undefined,
    );

  const adjusted = holmAdjust({
    pValues: present.map((entry) => entry.pValue),
    familySize,
  });

  const result = new Array<number | undefined>(pValues.length).fill(undefined);
  present.forEach((entry, rank) => {
    result[entry.index] = adjusted[rank];
  });

  return result;
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
