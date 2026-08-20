import { describe, expect, test } from "vitest";
import { compare } from "./compare.js";
import type { OptimizerResult } from "./optimizer.js";

const CANDIDATE = { instruction: "answer" };

describe("compare", () => {
  test("summarizes each entrant over the seeds it was run on", async () => {
    const comparison = await compare({
      seeds: [0, 1, 2],
      entrants: {
        steady: scoring({ bestScores: [0.5, 0.5, 0.5] }),
        swingy: scoring({ bestScores: [0.2, 0.5, 0.8] }),
      },
    });

    const steady = summaryFor(comparison, "steady");
    const swingy = summaryFor(comparison, "swingy");

    expect(steady.runs).toBe(3);
    expect(steady.meanScore).toBeCloseTo(0.5);
    expect(steady.sdScore).toBeCloseTo(0);
    expect(swingy.meanScore).toBeCloseTo(0.5);
    expect(swingy.minScore).toBeCloseTo(0.2);
    expect(swingy.maxScore).toBeCloseTo(0.8);
  });

  test("names the highest mean as the winner", async () => {
    const comparison = await compare({
      seeds: [0, 1],
      entrants: {
        weak: scoring({ bestScores: [0.1, 0.2] }),
        strong: scoring({ bestScores: [0.7, 0.8] }),
      },
    });

    expect(comparison.winner).toBe("strong");
  });

  test("compares on the held-out score when a run reports one", async () => {
    // The validation score is the one the search selected against, so an
    // entrant that overfits it looks best on exactly the number it fitted.
    const comparison = await compare({
      seeds: [0],
      entrants: {
        overfitter: scoring({ bestScores: [0.9], testScores: [0.1] }),
        generalizer: scoring({ bestScores: [0.6], testScores: [0.6] }),
      },
    });

    expect(comparison.winner).toBe("generalizer");
  });

  test("scores how likely the winner's margin is under a coin flip", async () => {
    // Not [0.1, 0.2, 0.3, 0.4, 0.5]: that makes every paired difference
    // exactly 0.5, which is the degenerate case a real p-value can't be
    // computed for (see "withholds..." below) — 0.15 keeps the margin
    // same-signed but not identical across seeds, so the exact answer is
    // still the unique all-positive sign assignment, still 1/32.
    const comparison = await compare({
      seeds: [0, 1, 2, 3, 4],
      entrants: {
        better: scoring({ bestScores: [0.6, 0.7, 0.8, 0.9, 1] }),
        worse: scoring({ bestScores: [0.1, 0.15, 0.3, 0.4, 0.5] }),
      },
    });

    expect(summaryFor(comparison, "worse").pValueVsWinner).toBeCloseTo(1 / 32);
  });

  test("reports no significance for the winner against itself", async () => {
    const comparison = await compare({
      seeds: [0],
      entrants: { only: scoring({ bestScores: [0.5] }) },
    });

    expect(summaryFor(comparison, "only").pValueVsWinner).toBeUndefined();
  });

  test("calls nothing significant when two entrants tie on every seed", async () => {
    const comparison = await compare({
      seeds: [0, 1, 2],
      entrants: {
        left: scoring({ bestScores: [0.5, 0.5, 0.5] }),
        right: scoring({ bestScores: [0.5, 0.5, 0.5] }),
      },
    });

    expect(summaryFor(comparison, "right").pValueVsWinner).toBe(1);
  });

  test("withholds the p-value when the seed never moves the margin", async () => {
    // Both entrants swing across seeds, but the winner's margin over the
    // loser is 0.4 every time — the seed changed both scores together and
    // never once told us whether the margin could have gone the other way.
    // A sign-flip p-value here would report a precision (1 in 2^n) that
    // three seeds never earned.
    const comparison = await compare({
      seeds: [0, 1, 2],
      entrants: {
        winner: scoring({ bestScores: [0.9, 1.0, 0.95] }),
        steady: scoring({ bestScores: [0.5, 0.6, 0.55] }),
      },
    });

    const steady = summaryFor(comparison, "steady");
    expect(steady.pValueVsWinner).toBeUndefined();
    expect(steady.pValueVsWinnerHolm).toBeUndefined();
    // distinctScores is about the entrant's own scores, not the margin: the
    // withholding above must not be confused with a run that never varies.
    expect(steady.distinctScores).toBe(3);
  });

  test("reports how many distinct scores an entrant produced across seeds", async () => {
    const comparison = await compare({
      seeds: [0, 1, 2],
      entrants: {
        winner: scoring({ bestScores: [0.9, 0.9, 0.9] }),
        varied: scoring({ bestScores: [0.3, 0.6, 0.9] }),
      },
    });

    expect(summaryFor(comparison, "winner").distinctScores).toBe(1);
    expect(summaryFor(comparison, "varied").distinctScores).toBe(3);
  });

  test("Holm-adjusts the p-value against the other entrants in the same call", async () => {
    // Both losers' differences are same-signed and unequal across the two
    // seeds, so each has a unique maximizing sign assignment on its own —
    // raw p = 1/4 apiece. With two losers in the family, Holm scales the
    // smallest by 2 and the running maximum carries that scaled value to
    // the other, so both land on 0.5 rather than reading as 0.25 twice.
    const comparison = await compare({
      seeds: [0, 1],
      entrants: {
        winner: scoring({ bestScores: [0.9, 0.95] }),
        loserA: scoring({ bestScores: [0.5, 0.6] }),
        loserB: scoring({ bestScores: [0.6, 0.5] }),
      },
    });

    const loserA = summaryFor(comparison, "loserA");
    const loserB = summaryFor(comparison, "loserB");

    expect(loserA.pValueVsWinner).toBeCloseTo(0.25);
    expect(loserB.pValueVsWinner).toBeCloseTo(0.25);
    expect(loserA.pValueVsWinnerHolm).toBeCloseTo(0.5);
    expect(loserB.pValueVsWinnerHolm).toBeCloseTo(0.5);
  });

  test("enumerates twenty seeds exactly rather than falling back to the normal approximation", async () => {
    // Same-signed, unequal-magnitude differences at n=20, the bench's own
    // seed count: the exact answer is the unique all-positive assignment,
    // 2^-20. Below EXACT_LIMIT=20 this fell back to a normal approximation
    // instead, which reads roughly 4x too large for exactly this shape.
    const seeds = Array.from({ length: 20 }, (_, index) => index);
    const winnerScores = seeds.map(() => 1.0);
    const loserScores = seeds.map((_, index) => (index === 19 ? 0.4 : 0.5));

    const comparison = await compare({
      seeds,
      entrants: {
        winner: scoring({ bestScores: winnerScores }),
        loser: scoring({ bestScores: loserScores }),
      },
    });

    expect(summaryFor(comparison, "loser").pValueVsWinner).toBeCloseTo(
      2 ** -20,
      9,
    );
  });

  test("keeps every run, so a caller can plot the spread rather than the mean", async () => {
    const comparison = await compare({
      seeds: [7, 8],
      entrants: { one: scoring({ bestScores: [0.4, 0.6] }) },
    });

    expect(comparison.runs.map((run) => run.seed)).toEqual([7, 8]);
    expect(comparison.runs.map((run) => run.score)).toEqual([0.4, 0.6]);
  });

  test("totals what each entrant spent, so a tie is broken on cost", async () => {
    const comparison = await compare({
      seeds: [0, 1],
      entrants: { one: scoring({ bestScores: [0.4, 0.6] }) },
    });

    expect(summaryFor(comparison, "one").meanMetricCalls).toBe(10);
    expect(summaryFor(comparison, "one").meanCostUsd).toBeCloseTo(2);
  });

  test("surfaces cache hits and reflection calls, so a caller can see rollouts that cost nothing", async () => {
    const comparison = await compare({
      seeds: [0, 1],
      entrants: {
        one: scoring({
          bestScores: [0.4, 0.6],
          cacheHits: 3,
          reflectionCalls: 2,
        }),
      },
    });

    expect(comparison.runs.map((run) => run.cacheHits)).toEqual([3, 3]);
    expect(comparison.runs.map((run) => run.reflectionCalls)).toEqual([2, 2]);
    expect(summaryFor(comparison, "one").meanCacheHits).toBeCloseTo(3);
    expect(summaryFor(comparison, "one").meanReflectionCalls).toBeCloseTo(2);
  });
});

function scoring(args: {
  bestScores: readonly number[];
  testScores?: readonly number[];
  cacheHits?: number;
  reflectionCalls?: number;
}) {
  const { bestScores, testScores, cacheHits = 0, reflectionCalls = 0 } = args;
  let call = 0;

  return async (): Promise<OptimizerResult<"instruction", string>> => {
    const index = call % bestScores.length;
    call += 1;

    return {
      bestCandidate: CANDIDATE,
      bestScore: bestScores[index] as number,
      ...(testScores === undefined
        ? {}
        : { testScore: testScores[index] as number }),
      metricCalls: 10,
      cacheHits,
      reflectionCalls,
      usage: {
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        costUsd: 2,
        rollouts: 10,
      },
      stopReason: "budgetExhausted",
    };
  };
}

function summaryFor(
  comparison: Awaited<ReturnType<typeof compare>>,
  entrant: string,
) {
  return comparison.summaries.find(
    (summary) => summary.entrant === entrant,
  ) as (typeof comparison.summaries)[number];
}
