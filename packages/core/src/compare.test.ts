import { describe, expect, test } from "vitest";
import { compare } from "./compare.js";
import type { OptimizerResult } from "./optimizer.js";

const CANDIDATE = { instruction: "answer" };

describe("compare", () => {
  test("summarizes each entrant over the seeds it was run on", async () => {
    const comparison = await compare({
      seeds: [0, 1, 2],
      entrants: {
        steady: scoring([0.5, 0.5, 0.5]),
        swingy: scoring([0.2, 0.5, 0.8]),
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
        weak: scoring([0.1, 0.2]),
        strong: scoring([0.7, 0.8]),
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
        overfitter: scoring([0.9], [0.1]),
        generalizer: scoring([0.6], [0.6]),
      },
    });

    expect(comparison.winner).toBe("generalizer");
  });

  test("scores how likely the winner's margin is under a coin flip", async () => {
    const comparison = await compare({
      seeds: [0, 1, 2, 3, 4],
      entrants: {
        better: scoring([0.6, 0.7, 0.8, 0.9, 1]),
        worse: scoring([0.1, 0.2, 0.3, 0.4, 0.5]),
      },
    });

    expect(summaryFor(comparison, "worse").pValueVsWinner).toBeCloseTo(1 / 32);
  });

  test("reports no significance for the winner against itself", async () => {
    const comparison = await compare({
      seeds: [0],
      entrants: { only: scoring([0.5]) },
    });

    expect(summaryFor(comparison, "only").pValueVsWinner).toBeUndefined();
  });

  test("calls nothing significant when two entrants tie on every seed", async () => {
    const comparison = await compare({
      seeds: [0, 1, 2],
      entrants: {
        left: scoring([0.5, 0.5, 0.5]),
        right: scoring([0.5, 0.5, 0.5]),
      },
    });

    expect(summaryFor(comparison, "right").pValueVsWinner).toBe(1);
  });

  test("keeps every run, so a caller can plot the spread rather than the mean", async () => {
    const comparison = await compare({
      seeds: [7, 8],
      entrants: { one: scoring([0.4, 0.6]) },
    });

    expect(comparison.runs.map((run) => run.seed)).toEqual([7, 8]);
    expect(comparison.runs.map((run) => run.score)).toEqual([0.4, 0.6]);
  });

  test("totals what each entrant spent, so a tie is broken on cost", async () => {
    const comparison = await compare({
      seeds: [0, 1],
      entrants: { one: scoring([0.4, 0.6]) },
    });

    expect(summaryFor(comparison, "one").meanMetricCalls).toBe(10);
    expect(summaryFor(comparison, "one").meanCostUsd).toBeCloseTo(2);
  });
});

function scoring(
  bestScores: readonly number[],
  testScores?: readonly number[],
) {
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
