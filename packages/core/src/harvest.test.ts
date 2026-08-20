import { describe, expect, test } from "vitest";
import { harvestRollouts } from "./harvest.js";
import { createSeededRng } from "./rng.js";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "./testing.js";
import type { Adapter } from "./types.js";

/** Eight instances, so a default cap of four would be visible. */
const POOL = [...KEYWORD_EXAMPLES, ...KEYWORD_EXAMPLES];

const ANSWERS_EVERYTHING =
  "hold ten seconds ticket portal thirty days billing prorated";

describe("harvestRollouts", () => {
  const adapter = (): Adapter<
    (typeof KEYWORD_EXAMPLES)[number],
    unknown,
    string
  > => createKeywordAdapter();

  test("keeps only rollouts that clear the score threshold", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      // Answers two of the four instances perfectly and neither of the others.
      candidate: { instruction: "hold ten seconds ticket portal" },
      data: KEYWORD_EXAMPLES,
      minScore: 1,
    });

    expect(result.rollouts).toHaveLength(2);
    for (const rollout of result.rollouts) {
      expect(rollout.score).toBe(1);
    }
  });

  test("keeps every rollout the metric rewarded when given no threshold", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      // Half credit on three instances and nothing on the fourth.
      candidate: { instruction: "hold ticket billing" },
      data: KEYWORD_EXAMPLES,
    });

    expect(result.rollouts).toHaveLength(3);
    for (const rollout of result.rollouts) {
      expect(rollout.score).toBe(0.5);
    }
  });

  test("keeps nothing the metric scored at zero when given no threshold", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      candidate: { instruction: "nothing useful here" },
      data: KEYWORD_EXAMPLES,
    });

    expect(result.rollouts).toEqual([]);
  });

  test("keeps every rewarded rollout when no ceiling is set", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      candidate: { instruction: ANSWERS_EVERYTHING },
      data: POOL,
      minScore: 1,
    });

    expect(result.rollouts).toHaveLength(POOL.length);
  });

  test("stops once it has collected the rollouts asked for", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      candidate: { instruction: ANSWERS_EVERYTHING },
      data: POOL,
      minScore: 1,
      maxRollouts: 3,
      batchSize: 1,
    });

    expect(result.rollouts).toHaveLength(3);
    // Three instances answered, the rest never attempted.
    expect(result.metricCalls).toBe(3);
  });

  test("stops at the metric call ceiling", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      candidate: { instruction: "nothing useful here" },
      data: POOL,
      maxMetricCalls: 2,
      batchSize: 1,
    });

    expect(result.metricCalls).toBe(2);
  });

  test("stops at the cost ceiling", async () => {
    // Rollout counts are a poor proxy for spend, and a harvesting pass over a
    // large pool is where that gap is widest: the caller bounding dollars can
    // only do so if this pass reads the same ceiling between its batches.
    const keyword = createKeywordAdapter();
    const result = await harvestRollouts({
      adapter: {
        evaluate: (args) => ({
          ...keyword.evaluate(args),
          usage: args.batch.map(() => ({ costUsd: 1 })),
        }),
      },
      candidate: { instruction: "nothing useful here" },
      data: POOL,
      batchSize: 2,
      maxCostUsd: 3,
    });

    // Two batches of two: the second crosses the ceiling and no third runs.
    expect(result.usage.costUsd).toBe(4);
    expect(result.metricCalls).toBe(4);
  });

  test("reports what it spent", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      candidate: { instruction: "nothing useful here" },
      data: KEYWORD_EXAMPLES,
    });

    expect(result.metricCalls).toBe(KEYWORD_EXAMPLES.length);
    expect(result.attempted).toBe(KEYWORD_EXAMPLES.length);
  });

  test("carries the instance each output came from", async () => {
    const result = await harvestRollouts({
      adapter: adapter(),
      candidate: { instruction: "hold ten seconds" },
      data: KEYWORD_EXAMPLES,
      minScore: 1,
    });

    expect(result.rollouts[0]?.input).toEqual({
      question: "How do I reset a device?",
      required: ["hold", "ten seconds"],
    });
    expect(typeof result.rollouts[0]?.output).toBe("string");
  });

  test("samples the pool in a reproducible order when given an rng", async () => {
    const run = async () =>
      (
        await harvestRollouts({
          adapter: adapter(),
          candidate: { instruction: "hold ten seconds ticket portal" },
          data: KEYWORD_EXAMPLES,
          minScore: 1,
          maxRollouts: 1,
          batchSize: 1,
          rng: createSeededRng(7),
        })
      ).rollouts.map((rollout) => JSON.stringify(rollout.input));

    expect(await run()).toEqual(await run());
  });

  test("refuses an empty pool", async () => {
    await expect(
      harvestRollouts({
        adapter: adapter(),
        candidate: { instruction: "x" },
        data: [],
      }),
    ).rejects.toThrow(/data/);
  });
});
