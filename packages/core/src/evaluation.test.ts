import { describe, expect, test } from "vitest";
import { createBudget } from "./budget.js";
import { createMemoryCache } from "./cache.js";
import { createEvaluator } from "./evaluation.js";
import type { Adapter, EvaluationBatch } from "./types.js";

const BATCH = ["a", "b", "c"];
const IDS = ["0", "1", "2"];

/**
 * An adapter whose second instance fails as infrastructure for the first
 * `failures` attempts it sees, and scores normally once it stops failing.
 */
function flakyAdapter(failures: number): Adapter<string, unknown, string> & {
  calls: () => number;
} {
  let remaining = failures;
  let calls = 0;

  return {
    calls: () => calls,
    evaluate: ({ batch }): EvaluationBatch<unknown, string> => {
      calls += 1;
      const scores: number[] = [];
      const transient: boolean[] = [];

      for (const datum of batch) {
        const failing = datum === "b" && remaining > 0;
        scores.push(failing ? 0 : 1);
        transient.push(failing);
      }
      remaining -= 1;

      return { outputs: [...batch], scores, transient };
    },
  };
}

function evaluate(args: {
  adapter: Adapter<string, unknown, string>;
  budget: ReturnType<typeof createBudget>;
  retry?: { attempts?: number; delayMs?: number };
}) {
  const { adapter, budget, retry } = args;
  const evaluator = createEvaluator<string, unknown, string, "instruction">({
    adapter,
    budget,
    cache: createMemoryCache(),
    ...(retry === undefined ? {} : { retry }),
  });

  return evaluator.evaluate({
    candidate: { instruction: "text" },
    batch: BATCH,
    ids: IDS,
    split: "val",
    phase: "validation",
    candidateId: 1,
    iteration: 0,
  });
}

describe("createEvaluator", () => {
  test("adds up the usage its adapter reports", async () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: [...batch],
          scores: batch.map(() => 1),
          usage: batch.map(() => ({
            inputTokens: 10,
            outputTokens: 4,
            costUsd: 0.002,
          })),
        }),
      },
      budget: createBudget({ maxMetricCalls: 100 }),
      cache: createMemoryCache(),
    });
    const run = () =>
      evaluator.evaluate({
        candidate: { instruction: "text" },
        batch: BATCH,
        ids: IDS,
        split: "val",
        phase: "validation",
        candidateId: 1,
        iteration: 0,
      });

    await run();
    // Cached instances buy no tokens, so the second pass must add none.
    await run();

    expect(evaluator.usage()).toEqual({
      inputTokens: 30,
      outputTokens: 12,
      totalTokens: 42,
      costUsd: 0.006,
      rollouts: 3,
    });
  });

  test("refuses a usage reading no ceiling could be checked against", async () => {
    // A NaN cost makes every `maxCostUsd` comparison false, so the ceiling
    // stops holding and the run spends the rest of its allowance without one.
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: [...batch],
          scores: batch.map(() => 1),
          usage: batch.map(() => ({ costUsd: Number.NaN })),
        }),
      },
      budget: createBudget({ maxMetricCalls: 100 }),
      cache: createMemoryCache(),
    });

    await expect(
      evaluator.evaluate({
        candidate: { instruction: "text" },
        batch: BATCH,
        ids: IDS,
        split: "val",
        phase: "validation",
        candidateId: 1,
        iteration: 0,
      }),
    ).rejects.toThrow(/costUsd/);
  });

  test("re-runs a transiently failed instance and keeps the score it earns", async () => {
    const adapter = flakyAdapter(1);

    const batch = await evaluate({
      adapter,
      budget: createBudget({ maxMetricCalls: 100 }),
      retry: { attempts: 2, delayMs: 0 },
    });

    expect(batch.scores).toEqual([1, 1, 1]);
    expect(batch.transient).toEqual([false, false, false]);
  });

  test("retries only the instances that failed", async () => {
    const adapter = flakyAdapter(1);
    const budget = createBudget({ maxMetricCalls: 100 });

    await evaluate({ adapter, budget, retry: { attempts: 2, delayMs: 0 } });

    // Three instances, then one retried instance.
    expect(budget.spent()).toBe(4);
  });

  test("reports an instance transient once its retries are spent", async () => {
    const adapter = flakyAdapter(5);

    const batch = await evaluate({
      adapter,
      budget: createBudget({ maxMetricCalls: 100 }),
      retry: { attempts: 2, delayMs: 0 },
    });

    expect(batch.transient).toEqual([false, true, false]);
    expect(adapter.calls()).toBe(3);
  });

  test("does not retry when the budget cannot cover the re-run", async () => {
    const adapter = flakyAdapter(1);

    const batch = await evaluate({
      adapter,
      budget: createBudget({ maxMetricCalls: 3 }),
      retry: { attempts: 2, delayMs: 0 },
    });

    expect(batch.transient).toEqual([false, true, false]);
    expect(adapter.calls()).toBe(1);
  });

  test("leaves a retried instance out of the cache when it never recovers", async () => {
    const cache = createMemoryCache();
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: flakyAdapter(5),
      budget: createBudget({ maxMetricCalls: 100 }),
      cache,
      retry: { attempts: 1, delayMs: 0 },
    });

    await evaluator.evaluate({
      candidate: { instruction: "text" },
      batch: BATCH,
      ids: IDS,
      split: "val",
      phase: "validation",
      candidateId: 1,
      iteration: 0,
    });

    expect(evaluator.entries()).toHaveLength(2);
  });
});
