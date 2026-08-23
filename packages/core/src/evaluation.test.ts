import { describe, expect, test } from "vitest";
import { createBudget } from "./budget.js";
import { createMemoryCache } from "./cache.js";
import { createEvaluator, withRetries } from "./evaluation.js";
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

/**
 * An adapter whose second instance comes back as a caught failure rather than
 * a measurement, for the first `failures` batches it sees. Nothing marks it
 * transient, which is what an adapter reports when no classifier was given.
 */
function failingAdapter(failures: number): Adapter<string, unknown, string> {
  let remaining = failures;

  return {
    evaluate: ({ batch }): EvaluationBatch<unknown, string> => {
      const failing = remaining > 0;
      remaining -= 1;

      return {
        outputs: [...batch],
        scores: batch.map((datum) => (datum === "b" && failing ? 0 : 1)),
        failed: batch.map((datum) => datum === "b" && failing),
      };
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

/** An evaluator whose adapter prices every rollout it runs. */
function pricedEvaluator() {
  return createEvaluator<string, unknown, string, "instruction">({
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
  });
}

function sweep(args: {
  evaluator: ReturnType<typeof pricedEvaluator>;
  split: "val" | "test";
  phase: "validation" | "test";
  charge?: boolean;
}) {
  const { evaluator, split, phase, charge } = args;

  return evaluator.evaluate({
    candidate: { instruction: "text" },
    batch: BATCH,
    ids: IDS,
    split,
    phase,
    candidateId: null,
    iteration: 0,
    ...(charge === undefined ? {} : { charge }),
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

  test("refuses a usage reading that is not a number at all", async () => {
    // `RolloutUsage` binds TypeScript callers and nothing else. A string cost
    // concatenates onto the totals instead of adding to them, and the ceiling
    // is then checked against text.
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: [...batch],
          scores: batch.map(() => 1),
          usage: batch.map(() => ({ costUsd: "0.002" as unknown as number })),
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

  test("refuses resumed usage no ceiling could be checked against", () => {
    expect(() =>
      createEvaluator<string, unknown, string, "instruction">({
        adapter: flakyAdapter(0),
        budget: createBudget({ maxMetricCalls: 100 }),
        usage: {
          inputTokens: 0,
          outputTokens: 0,
          totalTokens: 0,
          costUsd: Number.NaN,
          rollouts: 0,
        },
      }),
    ).toThrow(/costUsd/);
  });

  test("refuses absorbed usage no ceiling could be checked against", () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: flakyAdapter(0),
      budget: createBudget({ maxMetricCalls: 100 }),
    });

    expect(() =>
      evaluator.absorbUsage({
        inputTokens: 0,
        outputTokens: 0,
        totalTokens: 0,
        costUsd: Number.NaN,
        rollouts: 0,
      }),
    ).toThrow(/costUsd/);
  });

  test("keeps an uncharged sweep's usage out of the run's totals", async () => {
    const evaluator = pricedEvaluator();

    await sweep({ evaluator, split: "val", phase: "validation" });
    await sweep({
      evaluator,
      split: "test",
      phase: "test",
      charge: false,
    });

    // Only the charged sweep. `maxCostUsd` is checked against this, and a
    // held-out measurement the ceiling never bounded cannot be inside it.
    expect(evaluator.usage()).toEqual({
      inputTokens: 30,
      outputTokens: 12,
      totalTokens: 42,
      costUsd: 0.006,
      rollouts: 3,
    });
  });

  test("reports what an uncharged sweep spent on its own", async () => {
    const evaluator = pricedEvaluator();

    await sweep({ evaluator, split: "val", phase: "validation" });
    await sweep({
      evaluator,
      split: "test",
      phase: "test",
      charge: false,
    });

    expect(evaluator.unchargedUsage()).toEqual({
      inputTokens: 30,
      outputTokens: 12,
      totalTokens: 42,
      costUsd: 0.006,
      rollouts: 3,
    });
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

describe("failures the adapter caught rather than measured", () => {
  const CALL = {
    candidate: { instruction: "text" },
    batch: BATCH,
    ids: IDS,
    split: "val",
    phase: "validation",
    candidateId: 1,
    iteration: 0,
  } as const;

  test("re-runs a failed instance rather than serving its cached zero", async () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: failingAdapter(1),
      budget: createBudget({ maxMetricCalls: 100 }),
      cache: createMemoryCache(),
    });

    const first = await evaluator.evaluate(CALL);
    const second = await evaluator.evaluate(CALL);

    expect(first.scores).toEqual([1, 0, 1]);
    expect(second.scores).toEqual([1, 1, 1]);
  });

  test("keeps caching the instances the same batch did measure", async () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: failingAdapter(1),
      budget: createBudget({ maxMetricCalls: 100 }),
      cache: createMemoryCache(),
    });

    await evaluator.evaluate(CALL);

    expect(evaluator.entries()).toHaveLength(2);
  });

  test("counts a failure nothing classified as infrastructure", async () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: failingAdapter(1),
      budget: createBudget({ maxMetricCalls: 100 }),
    });

    await evaluator.evaluate(CALL);

    expect(evaluator.unclassifiedFailures()).toBe(1);
  });

  test("counts a failure reported through a traced evaluation too", async () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: failingAdapter(1),
      budget: createBudget({ maxMetricCalls: 100 }),
    });

    await evaluator.evaluateTraced({
      candidate: { instruction: "text" },
      batch: BATCH,
      split: "train",
      phase: "minibatch",
      candidateId: null,
      iteration: 0,
    });

    expect(evaluator.unclassifiedFailures()).toBe(1);
  });

  test("leaves a classified failure out of the count", async () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: {
        evaluate: ({ batch }): EvaluationBatch<unknown, string> => ({
          outputs: [...batch],
          scores: batch.map(() => 0),
          failed: batch.map(() => true),
          transient: batch.map(() => true),
        }),
      },
      budget: createBudget({ maxMetricCalls: 100 }),
      retry: { attempts: 0 },
    });

    await evaluator.evaluate(CALL);

    expect(evaluator.unclassifiedFailures()).toBe(0);
  });

  test("counts nothing when every rollout measured the candidate", async () => {
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: failingAdapter(0),
      budget: createBudget({ maxMetricCalls: 100 }),
    });

    await evaluator.evaluate(CALL);

    expect(evaluator.unclassifiedFailures()).toBe(0);
  });
});

describe("rollout progress", () => {
  /** An adapter that reports each rollout as it settles, as adapters should. */
  function reportingAdapter(): Adapter<string, unknown, string> {
    return {
      evaluate: ({ batch, onRollout }): EvaluationBatch<unknown, string> => {
        for (const _datum of batch) {
          onRollout?.();
        }
        return {
          outputs: [...batch],
          scores: batch.map(() => 1),
        };
      },
    };
  }

  test("reports each rollout against the size of the batch buying it", async () => {
    const progress: { completed: number; total: number }[] = [];
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: reportingAdapter(),
      budget: createBudget({ maxMetricCalls: 10 }),
      onRollout: (event) =>
        progress.push({ completed: event.completed, total: event.total }),
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

    expect(progress).toEqual([
      { completed: 1, total: 3 },
      { completed: 2, total: 3 },
      { completed: 3, total: 3 },
    ]);
  });

  test("counts only the instances the batch pays for", async () => {
    const progress: { completed: number; total: number }[] = [];
    const cache = createMemoryCache();
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: reportingAdapter(),
      budget: createBudget({ maxMetricCalls: 10 }),
      cache,
      onRollout: (event) =>
        progress.push({ completed: event.completed, total: event.total }),
    });
    const call = {
      candidate: { instruction: "text" },
      batch: BATCH,
      ids: IDS,
      split: "val" as const,
      phase: "validation" as const,
      candidateId: 1,
      iteration: 0,
    };

    await evaluator.evaluate(call);
    progress.length = 0;
    await evaluator.evaluate(call);

    expect(progress).toEqual([]);
  });

  test("names the evaluation each rollout belongs to", async () => {
    const seen: string[] = [];
    const evaluator = createEvaluator<string, unknown, string, "instruction">({
      adapter: reportingAdapter(),
      budget: createBudget({ maxMetricCalls: 10 }),
      onRollout: (event) =>
        seen.push(`${event.phase}/${event.split}#${String(event.candidateId)}`),
    });

    await evaluator.evaluate({
      candidate: { instruction: "text" },
      batch: ["a"],
      ids: ["0"],
      split: "train",
      phase: "minibatch",
      candidateId: null,
      iteration: 4,
    });

    expect(seen).toEqual(["minibatch/train#null"]);
  });
});

describe("withRetries", () => {
  test("returns what the attempt after a failure produced", async () => {
    let calls = 0;
    const model = withRetries(
      async () => {
        calls += 1;
        if (calls === 1) {
          throw new Error("429 rate limited");
        }
        return "proposed text";
      },
      { attempts: 2, delayMs: 0 },
    );

    expect(await model({ prompt: "x" })).toBe("proposed text");
    expect(calls).toBe(2);
  });

  test("rethrows once the attempts are spent", async () => {
    let calls = 0;
    const model = withRetries(
      async () => {
        calls += 1;
        throw new Error("429 rate limited");
      },
      { attempts: 2, delayMs: 0 },
    );

    await expect(model({ prompt: "x" })).rejects.toThrow("429 rate limited");
    expect(calls).toBe(3);
  });

  test("calls once when retrying is disabled", async () => {
    let calls = 0;
    const model = withRetries(
      async () => {
        calls += 1;
        throw new Error("429 rate limited");
      },
      { attempts: 0, delayMs: 0 },
    );

    await expect(model({ prompt: "x" })).rejects.toThrow("429 rate limited");
    expect(calls).toBe(1);
  });

  test("stops retrying once the run is aborted", async () => {
    const controller = new AbortController();
    let calls = 0;
    const model = withRetries(
      async () => {
        calls += 1;
        controller.abort();
        throw new Error("429 rate limited");
      },
      { attempts: 2, delayMs: 0, signal: controller.signal },
    );

    await expect(model({ prompt: "x" })).rejects.toThrow();
    expect(calls).toBe(1);
  });
});
