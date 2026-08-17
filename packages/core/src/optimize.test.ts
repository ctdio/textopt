import { describe, expect, test } from "vitest";
import { createMemoryCache } from "./cache.js";
import { optimize } from "./optimize.js";
import {
  KEYWORD_EXAMPLES,
  createDegradingReflector,
  createKeywordAdapter,
  createKeywordReflector,
} from "./testing.js";
import type { OptimizerEvent } from "./types.js";

const SEED = { instruction: "Answer the user question." };

describe("optimize", () => {
  test("improves the aggregate score over the seed candidate", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      minibatchSize: 2,
      seed: 1,
    });

    const seedRecord = result.candidates[0];

    expect(seedRecord?.aggregateScore).toBe(0);
    expect(result.bestScore).toBeGreaterThan(0);
    expect(result.bestCandidate.instruction).not.toBe(SEED.instruction);
  });

  test("reaches a perfect score on a fully learnable task", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      minibatchSize: 2,
      seed: 1,
    });

    expect(result.bestScore).toBe(1);
  });

  test("never spends more than the metric call budget", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 37,
      minibatchSize: 2,
      seed: 1,
    });

    expect(result.metricCalls).toBeLessThanOrEqual(37);
    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("returns the seed when the budget only covers the seed evaluation", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: KEYWORD_EXAMPLES.length,
      minibatchSize: 2,
      seed: 1,
    });

    expect(result.candidates).toHaveLength(1);
    expect(result.bestCandidate).toEqual(SEED);
    expect(result.iterations).toBe(0);
  });

  test("records parent lineage for every accepted candidate", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      minibatchSize: 2,
      seed: 1,
    });

    const children = result.candidates.filter((record) => record.id !== 0);

    expect(children.length).toBeGreaterThan(0);
    for (const child of children) {
      expect(child.parentIds.length).toBeGreaterThan(0);
      expect(child.source).not.toBe("seed");
    }
  });

  test("keeps one score per validation instance for every candidate", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      minibatchSize: 2,
      seed: 1,
    });

    for (const row of result.scoreMatrix) {
      expect(row).toHaveLength(KEYWORD_EXAMPLES.length);
    }
    expect(result.scoreMatrix).toHaveLength(result.candidates.length);
  });

  test("rejects children that do not beat their parent on the minibatch", async () => {
    const result = await optimize({
      seedCandidate: { instruction: "hold ten seconds ticket portal" },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createDegradingReflector(),
      maxMetricCalls: 200,
      minibatchSize: 2,
      seed: 1,
    });

    expect(result.candidates).toHaveLength(1);
    expect(result.bestCandidate.instruction).toBe(
      "hold ten seconds ticket portal",
    );
  });

  test("emits a start event first and a finish event last", async () => {
    const events: OptimizerEvent[] = [];

    await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      minibatchSize: 2,
      seed: 1,
      onEvent: (event) => events.push(event),
    });

    expect(events[0]?.type).toBe("start");
    expect(events.at(-1)?.type).toBe("finish");
    expect(events.some((event) => event.type === "candidateAccepted")).toBe(
      true,
    );
  });

  test("stops when the abort signal fires", async () => {
    const controller = new AbortController();

    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      minibatchSize: 2,
      seed: 1,
      signal: controller.signal,
      onEvent: (event) => {
        if (event.type === "candidateAccepted") {
          controller.abort();
        }
      },
    });

    expect(result.stopReason).toBe("aborted");
    expect(result.metricCalls).toBeLessThan(500);
  });

  test("produces identical results for the same seed", async () => {
    const run = () =>
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 150,
        minibatchSize: 2,
        seed: 7,
      });

    const first = await run();
    const second = await run();

    expect(second.bestCandidate).toEqual(first.bestCandidate);
    expect(second.metricCalls).toBe(first.metricCalls);
    expect(second.candidates.length).toBe(first.candidates.length);
  });

  test("a shared cache makes an identical run cheaper", async () => {
    const cache = createMemoryCache();
    const run = () =>
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 500,
        maxIterations: 5,
        minibatchSize: 2,
        seed: 7,
        cache,
      });

    const first = await run();
    const second = await run();

    expect(second.cacheHits).toBeGreaterThan(0);
    expect(second.metricCalls).toBeLessThan(first.metricCalls);
    expect(second.bestCandidate).toEqual(first.bestCandidate);
  });

  test("stops at maxIterations", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
    });

    expect(result.iterations).toBe(3);
    expect(result.stopReason).toBe("maxIterations");
  });

  test("uses a separate validation set when provided", async () => {
    const valset = KEYWORD_EXAMPLES.slice(0, 2);

    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      valset,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 120,
      minibatchSize: 2,
      seed: 1,
    });

    for (const row of result.scoreMatrix) {
      expect(row).toHaveLength(2);
    }
  });

  test("rotates across components of a multi-component candidate", async () => {
    const result = await optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      minibatchSize: 2,
      seed: 1,
    });

    const updated = new Set(
      result.candidates.flatMap((record) => record.updatedComponents),
    );

    expect(updated).toContain("retriever");
    expect(updated).toContain("writer");
  });

  test("reports a pareto frontier drawn from evaluated candidates", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      minibatchSize: 2,
      seed: 1,
    });

    expect(result.paretoFrontier.length).toBeGreaterThan(0);
    for (const record of result.paretoFrontier) {
      expect(result.candidates).toContain(record);
    }
  });

  test("merges complementary lineages into a stronger candidate", async () => {
    // Force the two first children to branch off the seed and improve
    // different components, which is exactly the situation system-aware merge
    // exists to exploit.
    const result = await optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 600,
      maxIterations: 6,
      minibatchSize: 2,
      seed: 1,
      candidateSelector: ({ state }) =>
        state.aggregateScores.length < 3 ? 0 : state.aggregateScores.length - 1,
      componentSelector: ({ iteration }) => [
        iteration % 2 === 0 ? "retriever" : "writer",
      ],
      merge: { enabled: true },
    });

    const merged = result.candidates.filter(
      (record) => record.source === "merge",
    );

    expect(merged.length).toBeGreaterThan(0);
    expect(merged[0]?.parentIds).toHaveLength(2);
  });

  test("keeps optimizing when a merge cannot be afforded", async () => {
    // The merge gate is charged against the same budget as mutation, so a
    // merge that no longer fits must be skipped — not treated as the end of
    // the run while mutation iterations are still affordable.
    const withMerge = await optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
      minibatchSize: 2,
      seed: 1,
      merge: { enabled: true },
    });
    const withoutMerge = await optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
      minibatchSize: 2,
      seed: 1,
      merge: { enabled: false },
    });

    // Enabling merge must not cost the run a meaningful slice of its budget.
    expect(withMerge.metricCalls).toBeGreaterThanOrEqual(
      withoutMerge.metricCalls - KEYWORD_EXAMPLES.length,
    );
  });

  test("does not merge when merging is disabled", async () => {
    const result = await optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 600,
      maxIterations: 6,
      minibatchSize: 2,
      seed: 1,
      candidateSelector: ({ state }) =>
        state.aggregateScores.length < 3 ? 0 : state.aggregateScores.length - 1,
      componentSelector: ({ cursor }) => [
        cursor % 2 === 0 ? "retriever" : "writer",
      ],
      merge: { enabled: false },
    });

    expect(result.candidates.every((record) => record.source !== "merge")).toBe(
      true,
    );
  });

  test("rejects an empty validation set", async () => {
    await expect(
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        valset: [],
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        seed: 1,
      }),
    ).rejects.toThrow(/valset/i);
  });

  test("scores validation instances separately from trainset instances sharing an id", async () => {
    // Positional instance ids are a reasonable thing for a user to write, and
    // they make train instance "0" and val instance "0" collide in the cache.
    type Instance = { split: "train" | "val" };
    const trainset: Instance[] = [{ split: "train" }, { split: "train" }];
    const valset: Instance[] = [{ split: "val" }];
    let revision = 0;

    const result = await optimize({
      seedCandidate: SEED,
      trainset,
      valset,
      adapter: {
        // Val instances always score 1; train instances improve with each
        // revision so children actually get accepted and reach validation.
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map((datum) =>
            datum.split === "val"
              ? 1
              : Math.min(
                  1,
                  Number(
                    /revision (\d+)/.exec(candidate.instruction ?? "")?.[1] ??
                      0,
                  ) / 10,
                ),
          ),
          feedback: batch.map(() => "keep going"),
          trajectories: batch.map(() => null),
        }),
        makeReflectiveDataset: ({ batch, componentsToUpdate }) =>
          Object.fromEntries(
            componentsToUpdate.map((component) => [
              component,
              batch.map(() => ({
                inputs: {},
                generatedOutputs: "",
                feedback: "keep going",
              })),
            ]),
          ),
      },
      reflect: async () => {
        revision += 1;
        return `\`\`\`\nrevision ${revision}\n\`\`\``;
      },
      maxMetricCalls: 60,
      minibatchSize: 2,
      seed: 1,
      instanceId: ({ index }) => String(index),
    });

    // Every val instance scores 1, so no candidate may record a 0 for it.
    for (const record of result.candidates) {
      expect(record.instanceScores).toEqual([1]);
    }
  });

  test("rethrows a failure that produced no evaluations, even when raiseOnError is false", async () => {
    // Tolerating an error that made no progress is how a run burns its entire
    // budget on iterations that did nothing.
    const adapter = createKeywordAdapter();

    await expect(
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: (args) => {
            if (args.captureTraces) {
              throw new Error("tracing backend down");
            }
            return adapter.evaluate(args);
          },
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 1000,
        minibatchSize: 2,
        seed: 1,
        raiseOnError: false,
      }),
    ).rejects.toThrow("tracing backend down");
  });

  test("tolerates a failure that arrives after evaluations succeeded", async () => {
    const adapter = createKeywordAdapter();
    let errorEvents = 0;

    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: {
        ...adapter,
        makeReflectiveDataset: () => {
          throw new Error("reflective dataset failed");
        },
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
      raiseOnError: false,
      onEvent: (event) => {
        if (event.type === "error") {
          errorEvents += 1;
        }
      },
    });

    expect(errorEvents).toBe(3);
    expect(result.stopReason).toBe("maxIterations");
    // Seed validation plus one parent minibatch per iteration — a failed
    // evaluation is never charged.
    expect(result.metricCalls).toBe(KEYWORD_EXAMPLES.length + 3 * 2);
  });

  test("advances a component cursor per candidate rather than per iteration", async () => {
    const result = await optimize({
      seedCandidate: { alpha: "a", beta: "b", gamma: "c" },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createDegradingReflector(),
      maxMetricCalls: 400,
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
      candidateSelector: () => 0,
    });

    // The seed was the parent twice, so its own cursor advanced twice —
    // independently of which global iteration each selection happened on.
    expect(result.candidates[0]?.componentCursor).toBe(2);
  });

  test("skips reflection when the parent minibatch is already perfect", async () => {
    let reflectCalls = 0;

    await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 1),
          feedback: batch.map(() => "perfect"),
          trajectories: batch.map(() => null),
        }),
        makeReflectiveDataset: ({ batch, componentsToUpdate }) =>
          Object.fromEntries(
            componentsToUpdate.map((component) => [
              component,
              batch.map(() => ({
                inputs: {},
                generatedOutputs: "",
                feedback: "perfect",
              })),
            ]),
          ),
      },
      reflect: async () => {
        reflectCalls += 1;
        return "```\nnever needed\n```";
      },
      maxMetricCalls: 100,
      maxIterations: 5,
      minibatchSize: 2,
      seed: 1,
    });

    expect(reflectCalls).toBe(0);
  });

  test("does not cache a score the adapter marked transient", async () => {
    // A rate limit is not the candidate's fault. Caching the zero it produced
    // would pin that instance to zero for every later evaluation of the same
    // text, with no rollout ever attempted again.
    const cache = createMemoryCache();
    const run = () =>
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: {
          evaluate: ({ batch }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0),
            feedback: batch.map(() => "rate limited"),
            transient: batch.map(() => true),
          }),
          makeReflectiveDataset: () => ({}),
        },
        reflect: async () => "```\nunused\n```",
        maxMetricCalls: 100,
        maxIterations: 2,
        minibatchSize: 2,
        seed: 1,
        cache,
      });

    await run();
    const second = await run();

    expect(second.cacheHits).toBe(0);
  });

  test("caches a score the adapter did not mark transient", async () => {
    const cache = createMemoryCache();
    const run = () =>
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: {
          evaluate: ({ batch }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0),
            feedback: batch.map(() => "genuinely wrong"),
            transient: batch.map(() => false),
          }),
          makeReflectiveDataset: () => ({}),
        },
        reflect: async () => "```\nunused\n```",
        maxMetricCalls: 100,
        maxIterations: 2,
        minibatchSize: 2,
        seed: 1,
        cache,
      });

    await run();
    const second = await run();

    expect(second.cacheHits).toBeGreaterThan(0);
  });

  test("rejects a non-finite score from the adapter", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: ({ batch }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map(() => Number.NaN),
            feedback: batch.map(() => ""),
          }),
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        minibatchSize: 2,
        seed: 1,
      }),
    ).rejects.toThrow(/finite/i);
  });

  test("names the offending instance when a score is not a number", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: ({ batch }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map((_, index) =>
              index === 1 ? Number.POSITIVE_INFINITY : 0,
            ),
            feedback: batch.map(() => ""),
          }),
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        minibatchSize: 2,
        seed: 1,
      }),
    ).rejects.toThrow(/index 1/i);
  });

  test("stops cleanly when an adapter aborts mid-iteration", async () => {
    // An adapter that honours the signal throws rather than returning zeros.
    // That is a cancellation, not a run failure, so it must end the run
    // cleanly instead of rejecting the way any other adapter error would.
    const controller = new AbortController();
    const adapter = createKeywordAdapter();

    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: {
        ...adapter,
        evaluate: (args) => {
          if (args.captureTraces) {
            controller.abort();
            throw new Error("The operation was aborted");
          }
          return adapter.evaluate(args);
        },
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      minibatchSize: 2,
      seed: 1,
      signal: controller.signal,
    });

    expect(result.stopReason).toBe("aborted");
    expect(result.metricCalls).toBe(KEYWORD_EXAMPLES.length);
  });

  test("propagates a failure during seed evaluation", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: () => {
            throw new Error("adapter exploded");
          },
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 50,
        minibatchSize: 2,
        seed: 1,
      }),
    ).rejects.toThrow("adapter exploded");
  });
});
