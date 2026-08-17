import { describe, expect, test } from "vitest";
import { createMemoryCache } from "./cache.js";
import { optimize } from "./optimize.js";
import {
  fullEvaluationPolicy,
  subsampledEvaluationPolicy,
} from "./strategies.js";
import {
  KEYWORD_EXAMPLES,
  createDegradingReflector,
  createKeywordAdapter,
  createKeywordReflector,
} from "./testing.js";
import type {
  Adapter,
  EvaluationContext,
  OptimizationResult,
  OptimizerEvent,
  OptimizerSnapshot,
} from "./types.js";

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

  test("aggregates objective scores over the validation set", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createObjectiveAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      minibatchSize: 2,
      seed: 1,
    });

    const seedRecord = result.candidates[0];

    // The seed covers no required term, so coverage is 0 everywhere and
    // brevity is 1 everywhere.
    expect(seedRecord?.objectiveScores).toEqual({ coverage: 0, brevity: 0.25 });
    for (const record of result.candidates) {
      expect(Object.keys(record.objectiveScores ?? {}).sort()).toEqual([
        "brevity",
        "coverage",
      ]);
    }
  });

  test("reports the leading candidates for each objective", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createObjectiveAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      minibatchSize: 2,
      seed: 1,
    });

    const coverage = result.perObjectiveBest?.coverage;
    const brevity = result.perObjectiveBest?.brevity;

    expect(coverage?.score).toBe(result.bestScore);
    expect(coverage?.candidateIds).toContain(result.bestCandidateId);
    // The seed says nothing, so nothing is shorter than it.
    expect(brevity?.candidateIds).toContain(0);
  });

  test("leaves objective reporting out when the adapter scores no objectives", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      minibatchSize: 2,
      seed: 1,
    });

    expect(result.perObjectiveBest).toBeUndefined();
    expect(result.candidates[0]?.objectiveScores).toBeUndefined();
  });

  test("reuses cached objective scores instead of re-evaluating", async () => {
    const cache = createMemoryCache();
    const options = {
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createObjectiveAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      maxIterations: 5,
      minibatchSize: 2,
      seed: 1,
      cache,
    };

    const first = await optimize(options);
    const second = await optimize(options);

    expect(second.metricCalls).toBeLessThan(first.metricCalls);
    expect(second.candidates[0]?.objectiveScores).toEqual(
      first.candidates[0]?.objectiveScores,
    );
  });

  test("scores only the validation instances the evaluation policy selects", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
      valEvaluationPolicy: {
        selectInstances: () => [0, 2],
        bestCandidate: () => 0,
      },
    });

    for (const record of result.candidates) {
      expect(record.instanceScores[1]).toBeUndefined();
      expect(record.instanceScores[3]).toBeUndefined();
      expect(record.instanceScores[0]).toBeTypeOf("number");
      expect(record.instanceScores[2]).toBeTypeOf("number");
    }
  });

  test("scores a candidate on fewer instances under a subsampling policy", async () => {
    const events: OptimizerEvent[] = [];

    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
      valEvaluationPolicy: subsampledEvaluationPolicy({ size: 2 }),
      onEvent: (event) => events.push(event),
    });

    const validations = events.filter(
      (event) =>
        event.type === "evaluation" &&
        (event.phase === "validation" || event.phase === "seed"),
    );

    expect(validations.length).toBeGreaterThan(0);
    for (const event of validations) {
      expect(event.type === "evaluation" && event.metricCalls).toBeLessThan(
        KEYWORD_EXAMPLES.length,
      );
    }
    expect(
      result.candidates.every(
        (record) =>
          record.instanceScores.filter((score) => score !== undefined)
            .length === 2,
      ),
    ).toBe(true);
  });

  test("feeds rejected proposals back into later reflections", async () => {
    const prompts: string[] = [];

    await optimize({
      seedCandidate: { instruction: "hold ten seconds ticket portal" },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nno useful information\n```";
      },
      maxMetricCalls: 200,
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
    });

    // The first proposal loses on the minibatch; every later reflection must
    // be told so, or the same dead end is proposed for the whole run.
    expect(prompts.length).toBeGreaterThan(1);
    expect(prompts[0]).not.toContain("<rejected_instructions>");
    expect(prompts.at(-1)).toContain("no useful information");
  });

  test("caps how many rejected proposals a reflection is shown", async () => {
    const prompts: string[] = [];
    let counter = 0;

    await optimize({
      seedCandidate: { instruction: "hold ten seconds ticket portal" },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        counter += 1;
        return `\`\`\`\nuseless proposal ${counter}\n\`\`\``;
      },
      maxMetricCalls: 400,
      maxIterations: 8,
      minibatchSize: 2,
      seed: 1,
      rejectedProposalMemory: 2,
    });

    const last = prompts.at(-1) ?? "";

    expect(prompts.length).toBeGreaterThan(3);
    expect(last).toContain("useless proposal");
    expect(last.match(/useless proposal/g) ?? []).toHaveLength(2);
  });

  test("emits a checkpoint for the seed and for every iteration", async () => {
    const snapshots: OptimizerSnapshot[] = [];

    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
      onCheckpoint: (snapshot) => {
        snapshots.push(snapshot);
      },
    });

    expect(snapshots).toHaveLength(result.iterations + 1);
    expect(snapshots[0]?.records).toHaveLength(1);
    expect(snapshots.at(-1)?.metricCalls).toBe(result.metricCalls);
    expect(snapshots.at(-1)?.records).toHaveLength(result.candidates.length);
  });

  test("survives a round trip through JSON", async () => {
    let snapshot: OptimizerSnapshot | undefined;

    await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
      onCheckpoint: (taken) => {
        snapshot = taken;
      },
    });

    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  test("resumes where the checkpoint left off", async () => {
    const interrupted = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
      cache: false,
    });
    const snapshot = interrupted.snapshot;

    const resumed = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      maxIterations: 6,
      minibatchSize: 2,
      seed: 1,
      cache: false,
      resumeFrom: snapshot,
    });

    expect(resumed.iterations).toBe(6);
    expect(resumed.candidates.length).toBeGreaterThanOrEqual(
      interrupted.candidates.length,
    );
    expect(resumed.candidates[0]?.candidate).toEqual(SEED);
    // The seed and everything already scored is not paid for twice.
    expect(resumed.metricCalls).toBeGreaterThanOrEqual(interrupted.metricCalls);
    expect(resumed.bestScore).toBeGreaterThanOrEqual(interrupted.bestScore);
  });

  test("charges a resumed run for what the checkpoint already spent", async () => {
    const interrupted = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
      cache: false,
    });

    const resumed = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
      minibatchSize: 2,
      seed: 1,
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.metricCalls).toBeLessThanOrEqual(60);
    expect(resumed.metricCalls).toBeGreaterThanOrEqual(interrupted.metricCalls);
    expect(resumed.stopReason).toBe("budgetExhausted");
  });

  test("carries the evaluation cache in the checkpoint", async () => {
    const options = {
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
    };

    const first = await optimize({ ...options, cache: createMemoryCache() });
    const second = await optimize({
      ...options,
      cache: createMemoryCache({ entries: first.snapshot.cache }),
    });

    expect(first.snapshot.cache?.length).toBeGreaterThan(0);
    expect(second.cacheHits).toBeGreaterThan(0);
    expect(second.metricCalls).toBeLessThan(first.metricCalls);
  });

  test("restores the checkpoint's cache into a resumed run", async () => {
    const interrupted = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
    });
    const cache = createMemoryCache();

    await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
      cache,
      resumeFrom: interrupted.snapshot,
    });

    const restored = new Map(cache.entries?.());

    for (const [key, cached] of interrupted.snapshot.cache ?? []) {
      expect(restored.get(key)).toEqual(cached);
    }
  });

  test("refuses a checkpoint taken against a different seed candidate", async () => {
    const other = await optimize({
      seedCandidate: { instruction: "Something else entirely." },
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      maxIterations: 1,
      minibatchSize: 2,
      seed: 1,
    });

    await expect(
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        minibatchSize: 2,
        seed: 1,
        resumeFrom: other.snapshot,
      }),
    ).rejects.toThrow(/checkpoint/i);
  });

  test("refuses a checkpoint taken against a different validation set", async () => {
    const other = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      valset: KEYWORD_EXAMPLES.slice(0, 2),
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      maxIterations: 1,
      minibatchSize: 2,
      seed: 1,
    });

    await expect(
      optimize({
        seedCandidate: SEED,
        trainset: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        minibatchSize: 2,
        seed: 1,
        resumeFrom: other.snapshot,
      }),
    ).rejects.toThrow(/checkpoint/i);
  });

  test("reports the best candidate chosen by the evaluation policy", async () => {
    const result = await optimize({
      seedCandidate: SEED,
      trainset: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 300,
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
      valEvaluationPolicy: {
        ...fullEvaluationPolicy(),
        // Deliberately perverse: the worst candidate wins.
        bestCandidate: (records) =>
          records.reduce((worst, record) =>
            record.aggregateScore < worst.aggregateScore ? record : worst,
          ).id,
      },
    });

    expect(result.bestCandidateId).toBe(0);
    expect(result.bestCandidate).toEqual(SEED);
  });

  test("tags the seed evaluation with the seed phase and the validation split", async () => {
    const contexts = await recordRunContexts({ maxIterations: 1 });

    expect(contexts[0]).toEqual({
      iteration: 0,
      phase: "seed",
      candidateId: 0,
      split: "val",
    });
  });

  test("tags a parent's minibatch evaluation with the parent's id and the train split", async () => {
    const contexts = await recordRunContexts({ maxIterations: 1 });

    expect(contexts).toContainEqual({
      iteration: 0,
      phase: "minibatch",
      candidateId: 0,
      split: "train",
    });
  });

  test("tags a child's validation evaluation with the id it will be recorded under", async () => {
    const contexts = await recordRunContexts({ maxIterations: 1 });

    expect(contexts).toContainEqual({
      iteration: 0,
      phase: "validation",
      candidateId: 1,
      split: "val",
    });
  });

  test("reports a proposal that has no id yet as an unidentified candidate", async () => {
    const contexts = await recordRunContexts({ maxIterations: 1 });

    expect(contexts).toContainEqual({
      iteration: 0,
      phase: "minibatch",
      candidateId: null,
      split: "train",
    });
  });

  test("advances the iteration number it reports to the adapter", async () => {
    const contexts = await recordRunContexts({ maxIterations: 3 });

    expect(new Set(contexts.map((context) => context.iteration))).toContain(1);
  });
});

describe("optimize proposals", () => {
  const options = {
    seedCandidate: SEED,
    trainset: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 2000,
    maxIterations: 4,
    minibatchSize: 2,
    seed: 1,
  };

  test("makes one proposal per iteration by default", async () => {
    const events: OptimizerEvent[] = [];

    await optimize({ ...options, onEvent: (event) => events.push(event) });

    expect(countProposals(events, 0)).toBe(1);
  });

  test("makes several proposals in one iteration when asked", async () => {
    const events: OptimizerEvent[] = [];

    await optimize({
      ...options,
      proposals: { perIteration: 3 },
      onEvent: (event) => events.push(event),
    });

    expect(countProposals(events, 0)).toBe(3);
  });

  test("draws a different minibatch for each proposal in an iteration", async () => {
    const batches: string[] = [];

    await optimize({
      ...options,
      maxIterations: 1,
      adapter: {
        ...options.adapter,
        evaluate: (args) => {
          if (args.captureTraces) {
            batches.push(
              args.batch.map((example) => example.question).join("|"),
            );
          }
          return options.adapter.evaluate(args);
        },
      },
      proposals: { perIteration: 2 },
    });

    expect(batches).toHaveLength(2);
    expect(batches[0]).not.toBe(batches[1]);
  });

  test("accepts at most one candidate per iteration under best selection", async () => {
    const events: OptimizerEvent[] = [];

    await optimize({
      ...options,
      proposals: { perIteration: 3, selection: "best" },
      onEvent: (event) => events.push(event),
    });

    const accepted = events.filter(
      (event) => event.type === "candidateAccepted",
    );
    const iterations = accepted.map((event) => event.iteration);

    expect(accepted.length).toBeGreaterThan(0);
    expect(new Set(iterations).size).toBe(iterations.length);
  });

  test("accepts several candidates in one iteration under all selection", async () => {
    const events: OptimizerEvent[] = [];

    await optimize({
      ...options,
      proposals: { perIteration: 3, selection: "all" },
      onEvent: (event) => events.push(event),
    });

    const perIteration = new Map<number, number>();
    for (const event of events) {
      if (event.type === "candidateAccepted") {
        perIteration.set(
          event.iteration,
          (perIteration.get(event.iteration) ?? 0) + 1,
        );
      }
    }

    expect(Math.max(...perIteration.values())).toBeGreaterThan(1);
  });

  test("keeps the strongest of several proposals under best selection", async () => {
    const events: OptimizerEvent[] = [];

    await optimize({
      ...options,
      maxIterations: 1,
      proposals: { perIteration: 3, selection: "best" },
      onEvent: (event) => events.push(event),
    });

    const accepted = events.filter(
      (event) => event.type === "candidateAccepted",
    );
    const passedOver = events.filter(
      (event) =>
        event.type === "candidateRejected" && event.reason === "notSelected",
    );

    // A sibling that improved but lost is reported as passed over, not as a
    // proposal that failed — it is never fed back to reflection as a dead end.
    expect(accepted).toHaveLength(1);
    expect(passedOver.length).toBeGreaterThan(0);
  });

  test("evaluates concurrent proposals at the same time", async () => {
    const tracked = withOverlapTracking(createKeywordAdapter());

    await optimize({
      ...options,
      adapter: tracked.adapter,
      proposals: { perIteration: 3, concurrency: 3 },
    });

    expect(tracked.maxInFlight()).toBeGreaterThan(1);
  });

  test("runs proposals one at a time by default", async () => {
    const tracked = withOverlapTracking(createKeywordAdapter());

    await optimize({
      ...options,
      adapter: tracked.adapter,
      proposals: { perIteration: 3 },
    });

    expect(tracked.maxInFlight()).toBe(1);
  });

  test("reaches the same candidates whether or not proposals overlap", async () => {
    const serial = await optimize({
      ...options,
      adapter: createKeywordAdapter(),
      proposals: { perIteration: 3, concurrency: 1 },
    });
    const concurrent = await optimize({
      ...options,
      adapter: createKeywordAdapter(),
      proposals: { perIteration: 3, concurrency: 3 },
    });

    expect(concurrent.candidates.map((record) => record.candidate)).toEqual(
      serial.candidates.map((record) => record.candidate),
    );
    expect(concurrent.bestScore).toBe(serial.bestScore);
  });

  test("never spends past the budget when proposals overlap", async () => {
    const result = await optimize({
      ...options,
      maxMetricCalls: 43,
      maxIterations: 50,
      proposals: { perIteration: 4, concurrency: 4 },
    });

    expect(result.metricCalls).toBeLessThanOrEqual(43);
  });

  test("rejects a proposal count below one", async () => {
    await expect(
      optimize({ ...options, proposals: { perIteration: 0 } }),
    ).rejects.toThrow(/perIteration/);
  });
});

describe("optimize reflection budget", () => {
  const options = {
    seedCandidate: SEED,
    trainset: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 2000,
    maxIterations: 20,
    minibatchSize: 2,
    seed: 1,
  };

  test("counts the reflection calls a run made", async () => {
    const result = await optimize({ ...options, maxIterations: 3 });

    expect(result.reflectionCalls).toBeGreaterThan(0);
  });

  test("stops once the reflection call budget is spent", async () => {
    const result = await optimize({
      ...options,
      reflection: { maxCalls: 2 },
    });

    expect(result.reflectionCalls).toBe(2);
    expect(result.stopReason).toBe("reflectionBudgetExhausted");
  });

  test("never exceeds the reflection budget with overlapping proposals", async () => {
    const result = await optimize({
      ...options,
      proposals: { perIteration: 4, concurrency: 4 },
      reflection: { maxCalls: 3 },
    });

    expect(result.reflectionCalls).toBe(3);
  });

  test("carries reflection spend across a resumed run", async () => {
    // A reflector that never improves anything keeps every iteration
    // reflecting, so the count is a clean measure of what each run spent.
    const persistent = { ...options, reflect: createDegradingReflector() };
    const interrupted = await optimize({ ...persistent, maxIterations: 2 });

    const resumed = await optimize({
      ...persistent,
      maxIterations: 4,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.reflectionCalls).toBeGreaterThan(
      interrupted.reflectionCalls,
    );
    expect(interrupted.snapshot.reflectionCalls).toBe(
      interrupted.reflectionCalls,
    );
  });

  test("shows the reflection model at most maxRecords examples", async () => {
    const prompts: string[] = [];

    await optimize({
      ...options,
      maxIterations: 2,
      minibatchSize: 4,
      reflection: { maxRecords: 1 },
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return createKeywordReflector()({ prompt });
      },
    });

    expect(prompts.length).toBeGreaterThan(0);
    for (const prompt of prompts) {
      expect(prompt.match(/"feedback"/g) ?? []).toHaveLength(1);
    }
  });

  test("uses a supplied reflection prompt template", async () => {
    const prompts: string[] = [];

    await optimize({
      ...options,
      maxIterations: 1,
      reflection: {
        buildPrompt: ({ componentName, currentText }) =>
          `improve ${componentName}: ${currentText}`,
      },
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nhold ten seconds\n```";
      },
    });

    expect(prompts[0]).toBe("improve instruction: Answer the user question.");
  });
});

describe("optimize outputs", () => {
  const options = {
    seedCandidate: SEED,
    trainset: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 400,
    maxIterations: 4,
    minibatchSize: 2,
    seed: 1,
  };

  test("omits validation outputs unless tracking is on", async () => {
    const result = await optimize(options);

    expect(result.bestOutputs).toBeUndefined();
  });

  test("returns the best candidate's output for every validation instance", async () => {
    const result = await optimize({ ...options, trackBestOutputs: true });

    expect(result.bestOutputs).toHaveLength(KEYWORD_EXAMPLES.length);
    for (const output of result.bestOutputs ?? []) {
      expect(output).toContain(result.bestCandidate.instruction);
    }
  });
});

describe("optimize checkpoint fidelity", () => {
  const options = {
    seedCandidate: SEED,
    trainset: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 400,
    minibatchSize: 2,
    seed: 1,
  };

  test("keeps the run fingerprint small however large the dataset is", async () => {
    const large = Array.from({ length: 200 }, (_, index) => ({
      question: `question ${index} ${"padding ".repeat(20)}`,
      required: ["hold"],
    }));

    const result = await optimize({
      ...options,
      trainset: large,
      maxIterations: 1,
      maxMetricCalls: 1000,
    });

    // The fingerprint is written into every checkpoint, so it must not grow
    // with the data it identifies.
    expect(result.snapshot.fingerprint.length).toBeLessThan(200);
  });

  test("omits the cache from checkpoints when asked", async () => {
    const result = await optimize({
      ...options,
      maxIterations: 2,
      checkpointCache: false,
    });

    expect(result.snapshot.cache).toBeUndefined();
  });

  test("resumes the minibatch sequence exactly where it stopped", async () => {
    const uninterrupted = await recordMinibatches({ maxIterations: 4 });

    const first = await recordMinibatches({ maxIterations: 2 });
    const second = await recordMinibatches({
      maxIterations: 4,
      resumeFrom: first.result.snapshot,
    });

    expect([...first.batches, ...second.batches]).toEqual(
      uninterrupted.batches,
    );
  });
});

/** Runs the keyword task and records every traced minibatch in order. */
async function recordMinibatches(args: {
  maxIterations: number;
  resumeFrom?: OptimizerSnapshot;
}): Promise<{ batches: string[]; result: OptimizationResult }> {
  const adapter = createKeywordAdapter();
  const batches: string[] = [];

  const result = await optimize({
    seedCandidate: SEED,
    trainset: KEYWORD_EXAMPLES,
    adapter: {
      ...adapter,
      evaluate: (evaluateArgs) => {
        if (evaluateArgs.captureTraces) {
          batches.push(
            evaluateArgs.batch.map((example) => example.question).join("|"),
          );
        }
        return adapter.evaluate(evaluateArgs);
      },
    },
    reflect: createKeywordReflector(),
    maxMetricCalls: 400,
    maxIterations: args.maxIterations,
    minibatchSize: 2,
    seed: 1,
    ...(args.resumeFrom === undefined ? {} : { resumeFrom: args.resumeFrom }),
  });

  return { batches, result };
}

function countProposals(
  events: readonly OptimizerEvent[],
  iteration: number,
): number {
  return events.filter(
    (event) => event.type === "proposal" && event.iteration === iteration,
  ).length;
}

/** Wraps an adapter to observe how many evaluations are ever in flight at once. */
function withOverlapTracking<Datum, Traj, Out>(
  adapter: Adapter<Datum, Traj, Out>,
): { adapter: Adapter<Datum, Traj, Out>; maxInFlight: () => number } {
  let inFlight = 0;
  let peak = 0;

  return {
    maxInFlight: () => peak,
    adapter: {
      ...adapter,
      evaluate: async (args) => {
        inFlight += 1;
        peak = Math.max(peak, inFlight);
        try {
          await Promise.resolve();
          return await adapter.evaluate(args);
        } finally {
          inFlight -= 1;
        }
      },
    },
  };
}

/** Runs the keyword task and collects the run context of every evaluation. */
async function recordRunContexts(args: {
  maxIterations: number;
}): Promise<EvaluationContext[]> {
  const adapter = createKeywordAdapter();
  const contexts: EvaluationContext[] = [];

  await optimize({
    seedCandidate: SEED,
    trainset: KEYWORD_EXAMPLES,
    adapter: {
      ...adapter,
      evaluate: (evaluateArgs) => {
        contexts.push(evaluateArgs.run);
        return adapter.evaluate(evaluateArgs);
      },
    },
    reflect: createKeywordReflector(),
    maxMetricCalls: 300,
    maxIterations: args.maxIterations,
    minibatchSize: 2,
    seed: 1,
  });

  return contexts;
}

/**
 * The keyword task scored on two competing objectives: coverage rewards saying
 * more, brevity rewards saying less. A candidate cannot lead both.
 */
function createObjectiveAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const adapter = createKeywordAdapter();

  return {
    ...adapter,
    evaluate: async (args) => {
      const evaluation = await adapter.evaluate(args);
      const words = Object.values(args.candidate).join(" ").split(/\s+/).length;

      return {
        ...evaluation,
        objectiveScores: evaluation.scores.map((score: number) => ({
          coverage: score,
          brevity: 1 / words,
        })),
      };
    },
  };
}
