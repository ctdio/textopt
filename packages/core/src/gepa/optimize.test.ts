import { afterEach, describe, expect, test, vi } from "vitest";
import { createMemoryCache } from "../cache.js";
import type { Optimizer, OptimizerResult } from "../optimizer.js";
import {
  KEYWORD_EXAMPLES,
  createDegradingReflector,
  createKeywordAdapter,
  createKeywordReflector,
} from "../testing.js";
import type { Candidate, EvaluationContext, TextModel } from "../types.js";
import { GepaOptimizer } from "./optimize.js";
import type { GepaResult } from "./optimize.js";
import {
  fullEvaluationPolicy,
  pairedPermutationAcceptance,
  subsampledEvaluationPolicy,
} from "./strategies.js";
import type {
  GepaAdapter,
  GepaEvent,
  GepaSnapshot,
  GepaStopReason,
} from "./types.js";

const SEED = { instruction: "Answer the user question." };
/** Instances that need a different number of marks to be answered. */
const MARK_EXAMPLES = [{ need: 1 }, { need: 2 }, { need: 3 }, { need: 4 }];
/** Instances each answered by one named component of the candidate. */
const PART_TASKS = ["alpha", "beta", "gamma"].flatMap((part) =>
  [1, 2, 3, 4].map((need) => ({ part, need })),
);

describe("optimize", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("stops once the wall clock deadline passes", async () => {
    // Rollout and cost ceilings bound what a run spends, not how long it
    // takes: a run stuck behind a rate limit costs nothing and runs forever.
    vi.useFakeTimers();

    const result = await new GepaOptimizer({
      maxIterations: 20,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: {
        ...createKeywordAdapter(),
        evaluate: ({ batch }) => {
          vi.advanceTimersByTime(400);
          return {
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0.5),
            feedback: batch.map(() => "measured"),
          };
        },
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 1000,
      maxWallClockMs: 1000,
    });

    expect(result.stopReason).toBe("deadlineReached");
  });
  test("stops once the reported cost reaches the ceiling", async () => {
    // Rollouts are the budget, but a run is paid for in dollars: a candidate
    // that grows its prompt costs more per rollout than the seed did, so a
    // rollout ceiling alone cannot bound spend.
    const result = await new GepaOptimizer({
      maxIterations: 5,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: {
        ...createKeywordAdapter(),
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 0.5),
          feedback: batch.map(() => "measured"),
          usage: batch.map(() => ({ inputTokens: 100, costUsd: 1 })),
        }),
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 1000,
      maxCostUsd: 4,
    });

    expect(result.stopReason).toBe("costExhausted");
    expect(result.usage).toEqual({
      inputTokens: 400,
      outputTokens: 0,
      totalTokens: 400,
      costUsd: 4,
      rollouts: 4,
    });
  });

  test("satisfies the Optimizer contract", async () => {
    const gepa = new GepaOptimizer();
    const contract: Optimizer<GepaStopReason> = gepa;

    const result = await gepa.optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 30,
    });
    const outcome: OptimizerResult<"instruction", GepaStopReason, string> =
      result;

    expect(contract).toBe(gepa);
    expect(outcome.bestScore).toBeGreaterThan(0);
    expect(outcome.bestCandidate.instruction).not.toBe(SEED.instruction);
  });

  test("improves the aggregate score over the seed candidate", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
    });

    const seedRecord = result.candidates[0];

    expect(seedRecord?.aggregateScore).toBe(0);
    expect(result.bestScore).toBeGreaterThan(0);
    expect(result.bestCandidate.instruction).not.toBe(SEED.instruction);
  });

  test("reaches a perfect score on a fully learnable task", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
    });

    expect(result.bestScore).toBe(1);
  });

  test("never spends more than the metric call budget", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 37,
    });

    expect(result.metricCalls).toBeLessThanOrEqual(37);
    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("does not start an iteration it cannot afford to promote", async () => {
    // Screening a child costs two minibatches; promoting it costs a full
    // validation sweep. A guard that only covers the screening spends the tail
    // of the budget on a child it then has to throw away.
    const result = await new GepaOptimizer({
      minibatchSize: 1,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: KEYWORD_EXAMPLES.length + 3,
    });

    expect(result.metricCalls).toBe(KEYWORD_EXAMPLES.length);
    expect(result.candidates).toHaveLength(1);
    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("ends on the snapshot its last checkpoint reported", async () => {
    // types.ts tells callers to hand `result.snapshot` straight back as
    // `resumeFrom`. If it runs ahead of the last checkpoint they persisted,
    // resuming from either one replays work the other already paid for.
    const snapshots: GepaSnapshot[] = [];

    const result = await new GepaOptimizer({
      minibatchSize: 1,
      seed: 1,
      proposals: { perIteration: 3, concurrency: 3 },
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: KEYWORD_EXAMPLES.length + 10,
      onCheckpoint: (snapshot) => {
        snapshots.push(snapshot);
      },
    });

    expect(result.stopReason).toBe("budgetExhausted");
    expect(snapshots.at(-1)).toEqual(result.snapshot);
  });

  test("returns the seed when the budget only covers the seed evaluation", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: KEYWORD_EXAMPLES.length,
    });

    expect(result.candidates).toHaveLength(1);
    expect(result.bestCandidate).toEqual(SEED);
    expect(result.iterations).toBe(0);
  });

  test("records parent lineage for every accepted candidate", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
    });

    const children = result.candidates.filter((record) => record.id !== 0);

    expect(children.length).toBeGreaterThan(0);
    for (const child of children) {
      expect(child.parentIds.length).toBeGreaterThan(0);
      expect(child.source).not.toBe("seed");
    }
  });

  test("keeps one score per validation instance for every candidate", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
    });

    for (const row of result.scoreMatrix) {
      expect(row).toHaveLength(KEYWORD_EXAMPLES.length);
    }
    expect(result.scoreMatrix).toHaveLength(result.candidates.length);
  });

  test("rejects children that do not beat their parent on the minibatch", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: { instruction: "hold ten seconds ticket portal" },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createDegradingReflector(),
      maxMetricCalls: 200,
    });

    expect(result.candidates).toHaveLength(1);
    expect(result.bestCandidate.instruction).toBe(
      "hold ten seconds ticket portal",
    );
  });

  test("emits a start event first and a finish event last", async () => {
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    expect(events[0]?.type).toBe("start");
    expect(events.at(-1)?.type).toBe("finish");
    expect(events.some((event) => event.type === "candidateAccepted")).toBe(
      true,
    );
  });

  test("reports the frontier row an accepted candidate contributed", async () => {
    // An aggregate says a candidate improved. The row says which instances it
    // won and which it paid for, which is the whole basis of Pareto selection
    // and the only view that shows a trade rather than a number.
    const rows = new Map<number, readonly (number | undefined)[]>();

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              rows.set(event.candidateId, event.instanceScores);
            }
          },
        },
      ],
    });

    for (const record of result.candidates) {
      expect(rows.get(record.id)).toEqual(record.instanceScores);
    }
  });

  test("reports the text an accepted candidate was accepted for", async () => {
    const texts = new Map<number, Candidate>();

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              texts.set(event.candidateId, event.candidate);
            }
          },
        },
      ],
    });

    expect(texts.get(result.bestCandidateId)).toEqual(result.bestCandidate);
  });

  test("reports what an accepted candidate produced when outputs are tracked", async () => {
    let outputs: readonly unknown[] | undefined;

    await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
      trackBestOutputs: true,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              outputs = event.outputs;
            }
          },
        },
      ],
    });

    expect(outputs).toHaveLength(KEYWORD_EXAMPLES.length);
  });

  test("omits accepted outputs when the run is not tracking them", async () => {
    let seen = 0;
    let withOutputs = 0;

    await new GepaOptimizer({ minibatchSize: 2, seed: 1 }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              seen += 1;
              withOutputs += event.outputs === undefined ? 0 : 1;
            }
          },
        },
      ],
    });

    expect(seen).toBeGreaterThan(0);
    expect(withOutputs).toBe(0);
  });

  test("gives every reporter the same events", async () => {
    const first: GepaEvent[] = [];
    const second: GepaEvent[] = [];

    await new GepaOptimizer({ minibatchSize: 2, seed: 1 }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        { onEvent: (event) => first.push(event) },
        { onEvent: (event) => second.push(event) },
      ],
    });

    expect(first).toHaveLength(second.length);
    expect(first.map((event) => event.type)).toEqual(
      second.map((event) => event.type),
    );
  });

  test("finishes the search when a reporter throws", async () => {
    // A reporter observes the search. A logging endpoint being down is not a
    // reason for the run that pays for rollouts to fail.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const survivor: GepaEvent[] = [];

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        {
          onEvent: () => {
            throw new Error("logging endpoint is down");
          },
        },
        { onEvent: (event) => survivor.push(event) },
      ],
    });

    expect(result.stopReason).toBe("budgetExhausted");
    expect(survivor.at(-1)?.type).toBe("finish");
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  test("flushes every reporter once the run ends", async () => {
    const flushed: string[] = [];

    await new GepaOptimizer({ minibatchSize: 2, seed: 1 }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        { flush: async () => void flushed.push("first") },
        { flush: async () => void flushed.push("second") },
      ],
    });

    expect(flushed.sort()).toEqual(["first", "second"]);
  });

  test("flushes reporters when the run ends by throwing", async () => {
    // The run that dies mid-search is the one whose buffered events are worth
    // the most, and the one that never reaches its last line.
    let flushed = false;

    await expect(
      new GepaOptimizer({ minibatchSize: 2, seed: 1 }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...createKeywordAdapter(),
          evaluate: () => {
            throw new Error("provider is down");
          },
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        reporters: [
          {
            flush: async () => {
              flushed = true;
            },
          },
        ],
      }),
    ).rejects.toThrow("provider is down");

    expect(flushed).toBe(true);
  });

  test("stops when the abort signal fires", async () => {
    const controller = new AbortController();

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      signal: controller.signal,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              controller.abort();
            }
          },
        },
      ],
    });

    expect(result.stopReason).toBe("aborted");
    expect(result.metricCalls).toBeLessThan(500);
  });

  test("produces identical results for the same seed", async () => {
    const run = () =>
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 7,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 150,
      });

    const first = await run();
    const second = await run();

    expect(second.bestCandidate).toEqual(first.bestCandidate);
    expect(second.metricCalls).toBe(first.metricCalls);
    expect(second.candidates.length).toBe(first.candidates.length);
  });

  test("keeps two runs of one optimizer independent", async () => {
    // The batch sampler holds a shuffle position and the default cache holds
    // scores. Building either once per optimizer instead of once per run would
    // make the second run resume the first one's schedule and read the first
    // one's scores.
    const gepa = new GepaOptimizer({
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
    });
    const run = async () => {
      const adapter = createKeywordAdapter();
      const batches: string[] = [];

      const result = await gepa.optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: (args) => {
            if (args.captureTraces) {
              batches.push(
                args.batch.map((example) => example.question).join("|"),
              );
            }
            return adapter.evaluate(args);
          },
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 400,
      });

      return { batches, metricCalls: result.metricCalls };
    };

    const first = await run();
    const second = await run();

    expect(first.batches.length).toBeGreaterThan(0);
    expect(second.batches).toEqual(first.batches);
    expect(second.metricCalls).toBe(first.metricCalls);
  });

  test("a shared cache makes an identical run cheaper", async () => {
    const cache = createMemoryCache();
    const run = () =>
      new GepaOptimizer({
        maxIterations: 5,
        minibatchSize: 2,
        seed: 7,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 500,
        cache,
      });

    const first = await run();
    const second = await run();

    expect(second.cacheHits).toBeGreaterThan(0);
    expect(second.metricCalls).toBeLessThan(first.metricCalls);
    expect(second.bestCandidate).toEqual(first.bestCandidate);
  });

  test("stops at maxIterations", async () => {
    const result = await new GepaOptimizer({
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
    });

    expect(result.iterations).toBe(3);
    expect(result.stopReason).toBe("maxIterations");
  });

  test("uses a separate validation set when provided", async () => {
    const validationSet = KEYWORD_EXAMPLES.slice(0, 2);

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      validationSet,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 120,
    });

    for (const row of result.scoreMatrix) {
      expect(row).toHaveLength(2);
    }
  });

  test("rotates across components of a multi-component candidate", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
    });

    const updated = new Set(
      result.candidates.flatMap((record) => record.updatedComponents),
    );

    expect(updated).toContain("retriever");
    expect(updated).toContain("writer");
  });

  test("reports a pareto frontier drawn from evaluated candidates", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
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
    const result = await new GepaOptimizer({
      maxIterations: 6,
      minibatchSize: 2,
      seed: 1,
      candidateSelector: ({ state }) =>
        state.aggregateScores.length < 3 ? 0 : state.aggregateScores.length - 1,
      // Four validation instances cannot clear the default overlap floor of
      // five, and this fixture is about what merge does with two complementary
      // lineages rather than about how much evidence it demands first.
      merge: { enabled: true, valOverlapFloor: 3 },
    }).optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 600,
      componentSelector: ({ iteration }) => [
        iteration % 2 === 0 ? "retriever" : "writer",
      ],
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
    const withMerge = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
      merge: { enabled: true },
    }).optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });
    const withoutMerge = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
      merge: { enabled: false },
    }).optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });

    // Enabling merge must not cost the run a meaningful slice of its budget.
    expect(withMerge.metricCalls).toBeGreaterThanOrEqual(
      withoutMerge.metricCalls - KEYWORD_EXAMPLES.length,
    );
  });

  test("does not merge when merging is disabled", async () => {
    const result = await new GepaOptimizer({
      maxIterations: 6,
      minibatchSize: 2,
      seed: 1,
      candidateSelector: ({ state }) =>
        state.aggregateScores.length < 3 ? 0 : state.aggregateScores.length - 1,
      merge: { enabled: false },
    }).optimize({
      seedCandidate: { retriever: "Find documents.", writer: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 600,
      componentSelector: ({ cursor }) => [
        cursor % 2 === 0 ? "retriever" : "writer",
      ],
    });

    expect(result.candidates.every((record) => record.source !== "merge")).toBe(
      true,
    );
  });

  test("stops testing merges once maxInvocations is reached", async () => {
    // Every acceptance schedules a merge, so a run that keeps improving builds
    // a backlog. The cap has to be checked where merges are triggered, not
    // only where they are scheduled, or the backlog spends straight past it.
    const result = await new GepaOptimizer({
      ...mergeRunwayConfig(),
      merge: { enabled: true, maxInvocations: 1 },
    }).optimize(mergeRunwayTask());

    expect(
      result.candidates.filter((record) => record.source === "merge"),
    ).toHaveLength(1);
  });

  test("keeps merging while the invocation cap allows it", async () => {
    const result = await new GepaOptimizer({
      ...mergeRunwayConfig(),
      merge: { enabled: true, maxInvocations: 5 },
    }).optimize(mergeRunwayTask());

    expect(
      result.candidates.filter((record) => record.source === "merge").length,
    ).toBeGreaterThan(1);
  });

  test("announces a merge iteration the way it announces a mutation", async () => {
    // Without an iterationStart, an event consumer watching a merge iteration
    // sees evaluations and an acceptance appear from nowhere.
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      ...mergeRunwayConfig(),
      merge: { enabled: true },
    }).optimize({
      ...mergeRunwayTask(),
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    const merged = events.flatMap((event) =>
      event.type === "candidateAccepted" && event.source === "merge"
        ? [event.iteration]
        : [],
    );
    const started = events.flatMap((event) =>
      event.type === "iterationStart" ? [event.iteration] : [],
    );

    expect(merged.length).toBeGreaterThan(0);
    for (const iteration of merged) {
      expect(started).toContain(iteration);
    }
  });

  test("returns to mutation after a merge is accepted", async () => {
    // A merge is worth attempting because a *mutation* put something new on
    // the frontier. An accepted merge is not that: it recombines what two
    // lineages already had, so the reference clears the flag and goes back to
    // mutating rather than merging the merge.
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      ...mergeRunwayConfig(),
      merge: { enabled: true },
    }).optimize({
      ...mergeRunwayTask(),
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    const merged = events.flatMap((event) =>
      event.type === "candidateAccepted" && event.source === "merge"
        ? [event.iteration]
        : [],
    );

    expect(merged.length).toBeGreaterThan(0);
    for (const iteration of merged) {
      expect(merged).not.toContain(iteration + 1);
    }
  });

  test("rejects an empty validation set", async () => {
    await expect(
      new GepaOptimizer({
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        validationSet: [],
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
      }),
    ).rejects.toThrow(/validationSet/i);
  });

  test("rejects a minibatch size below one", () => {
    // An empty minibatch is vacuously perfect, so every iteration skips
    // straight past reflection and charges nothing: with the default
    // maxIterations the run never terminates and never spends.
    expect(() => new GepaOptimizer({ minibatchSize: 0, seed: 1 })).toThrow(
      /minibatchSize/,
    );
  });

  test("rejects a non-finite perfect score", () => {
    expect(
      () => new GepaOptimizer({ perfectScore: Number.NaN, seed: 1 }),
    ).toThrow(/perfectScore/);
  });

  test("rejects a negative rejected proposal memory", () => {
    expect(
      () => new GepaOptimizer({ rejectedProposalMemory: -1, seed: 1 }),
    ).toThrow(/rejectedProposalMemory/);
  });

  test("rejects a fractional iteration ceiling", () => {
    expect(() => new GepaOptimizer({ maxIterations: 1.5, seed: 1 })).toThrow(
      /maxIterations/,
    );
  });

  test("warns that selection reused the instances reflection read", async () => {
    // The reflection prompt mines domain facts out of the traces it is shown.
    // With no validationSet those traces are the instances that then pick the
    // winner, so the facts are memorised and scored on the same rows.
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      maxIterations: 1,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });

    expect(result.warnings.map((warning) => warning.code)).toContain(
      "validationSetReusesTraining",
    );
  });

  test("stays quiet when the caller named the reuse", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      maxIterations: 1,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      validationSet: "reuseTraining",
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });

    expect(result.warnings.map((warning) => warning.code)).not.toContain(
      "validationSetReusesTraining",
    );
  });

  test("carries the warnings on the finish event a reporter reads", async () => {
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      minibatchSize: 2,
      maxIterations: 1,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    const finish = events.find((event) => event.type === "finish");

    expect(finish?.type === "finish" && finish.warnings).toContainEqual(
      expect.objectContaining({ code: "validationSetReusesTraining" }),
    );
  });

  test("warns when the seed already scores perfectly on every instance", async () => {
    // Nothing left to improve: every proposal ties, and acceptance resolves
    // ties by whatever the metric's noise did that iteration.
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      maxIterations: 1,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      validationSet: "reuseTraining",
      adapter: {
        ...createKeywordAdapter(),
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 1),
          feedback: batch.map(() => "perfect"),
        }),
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });

    expect(result.warnings.map((warning) => warning.code)).toEqual([
      "seedScoreSaturated",
    ]);
  });

  test("warns when the seed scores zero on every instance", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      maxIterations: 1,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      validationSet: "reuseTraining",
      adapter: {
        ...createKeywordAdapter(),
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 0),
          feedback: batch.map(() => "nothing landed"),
        }),
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });

    expect(result.warnings.map((warning) => warning.code)).toEqual([
      "seedScoreFloored",
    ]);
  });

  test("rejects a significance bar the minibatch is too small to ever clear", () => {
    // A sign-flip test over three instances bottoms out at p = 0.125, so at
    // alpha 0.05 every proposal is rejected on arithmetic. The run spends its
    // whole budget and returns the seed, which is indistinguishable from a
    // search that genuinely found nothing.
    expect(
      () =>
        new GepaOptimizer({
          minibatchSize: 3,
          acceptance: pairedPermutationAcceptance({ alpha: 0.05 }),
        }),
    ).toThrow(/minibatchSize/);
  });

  test("allows a minibatch exactly wide enough for the bar", () => {
    expect(
      () =>
        new GepaOptimizer({
          minibatchSize: 5,
          acceptance: pairedPermutationAcceptance({ alpha: 0.05 }),
        }),
    ).not.toThrow();
  });

  test("scores validation instances separately from trainingSet instances sharing an id", async () => {
    // Positional instance ids are a reasonable thing for a user to write, and
    // they make train instance "0" and val instance "0" collide in the cache.
    type Instance = { split: "train" | "val" };
    const trainingSet: Instance[] = [{ split: "train" }, { split: "train" }];
    const validationSet: Instance[] = [{ split: "val" }];
    let revision = 0;

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet,
      validationSet,
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
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
        raiseOnError: false,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
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
      }),
    ).rejects.toThrow("tracing backend down");
  });

  test("tolerates a failure that arrives after evaluations succeeded", async () => {
    const adapter = createKeywordAdapter();
    let errorEvents = 0;

    const result = await new GepaOptimizer({
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
      raiseOnError: false,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: {
        ...adapter,
        makeReflectiveDataset: () => {
          throw new Error("reflective dataset failed");
        },
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "error") {
              errorEvents += 1;
            }
          },
        },
      ],
    });

    expect(errorEvents).toBe(3);
    expect(result.stopReason).toBe("maxIterations");
    // Seed validation plus one parent minibatch per iteration — a failed
    // evaluation is never charged.
    expect(result.metricCalls).toBe(KEYWORD_EXAMPLES.length + 3 * 2);
  });

  test("advances a component cursor per candidate rather than per iteration", async () => {
    const result = await new GepaOptimizer({
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
      candidateSelector: () => 0,
    }).optimize({
      seedCandidate: { alpha: "a", beta: "b", gamma: "c" },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createDegradingReflector(),
      maxMetricCalls: 400,
    });

    // The seed was the parent twice, so its own cursor advanced twice —
    // independently of which global iteration each selection happened on.
    expect(result.candidates[0]?.componentCursor).toBe(2);
  });

  test("skips reflection when the parent minibatch is already perfect", async () => {
    let reflectCalls = 0;

    await new GepaOptimizer({
      maxIterations: 5,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
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
    });

    expect(reflectCalls).toBe(0);
  });

  test("does not cache a score the adapter marked transient", async () => {
    // A rate limit is not the candidate's fault. Caching the zero it produced
    // would pin that instance to zero for every later evaluation of the same
    // text, with no rollout ever attempted again.
    const cache = createMemoryCache();
    const run = () =>
      new GepaOptimizer({
        maxIterations: 2,
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
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
        cache,
        // The subject here is what happens to a row that stays transient, not
        // the retrying that usually rescues one.
        retry: { attempts: 0 },
      });

    await run();
    const second = await run();

    expect(second.cacheHits).toBe(0);
  });

  test("leaves a validation instance unscored when its score was transient", async () => {
    // Skipping the cache is not enough: nothing ever re-measures a promoted
    // candidate, so a transient zero written into the record is permanent, and
    // it drops the candidate off that instance's front for the rest of the run.
    const adapter = createKeywordAdapter();

    const result = await new GepaOptimizer({
      maxIterations: 1,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: {
        instruction:
          "hold ten seconds ticket portal thirty days billing prorated",
      },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: {
        ...adapter,
        evaluate: async (args) => {
          const evaluation = await adapter.evaluate(args);
          if (args.run.phase !== "seed") {
            return evaluation;
          }
          return {
            ...evaluation,
            scores: evaluation.scores.map((score, index) =>
              index === 2 ? 0 : score,
            ),
            transient: evaluation.scores.map((_, index) => index === 2),
          };
        },
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
      retry: { attempts: 0 },
    });

    const seedRecord = result.candidates[0];

    expect(seedRecord?.instanceScores).toEqual([1, 1, undefined, 1]);
    expect(seedRecord?.aggregateScore).toBe(1);
  });

  test("caches a score the adapter did not mark transient", async () => {
    const cache = createMemoryCache();
    const run = () =>
      new GepaOptimizer({
        maxIterations: 2,
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
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
        cache,
        // The subject here is what happens to a row that stays transient, not
        // the retrying that usually rescues one.
        retry: { attempts: 0 },
      });

    await run();
    const second = await run();

    expect(second.cacheHits).toBeGreaterThan(0);
  });

  test("screens a proposal on the instances both it and its parent measured", async () => {
    // Screening compares two rollout sets over the same instances. An instance
    // that failed transiently on one side was never measured on that side, so
    // scoring the pair on it compares a candidate against an outage — here it
    // hides a child that is better everywhere it actually ran.
    const rows = [{ id: 0 }, { id: 1 }];

    const result = await new GepaOptimizer({
      maxIterations: 1,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: rows,
      adapter: {
        evaluate: ({ batch, candidate }) => {
          const isSeed = candidate.instruction === SEED.instruction;
          return {
            outputs: batch.map(() => ""),
            scores: batch.map((row) => (isSeed ? 1 - row.id : row.id)),
            feedback: batch.map(() => "measured"),
            // The child's first instance never ran; the seed's did.
            transient: batch.map((row) => !isSeed && row.id === 0),
          };
        },
        makeReflectiveDataset: ({ batch, evaluation }) => ({
          instruction: batch.map((row, index) => ({
            inputs: { id: row.id },
            generatedOutputs: "",
            feedback: evaluation.feedback?.[index] ?? "",
            score: evaluation.scores[index] as number,
          })),
        }),
      },
      reflect: async () => "```\nrewritten instruction\n```",
      maxMetricCalls: 100,
      retry: { attempts: 0 },
    });

    expect(result.candidates).toHaveLength(2);
  });

  test("keeps scores measured under different environments apart", async () => {
    // Nothing about a cached score records which model produced it. Swapping
    // the task model, its temperature, or the scorer leaves the same key
    // pointing at a measurement of a system that no longer exists.
    const cache = createMemoryCache();
    const run = (environment: string) =>
      new GepaOptimizer({
        maxIterations: 1,
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        cache,
        cacheNamespace: environment,
      });

    await run("gpt-5-mini@0.0");
    const second = await run("gpt-5@0.7");

    expect(second.cacheHits).toBe(0);
  });

  test("rejects a component selector that names a component the candidate lacks", async () => {
    // The patch is merged over the parent, so an unknown name does not fail —
    // it grows the candidate a component the system under optimization never
    // reads, and every descendant carries it.
    //
    // Annotating the seed as `Candidate` widens the component names back to
    // `string`, which is what a JavaScript caller and a dynamically assembled
    // candidate both get. The typo is a compile error without it, and the
    // runtime guard is what covers everyone who lost the union.
    const untyped: Candidate = SEED;

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: untyped,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        componentSelector: () => ["instrution"],
      }),
    ).rejects.toThrow(/instrution/);
  });

  test("rejects a proposal that names a component the candidate lacks", async () => {
    const adapter = createKeywordAdapter();
    const untyped: Candidate = SEED;

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: untyped,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          proposeNewTexts: () => ({ instrution: "hold ten seconds" }),
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
      }),
    ).rejects.toThrow(/instrution/);
  });

  test("rejects a non-finite score from the adapter", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
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
      }),
    ).rejects.toThrow(/finite/i);
  });

  test("names the offending instance when a score is not a number", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
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
      }),
    ).rejects.toThrow(/index 1/i);
  });

  test("rejects feedback that does not align with the batch", async () => {
    // Feedback is read positionally into the reflective dataset, so a short
    // array silently attributes one instance's diagnosis to another.
    const adapter = createKeywordAdapter();

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: ({ batch }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0),
            feedback: ["only one"],
          }),
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
      }),
    ).rejects.toThrow(/feedback/i);
  });

  test("rejects objective scores that do not align with the batch", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: ({ batch }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0),
            feedback: batch.map(() => ""),
            objectiveScores: [{ coverage: 1 }],
          }),
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
      }),
    ).rejects.toThrow(/objectiveScores/i);
  });

  test("rejects outputs that do not align with the batch", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: ({ batch }) => ({
            outputs: [""],
            scores: batch.map(() => 0),
            feedback: batch.map(() => ""),
          }),
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
      }),
    ).rejects.toThrow(/outputs/i);
  });

  test("rejects transient flags that do not align with the batch", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: ({ batch }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0),
            feedback: batch.map(() => ""),
            transient: [true],
          }),
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
      }),
    ).rejects.toThrow(/transient/i);
  });

  test("stops cleanly when an adapter aborts mid-iteration", async () => {
    // An adapter that honours the signal throws rather than returning zeros.
    // That is a cancellation, not a run failure, so it must end the run
    // cleanly instead of rejecting the way any other adapter error would.
    const controller = new AbortController();
    const adapter = createKeywordAdapter();

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
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
      signal: controller.signal,
    });

    expect(result.stopReason).toBe("aborted");
    expect(result.metricCalls).toBe(KEYWORD_EXAMPLES.length);
  });

  test("propagates a failure during seed evaluation", async () => {
    const adapter = createKeywordAdapter();

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: {
          ...adapter,
          evaluate: () => {
            throw new Error("adapter exploded");
          },
        },
        reflect: createKeywordReflector(),
        maxMetricCalls: 50,
      }),
    ).rejects.toThrow("adapter exploded");
  });

  test("aggregates objective scores over the validation set", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createObjectiveAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
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
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createObjectiveAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
    });

    const coverage = result.perObjectiveBest?.coverage;
    const brevity = result.perObjectiveBest?.brevity;

    expect(coverage?.score).toBe(result.bestScore);
    expect(coverage?.candidateIds).toContain(result.bestCandidateId);
    // The seed says nothing, so nothing is shorter than it.
    expect(brevity?.candidateIds).toContain(0);
  });

  test("omits an objective only some instances reported", async () => {
    // Averaged over the instances that reported it, an objective seen once
    // competes on the objective frontier against one seen everywhere, as
    // though the two numbers meant the same thing.
    const adapter = createKeywordAdapter();

    const result = await new GepaOptimizer({
      maxIterations: 1,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: {
        ...adapter,
        evaluate: async (args) => {
          const evaluation = await adapter.evaluate(args);
          return {
            ...evaluation,
            objectiveScores: evaluation.scores.map(
              (score, index): Record<string, number> =>
                index === 0
                  ? { coverage: score, speed: 1 }
                  : { coverage: score },
            ),
          };
        },
      },
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
    });

    expect(Object.keys(result.candidates[0]?.objectiveScores ?? {})).toEqual([
      "coverage",
    ]);
  });

  test("leaves objective reporting out when the adapter scores no objectives", async () => {
    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
    });

    expect(result.perObjectiveBest).toBeUndefined();
    expect(result.candidates[0]?.objectiveScores).toBeUndefined();
  });

  test("reuses cached objective scores instead of re-evaluating", async () => {
    const cache = createMemoryCache();
    const gepa = new GepaOptimizer({
      maxIterations: 5,
      minibatchSize: 2,
      seed: 1,
    });
    const task = {
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createObjectiveAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      cache,
    };

    const first = await gepa.optimize(task);
    const second = await gepa.optimize(task);

    expect(second.metricCalls).toBeLessThan(first.metricCalls);
    expect(second.candidates[0]?.objectiveScores).toEqual(
      first.candidates[0]?.objectiveScores,
    );
  });

  test("scores only the validation instances the evaluation policy selects", async () => {
    const result = await new GepaOptimizer({
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
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
    const events: GepaEvent[] = [];

    const result = await new GepaOptimizer({
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 500,
      valEvaluationPolicy: subsampledEvaluationPolicy({ size: 2 }),
      reporters: [{ onEvent: (event) => events.push(event) }],
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

    await new GepaOptimizer({
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: { instruction: "hold ten seconds ticket portal" },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nno useful information\n```";
      },
      maxMetricCalls: 200,
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

    await new GepaOptimizer({
      maxIterations: 8,
      minibatchSize: 2,
      seed: 1,
      rejectedProposalMemory: 2,
    }).optimize({
      seedCandidate: { instruction: "hold ten seconds ticket portal" },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        counter += 1;
        return `\`\`\`\nuseless proposal ${counter}\n\`\`\``;
      },
      maxMetricCalls: 400,
    });

    const last = prompts.at(-1) ?? "";

    expect(prompts.length).toBeGreaterThan(3);
    expect(last).toContain("useless proposal");
    expect(last.match(/useless proposal/g) ?? []).toHaveLength(2);
  });

  test("emits a checkpoint for the seed and for every iteration", async () => {
    const snapshots: GepaSnapshot[] = [];

    const result = await new GepaOptimizer({
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
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
    let snapshot: GepaSnapshot | undefined;

    await new GepaOptimizer({
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 200,
      onCheckpoint: (taken) => {
        snapshot = taken;
      },
    });

    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  test("resumes where the checkpoint left off", async () => {
    const interrupted = await new GepaOptimizer({
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      cache: false,
    });
    const snapshot = interrupted.snapshot;

    const resumed = await new GepaOptimizer({
      maxIterations: 6,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
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
    const interrupted = await new GepaOptimizer({
      maxIterations: 2,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
      cache: false,
    });

    const resumed = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.metricCalls).toBeLessThanOrEqual(60);
    expect(resumed.metricCalls).toBeGreaterThanOrEqual(interrupted.metricCalls);
    expect(resumed.stopReason).toBe("budgetExhausted");
  });

  test("carries usage already spent into a resumed run", async () => {
    // `maxCostUsd` is a ceiling on the run, not on the segment. A resumed run
    // that restarts its dollar accounting at zero lets an interrupted-and-
    // resumed loop spend the ceiling over and over.
    const keyword = createKeywordAdapter();
    const adapter = {
      ...keyword,
      evaluate: async (args: Parameters<typeof keyword.evaluate>[0]) => {
        const evaluation = await keyword.evaluate(args);
        return {
          ...evaluation,
          usage: args.batch.map(() => ({ inputTokens: 10, outputTokens: 5 })),
        };
      },
    };
    const task = {
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter,
      reflect: createKeywordReflector(),
      maxMetricCalls: 30,
      cache: false as const,
    };

    const interrupted = await new GepaOptimizer({
      minibatchSize: 2,
      maxIterations: 1,
      seed: 1,
    }).optimize(task);

    const resumed = await new GepaOptimizer({
      minibatchSize: 2,
      maxIterations: 2,
      seed: 1,
    }).optimize({ ...task, resumeFrom: interrupted.snapshot });

    expect(interrupted.usage.inputTokens).toBeGreaterThan(0);
    expect(resumed.usage.inputTokens).toBeGreaterThanOrEqual(
      interrupted.usage.inputTokens,
    );
  });

  test("carries the evaluation cache in the checkpoint", async () => {
    const gepa = new GepaOptimizer({
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
    });
    const task = {
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
    };

    const first = await gepa.optimize({ ...task, cache: createMemoryCache() });
    const second = await gepa.optimize({
      ...task,
      cache: createMemoryCache({ entries: first.snapshot.cache }),
    });

    expect(first.snapshot.cache?.length).toBeGreaterThan(0);
    expect(second.cacheHits).toBeGreaterThan(0);
    expect(second.metricCalls).toBeLessThan(first.metricCalls);
  });

  test("restores the checkpoint's cache into a resumed run", async () => {
    const interrupted = await new GepaOptimizer({
      maxIterations: 3,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
    });
    const cache = createMemoryCache();

    await new GepaOptimizer({
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 400,
      cache,
      resumeFrom: interrupted.snapshot,
    });

    const restored = new Map(cache.entries?.());

    for (const [key, cached] of interrupted.snapshot.cache ?? []) {
      expect(restored.get(key)).toEqual(cached);
    }
  });

  test("does not mutate the snapshot it resumes from", async () => {
    const interrupted = await new GepaOptimizer(
      resumeConfig({ maxIterations: 2 }),
    ).optimize(resumeTask());
    const pristine = JSON.parse(JSON.stringify(interrupted.snapshot));

    await new GepaOptimizer(resumeConfig({ maxIterations: 4 })).optimize({
      ...resumeTask(),
      resumeFrom: interrupted.snapshot,
    });

    // The caller persisted this object. A run that writes component cursors
    // and rejected proposals straight back into it corrupts every later resume.
    expect(interrupted.snapshot).toEqual(pristine);
  });

  test("resumes twice from one snapshot to the same result", async () => {
    const interrupted = await new GepaOptimizer(
      resumeConfig({ maxIterations: 2 }),
    ).optimize(resumeTask());

    const first = await new GepaOptimizer(
      resumeConfig({ maxIterations: 4 }),
    ).optimize({
      ...resumeTask(),
      resumeFrom: interrupted.snapshot,
    });
    const second = await new GepaOptimizer(
      resumeConfig({ maxIterations: 4 }),
    ).optimize({
      ...resumeTask(),
      resumeFrom: interrupted.snapshot,
    });

    expect(second.metricCalls).toBe(first.metricCalls);
    expect(second.candidates).toEqual(first.candidates);
  });

  test("refuses a checkpoint taken against a different seed candidate", async () => {
    const other = await new GepaOptimizer({
      maxIterations: 1,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: { instruction: "Something else entirely." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
    });

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        resumeFrom: other.snapshot,
      }),
    ).rejects.toThrow(/checkpoint/i);
  });

  test("refuses a checkpoint taken against a different validation set", async () => {
    const other = await new GepaOptimizer({
      maxIterations: 1,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      validationSet: KEYWORD_EXAMPLES.slice(0, 2),
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 100,
    });

    await expect(
      new GepaOptimizer({
        minibatchSize: 2,
        seed: 1,
      }).optimize({
        seedCandidate: SEED,
        trainingSet: KEYWORD_EXAMPLES,
        adapter: createKeywordAdapter(),
        reflect: createKeywordReflector(),
        maxMetricCalls: 100,
        resumeFrom: other.snapshot,
      }),
    ).rejects.toThrow(/checkpoint/i);
  });

  test("reports the best candidate chosen by the evaluation policy", async () => {
    const result = await new GepaOptimizer({
      maxIterations: 4,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: KEYWORD_EXAMPLES,
      adapter: createKeywordAdapter(),
      reflect: createKeywordReflector(),
      maxMetricCalls: 300,
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
  const config = {
    maxIterations: 4,
    minibatchSize: 2,
    seed: 1,
  };
  const task = {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 2000,
  };

  test("makes one proposal per iteration by default", async () => {
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      ...config,
    }).optimize({
      ...task,
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    expect(countProposals(events, 0)).toBe(1);
  });

  test("makes several proposals in one iteration when asked", async () => {
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 3 },
    }).optimize({
      ...task,
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    expect(countProposals(events, 0)).toBe(3);
  });

  test("draws a different minibatch for each proposal in an iteration", async () => {
    const batches: string[] = [];

    await new GepaOptimizer({
      ...config,
      maxIterations: 1,
      proposals: { perIteration: 2 },
    }).optimize({
      ...task,
      adapter: {
        ...task.adapter,
        evaluate: (args) => {
          if (args.captureTraces) {
            batches.push(
              args.batch.map((example) => example.question).join("|"),
            );
          }
          return task.adapter.evaluate(args);
        },
      },
    });

    expect(batches).toHaveLength(2);
    expect(batches[0]).not.toBe(batches[1]);
  });

  test("accepts at most one candidate per iteration under best selection", async () => {
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 3, selection: "best" },
    }).optimize({
      ...task,
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    const accepted = events
      .filter((event) => event.type === "candidateAccepted")
      .filter((event) => event.source !== "seed");
    const iterations = accepted.map((event) => event.iteration);

    expect(accepted.length).toBeGreaterThan(0);
    expect(new Set(iterations).size).toBe(iterations.length);
  });

  test("accepts several candidates in one iteration under all selection", async () => {
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 3, selection: "all" },
    }).optimize({
      ...task,
      reporters: [{ onEvent: (event) => events.push(event) }],
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
    const events: GepaEvent[] = [];

    await new GepaOptimizer({
      ...config,
      maxIterations: 1,
      proposals: { perIteration: 3, selection: "best" },
    }).optimize({
      ...task,
      reporters: [{ onEvent: (event) => events.push(event) }],
    });

    const accepted = events
      .filter((event) => event.type === "candidateAccepted")
      .filter((event) => event.source !== "seed");
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

    await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 3, concurrency: 3 },
    }).optimize({
      ...task,
      adapter: tracked.adapter,
    });

    expect(tracked.maxInFlight()).toBeGreaterThan(1);
  });

  test("runs proposals one at a time by default", async () => {
    const tracked = withOverlapTracking(createKeywordAdapter());

    await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 3 },
    }).optimize({
      ...task,
      adapter: tracked.adapter,
    });

    expect(tracked.maxInFlight()).toBe(1);
  });

  test("reaches the same candidates whether or not proposals overlap", async () => {
    const serial = await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 3, concurrency: 1 },
    }).optimize({
      ...task,
      adapter: createKeywordAdapter(),
    });
    const concurrent = await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 3, concurrency: 3 },
    }).optimize({
      ...task,
      adapter: createKeywordAdapter(),
    });

    expect(concurrent.candidates.map((record) => record.candidate)).toEqual(
      serial.candidates.map((record) => record.candidate),
    );
    expect(concurrent.bestScore).toBe(serial.bestScore);
  });

  test("never spends past the budget when proposals overlap", async () => {
    const result = await new GepaOptimizer({
      ...config,
      maxIterations: 50,
      proposals: { perIteration: 4, concurrency: 4 },
    }).optimize({
      ...task,
      maxMetricCalls: 43,
    });

    expect(result.metricCalls).toBeLessThanOrEqual(43);
  });

  test("reaches the same lineage whichever order overlapping proposals finish in", async () => {
    // Siblings in one iteration routinely converge on the same child text. Which
    // of them the engine screens must be decided in plan order, not by whichever
    // reflection call happened to return first — otherwise the surviving
    // outcome carries a different parent, minibatch and parent score, and the
    // seeded run stops being reproducible.
    const run = (pace: (prompt: string) => number) => {
      // Two children of the seed that differ in text but not in score, so the
      // next iteration reflects on both and proposes the same replacement.
      const parents = [0, 0, 1, 2];
      let pick = -1;

      return new GepaOptimizer({
        maxIterations: 2,
        minibatchSize: 2,
        seed: 1,
        candidateSelector: () => {
          pick += 1;
          return parents[pick] ?? 0;
        },
        proposals: { perIteration: 2, concurrency: 2 },
      }).optimize({
        seedCandidate: { instruction: "" },
        trainingSet: MARK_EXAMPLES,
        adapter: createMarkAdapter(),
        reflect: createMarkReflector(pace),
        maxMetricCalls: 1000,
        // Both siblings of the colliding iteration diagnose the same failures.
        batchSampler: ({ iteration }) =>
          iteration < 2 ? [iteration, 2] : [0, 1],
      });
    };

    const firstSiblingLast = await run((prompt) =>
      prompt.includes("+1-3") ? 3 : 0,
    );
    const firstSiblingFirst = await run((prompt) =>
      prompt.includes("+1-3") ? 0 : 3,
    );

    expect(lineageOf(firstSiblingLast)).toEqual(lineageOf(firstSiblingFirst));
    expect(firstSiblingLast.metricCalls).toBe(firstSiblingFirst.metricCalls);
  });

  test("never announces a validation id it does not go on to record", async () => {
    // Sweeps are scheduled with the id each candidate will be recorded under.
    // A sweep that runs and is paid for, and is then thrown away to keep the
    // remaining ids aligned, costs the run rollouts and breaks that contract.
    const adapter = createKeywordAdapter();
    const contexts: EvaluationContext[] = [];
    let selections = -1;

    const result = await new GepaOptimizer({
      ...config,
      maxIterations: 10,
      minibatchSize: 1,
      proposals: { perIteration: 3, concurrency: 3 },
    }).optimize({
      ...task,
      adapter: {
        ...adapter,
        evaluate: (args) => {
          contexts.push(args.run);
          return adapter.evaluate(args);
        },
      },
      maxMetricCalls: KEYWORD_EXAMPLES.length + 10,
      cache: false,
      // Later survivors are cheaper to sweep than earlier ones, so a survivor
      // can still be affordable after the one before it was not.
      valEvaluationPolicy: {
        selectInstances: () => {
          selections += 1;
          return selections % 2 === 0 ? [0, 1, 2, 3] : [0];
        },
        bestCandidate: fullEvaluationPolicy().bestCandidate,
      },
    });

    const announced = contexts
      .filter((context) => context.phase === "validation")
      .map((context) => context.candidateId);
    const recorded = result.candidates.map((record) => record.id);

    expect(announced.length).toBeGreaterThan(0);
    for (const candidateId of announced) {
      expect(recorded).toContain(candidateId);
    }
  });

  test("rejects a proposal count below one", () => {
    expect(
      () => new GepaOptimizer({ ...config, proposals: { perIteration: 0 } }),
    ).toThrow(/perIteration/);
  });
});

describe("optimize reflection budget", () => {
  const config = {
    maxIterations: 20,
    minibatchSize: 2,
    seed: 1,
  };
  const task = {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 2000,
  };

  test("counts the reflection calls a run made", async () => {
    const result = await new GepaOptimizer({
      ...config,
      maxIterations: 3,
    }).optimize({
      ...task,
    });

    expect(result.reflectionCalls).toBeGreaterThan(0);
  });

  test("stops once the reflection call budget is spent", async () => {
    const result = await new GepaOptimizer({
      ...config,
      reflection: { maxCalls: 2 },
    }).optimize({
      ...task,
    });

    expect(result.reflectionCalls).toBe(2);
    expect(result.stopReason).toBe("reflectionBudgetExhausted");
  });

  test("never exceeds the reflection budget with overlapping proposals", async () => {
    const result = await new GepaOptimizer({
      ...config,
      proposals: { perIteration: 4, concurrency: 4 },
      reflection: { maxCalls: 3 },
    }).optimize({
      ...task,
    });

    expect(result.reflectionCalls).toBe(3);
  });

  test("carries reflection spend across a resumed run", async () => {
    // A reflector that never improves anything keeps every iteration
    // reflecting, so the count is a clean measure of what each run spent.
    const persistentTask = { ...task, reflect: createDegradingReflector() };
    const interrupted = await new GepaOptimizer({
      ...config,
      maxIterations: 2,
    }).optimize({
      ...persistentTask,
    });

    const resumed = await new GepaOptimizer({
      ...config,
      maxIterations: 4,
    }).optimize({
      ...persistentTask,
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

    await new GepaOptimizer({
      ...config,
      maxIterations: 2,
      minibatchSize: 4,
      reflection: { maxRecords: 1 },
    }).optimize({
      ...task,
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

    await new GepaOptimizer({
      ...config,
      maxIterations: 1,
      reflection: {
        buildPrompt: ({ componentName, currentText }) =>
          `improve ${componentName}: ${currentText}`,
      },
    }).optimize({
      ...task,
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nhold ten seconds\n```";
      },
    });

    expect(prompts[0]).toBe("improve instruction: Answer the user question.");
  });
});

describe("optimize outputs", () => {
  const config = {
    maxIterations: 4,
    minibatchSize: 2,
    seed: 1,
  };
  const task = {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 400,
  };

  test("omits validation outputs unless tracking is on", async () => {
    const result = await new GepaOptimizer(config).optimize(task);

    expect(result.bestOutputs).toBeUndefined();
  });

  test("returns the best candidate's output for every validation instance", async () => {
    const result = await new GepaOptimizer({
      ...config,
      trackBestOutputs: true,
    }).optimize({
      ...task,
    });

    expect(result.bestOutputs).toHaveLength(KEYWORD_EXAMPLES.length);
    for (const output of result.bestOutputs ?? []) {
      expect(output).toContain(result.bestCandidate.instruction);
    }
  });
});

/**
 * Selection pressure is applied to the validation set for the whole run, so the
 * winner's score on it is partly fitted to those instances. The held-out sweep
 * is the only number in a result that no candidate was ever selected against.
 */
describe("optimize held-out evaluation", () => {
  const config = { maxIterations: 4, minibatchSize: 2, seed: 1 };
  const task = {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 400,
  };
  // The keyword reflector only ever appends, so a term present in the seed is
  // present in every descendant: one instance always scores 1, one always 0.
  const TESTSET = [
    { question: "held out, satisfied", required: ["answer"] },
    { question: "held out, unsatisfiable", required: ["zzz-never-proposed"] },
  ];

  test("omits the held-out score when no testSet is given", async () => {
    const result = await new GepaOptimizer(config).optimize(task);

    expect(result.testScore).toBeUndefined();
    expect(result.testMetricCalls).toBeUndefined();
  });

  test("scores the best candidate on instances the search never saw", async () => {
    const result = await new GepaOptimizer(config).optimize({
      ...task,
      testSet: TESTSET,
    });

    expect(result.testScore).toBe(0.5);
  });

  test("keeps the held-out sweep out of the search budget", async () => {
    const searchOnly = await new GepaOptimizer(config).optimize(task);
    const withHoldout = await new GepaOptimizer(config).optimize({
      ...task,
      testSet: TESTSET,
    });

    expect(withHoldout.metricCalls).toBe(searchOnly.metricCalls);
    expect(withHoldout.testMetricCalls).toBe(TESTSET.length);
  });

  test("keeps the held-out sweep out of the run's usage", async () => {
    // No ceiling bounds the held-out sweep: it runs once the search has already
    // stopped. Counting it in `usage` would describe a run that honoured
    // `maxCostUsd` as having overrun it.
    const keyword = createKeywordAdapter();
    const result = await new GepaOptimizer(config).optimize({
      ...task,
      adapter: {
        ...keyword,
        evaluate: async (args: Parameters<typeof keyword.evaluate>[0]) => ({
          ...(await keyword.evaluate(args)),
          usage: args.batch.map(() => ({ inputTokens: 10, outputTokens: 5 })),
        }),
      },
      testSet: TESTSET,
    });

    expect(result.usage.rollouts).toBe(result.metricCalls);
    expect(result.testUsage).toEqual({
      inputTokens: 20,
      outputTokens: 10,
      totalTokens: 30,
      costUsd: 0,
      rollouts: 2,
    });
  });
  test("never draws a held-out instance into training or validation", async () => {
    const seen: string[] = [];
    const adapter = createKeywordAdapter();

    await new GepaOptimizer(config).optimize({
      ...task,
      adapter: {
        ...adapter,
        evaluate: (args) => {
          if (args.run.split !== "test") {
            seen.push(...args.batch.map((datum) => datum.question));
          }
          return adapter.evaluate(args);
        },
      },
      testSet: TESTSET,
    });

    for (const question of TESTSET.map((datum) => datum.question)) {
      expect(seen).not.toContain(question);
    }
  });

  test("reports the held-out sweep as its own evaluation phase", async () => {
    const phases: EvaluationContext[] = [];

    await new GepaOptimizer(config).optimize({
      ...task,
      testSet: TESTSET,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "evaluation" && event.phase === "test") {
              phases.push({
                iteration: event.iteration,
                phase: event.phase,
                split: "test",
                candidateId: event.candidateId,
              });
            }
          },
        },
      ],
    });

    expect(phases).toHaveLength(1);
    expect(phases[0]?.candidateId).not.toBeNull();
  });

  test("reports the held-out score on the finish event", async () => {
    let finished: { testScore?: number } | undefined;

    const result = await new GepaOptimizer(config).optimize({
      ...task,
      testSet: TESTSET,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              finished = event;
            }
          },
        },
      ],
    });

    expect(finished?.testScore).toBe(result.testScore);
  });

  test("reports the held-out score for every instance, not just the mean", async () => {
    let finished: GepaEvent | undefined;

    await new GepaOptimizer(config).optimize({
      ...task,
      testSet: TESTSET,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              finished = event;
            }
          },
        },
      ],
    });

    expect(finished?.type).toBe("finish");
    expect(
      finished?.type === "finish" ? finished.testInstanceScores : undefined,
    ).toEqual([1, 0]);
  });

  test("leaves a held-out instance an infrastructure failure lost unknown", async () => {
    // Reported as a zero it reads as the winner failing an instance it was
    // never actually measured on, which is the one thing a held-out number is
    // supposed to be trustworthy about.
    const adapter = createKeywordAdapter();
    let finished: GepaEvent | undefined;

    const result = await new GepaOptimizer(config).optimize({
      ...task,
      adapter: {
        ...adapter,
        evaluate: async (args) => {
          const evaluation = await adapter.evaluate(args);
          if (args.run.split !== "test") {
            return evaluation;
          }
          return {
            ...evaluation,
            scores: evaluation.scores.map(() => 0),
            transient: args.batch.map((_datum, index) => index === 0),
          };
        },
      },
      testSet: TESTSET,
      retry: { attempts: 0 },
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              finished = event;
            }
          },
        },
      ],
    });

    expect(
      finished?.type === "finish" ? finished.testInstanceScores : undefined,
    ).toEqual([undefined, 0]);
    expect(result.testScore).toBe(0);
  });

  test("reports what the winner produced on held-out instances when outputs are tracked", async () => {
    let finished: GepaEvent | undefined;

    await new GepaOptimizer({ ...config, trackBestOutputs: true }).optimize({
      ...task,
      testSet: TESTSET,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              finished = event;
            }
          },
        },
      ],
    });

    expect(
      finished?.type === "finish" ? finished.testOutputs : undefined,
    ).toHaveLength(TESTSET.length);
  });

  test("omits held-out outputs when the run is not tracking them", async () => {
    let finished: GepaEvent | undefined;

    await new GepaOptimizer(config).optimize({
      ...task,
      testSet: TESTSET,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              finished = event;
            }
          },
        },
      ],
    });

    expect(
      finished?.type === "finish" ? finished.testOutputs : undefined,
    ).toBeUndefined();
  });

  test("refuses an empty testSet rather than reporting a meaningless zero", async () => {
    await expect(
      new GepaOptimizer(config).optimize({ ...task, testSet: [] }),
    ).rejects.toThrow(/testSet/);
  });
});

describe("optimize checkpoint fidelity", () => {
  const config = {
    minibatchSize: 2,
    seed: 1,
  };
  const task = {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 400,
  };

  test("keeps the run fingerprint small however large the dataset is", async () => {
    const large = Array.from({ length: 200 }, (_, index) => ({
      question: `question ${index} ${"padding ".repeat(20)}`,
      required: ["hold"],
    }));

    const result = await new GepaOptimizer({
      ...config,
      maxIterations: 1,
    }).optimize({
      ...task,
      trainingSet: large,
      maxMetricCalls: 1000,
    });

    // The fingerprint is written into every checkpoint, so it must not grow
    // with the data it identifies.
    expect(result.snapshot.fingerprint.length).toBeLessThan(200);
  });

  test("omits the cache from checkpoints when asked", async () => {
    const result = await new GepaOptimizer({
      ...config,
      maxIterations: 2,
      checkpointCache: false,
    }).optimize({
      ...task,
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
  resumeFrom?: GepaSnapshot;
}): Promise<{ batches: string[]; result: GepaResult }> {
  const adapter = createKeywordAdapter();
  const batches: string[] = [];

  const result = await new GepaOptimizer({
    maxIterations: args.maxIterations,
    minibatchSize: 2,
    seed: 1,
  }).optimize({
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
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
    ...(args.resumeFrom === undefined ? {} : { resumeFrom: args.resumeFrom }),
  });

  return { batches, result };
}

/**
 * A task where each component answers its own instances, so lineages that
 * improved different components stay complementary for long enough that a run
 * has many merges available to it.
 */
function mergeRunwayConfig() {
  return {
    maxIterations: 40,
    minibatchSize: 3,
    seed: 1,
    proposals: { perIteration: 3, selection: "all" as const },
  };
}

function mergeRunwayTask() {
  return {
    seedCandidate: { alpha: "", beta: "", gamma: "" },
    trainingSet: PART_TASKS,
    adapter: createPartAdapter(),
    reflect: createAppendingReflector(),
    maxMetricCalls: 5000,
  };
}

function createPartAdapter(): GepaAdapter<
  (typeof PART_TASKS)[number],
  null,
  string
> {
  const marks = (text: string) => (text.match(/\+/g) ?? []).length;

  return {
    evaluate: ({ batch, candidate }) => ({
      outputs: batch.map(() => ""),
      scores: batch.map((datum) =>
        Math.min(1, marks(candidate[datum.part] ?? "") / datum.need),
      ),
      feedback: batch.map(
        (datum) =>
          `${datum.part} has ${marks(candidate[datum.part] ?? "")} of ${datum.need}`,
      ),
      trajectories: batch.map(() => null),
    }),
    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) =>
      Object.fromEntries(
        componentsToUpdate.map((component) => [
          component,
          batch.map((datum, index) => ({
            inputs: { part: datum.part },
            generatedOutputs: "",
            feedback: evaluation.feedback?.[index] ?? "",
          })),
        ]),
      ),
  };
}

/** Adds one mark to whichever component it is asked to rewrite. */
function createAppendingReflector(): TextModel {
  return async ({ prompt }) => {
    const current =
      /<current_instruction>\n([\s\S]*?)\n<\/current_instruction>/.exec(
        prompt,
      )?.[1] ?? "";

    return `\`\`\`\n${current.trim()}+\n\`\`\``;
  };
}

/**
 * A run whose every proposal loses, over a candidate with enough components to
 * move a cursor — the two pieces of snapshot state a resumed run writes back.
 */
function resumeConfig(args: { maxIterations: number }) {
  return {
    maxIterations: args.maxIterations,
    minibatchSize: 2,
    seed: 1,
  };
}

function resumeTask() {
  return {
    seedCandidate: { alpha: "a", beta: "b", gamma: "c" },
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createAlternatingReflector(),
    maxMetricCalls: 400,
  };
}

/**
 * Improves on every other call, so a run both advances component cursors and
 * accumulates rejected proposals — an always-improving reflector records no
 * rejections, and an always-degrading one never moves the frontier.
 */
function createAlternatingReflector(): TextModel {
  const improving = createKeywordReflector();
  let calls = 0;

  return async (args) => {
    calls += 1;
    return calls % 2 === 0
      ? "```\nno useful information\n```"
      : improving(args);
  };
}

/** Everything about a run that a reproducible engine must reach identically. */
function lineageOf(
  result: GepaResult,
): [number, number[], Record<string, string>][] {
  return result.candidates.map((record) => [
    record.id,
    record.parentIds,
    record.candidate,
  ]);
}

/**
 * Scores a candidate on how many marks it carries, so two candidates with the
 * same number of marks score identically however differently they are written.
 */
function createMarkAdapter(): GepaAdapter<
  (typeof MARK_EXAMPLES)[number],
  null,
  string
> {
  return {
    evaluate: ({ batch, candidate }) => {
      const marks = (Object.values(candidate).join("").match(/\+/g) ?? [])
        .length;

      return {
        outputs: batch.map(() => ""),
        scores: batch.map((datum) => Math.min(1, marks / datum.need)),
        feedback: batch.map((datum) => `have ${marks} of ${datum.need}`),
        trajectories: batch.map(() => null),
      };
    },
    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) =>
      Object.fromEntries(
        componentsToUpdate.map((component) => [
          component,
          batch.map((datum, index) => ({
            inputs: { need: datum.need },
            generatedOutputs: "",
            feedback: evaluation.feedback?.[index] ?? "",
          })),
        ]),
      ),
  };
}

/**
 * Proposes one more mark than the feedback reports, so two parents that scored
 * the same on the same minibatch get the same proposal — which is what makes
 * two siblings of one iteration collide. A pure function of its prompt: `pace`
 * decides only how long the call takes, which nothing may depend on.
 */
function createMarkReflector(pace: (prompt: string) => number): TextModel {
  return async ({ prompt }) => {
    const marks = Number(/have (\d+) of/.exec(prompt)?.[1] ?? 0);
    const needs = [...prompt.matchAll(/have \d+ of (\d+)/g)]
      .map((match) => match[1])
      .join("-");

    for (let tick = 0; tick < pace(prompt); tick += 1) {
      await Promise.resolve();
    }
    return `\`\`\`\n${"+".repeat(marks + 1)}${needs}\n\`\`\``;
  };
}

function countProposals(
  events: readonly GepaEvent[],
  iteration: number,
): number {
  return events.filter(
    (event) => event.type === "proposal" && event.iteration === iteration,
  ).length;
}

/** Wraps an adapter to observe how many evaluations are ever in flight at once. */
function withOverlapTracking<Datum, Trajectory, Output>(
  adapter: GepaAdapter<Datum, Trajectory, Output>,
): {
  adapter: GepaAdapter<Datum, Trajectory, Output>;
  maxInFlight: () => number;
} {
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

  await new GepaOptimizer({
    maxIterations: args.maxIterations,
    minibatchSize: 2,
    seed: 1,
  }).optimize({
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: {
      ...adapter,
      evaluate: (evaluateArgs) => {
        contexts.push(evaluateArgs.run);
        return adapter.evaluate(evaluateArgs);
      },
    },
    reflect: createKeywordReflector(),
    maxMetricCalls: 300,
  });

  return contexts;
}

/**
 * The keyword task scored on two competing objectives: coverage rewards saying
 * more, brevity rewards saying less. A candidate cannot lead both.
 */
function createObjectiveAdapter(): GepaAdapter<
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

describe("optimize proposal strategies", () => {
  const task = {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 400,
  };

  test("numbers every proposal in the run distinctly", async () => {
    const attempts: number[] = [];
    const adapter = createKeywordAdapter();

    await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
      maxIterations: 3,
      proposals: { perIteration: 2 },
    }).optimize({
      ...task,
      adapter: {
        ...adapter,
        proposeNewTexts: ({ attempt, candidate, componentsToUpdate }) => {
          attempts.push(attempt as number);
          return {
            [componentsToUpdate[0] as "instruction"]: `${candidate.instruction} hold`,
          };
        },
      },
    });

    expect(attempts).toEqual([...new Set(attempts)]);
    expect(attempts).toHaveLength(6);
  });

  test("rotates the configured strategies across proposals", async () => {
    const used: string[] = [];
    const label = (name: string) => () => {
      used.push(name);
      return "```\nrewritten\n```";
    };

    await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
      maxIterations: 2,
      proposals: { perIteration: 2 },
      reflection: { strategies: [label("a"), label("b"), label("c")] },
    }).optimize(task);

    expect(used).toEqual(["a", "b", "c", "a"]);
  });

  test("refuses a single prompt builder alongside a strategy list", () => {
    expect(
      () =>
        new GepaOptimizer({
          reflection: {
            buildPrompt: () => "prompt",
            strategies: [() => "prompt"],
          },
        }),
    ).toThrow(/buildPrompt|strategies/);
  });
});
