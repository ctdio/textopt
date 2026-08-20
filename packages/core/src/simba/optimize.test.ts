import { describe, expect, test } from "vitest";
import { createMemoryCache, stableHash } from "../cache.js";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "../testing.js";
import type { Adapter, TextModel } from "../types.js";
import { SimbaOptimizer } from "./optimize.js";
import type { SimbaEvent, SimbaSnapshot } from "./optimize.js";

const SEED = { instruction: "Answer the user question." };
const SEED_WITH_DEMOS = { instruction: "Answer the user question.", demos: "" };

function baseAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const keyword = createKeywordAdapter();
  return { evaluate: (args) => keyword.evaluate(args) };
}

/** The base adapter, reporting ten input tokens for every rollout it runs. */
function pricedAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const keyword = createKeywordAdapter();
  return {
    evaluate: (args) => ({
      ...keyword.evaluate(args),
      usage: args.batch.map(() => ({ inputTokens: 10, outputTokens: 5 })),
    }),
  };
}

/**
 * A stand-in for the advice model: it reads the missing terms out of whichever
 * trajectory the prompt carries and hands each component the terms it lacks.
 */
function createAdviceReflector(): TextModel {
  return async ({ prompt }) => {
    const components = [...prompt.matchAll(/<component name="([^"]+)">/g)].map(
      (match) => match[1] as string,
    );

    const missing = new Set<string>();
    for (const match of prompt.matchAll(
      /Missing required terms: ([^"\\\n]+)/g,
    )) {
      for (const term of (match[1] ?? "").split(",")) {
        missing.add(term.trim());
      }
    }

    return components
      .map(
        (component) =>
          `<advice component="${component}">${[...missing].join(" ")}</advice>`,
      )
      .join("\n");
  };
}

function ruleTask() {
  return {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: baseAdapter(),
    reflect: createAdviceReflector(),
    maxMetricCalls: 400,
  };
}

function optimizer(config = {}) {
  return new SimbaOptimizer({
    minibatchSize: 2,
    candidates: 2,
    maxSteps: 2,
    strategies: ["appendRule"] as const,
    ...config,
  });
}

describe("SimbaOptimizer", () => {
  test("improves on the seed candidate", async () => {
    const result = await optimizer().optimize(ruleTask());

    expect(result.bestScore).toBeGreaterThan(result.seedScore);
  });

  test("appends the advice it was given to the instruction", async () => {
    const result = await optimizer().optimize(ruleTask());

    expect(result.bestCandidate.instruction).toContain(SEED.instruction);
    expect(result.bestCandidate.instruction.length).toBeGreaterThan(
      SEED.instruction.length,
    );
  });

  test("reports the seed's own validation score", async () => {
    const result = await optimizer().optimize(ruleTask());

    expect(result.seedScore).toBe(0);
  });

  test("returns the highest scoring finalist as the winner", async () => {
    const result = await optimizer().optimize(ruleTask());
    const scores = result.finalists.map((finalist) => finalist.score);

    expect(result.bestScore).toBe(Math.max(...scores));
  });

  test("ranks the finalists from best to worst", async () => {
    const result = await optimizer().optimize(ruleTask());
    const scores = result.finalists.map((finalist) => finalist.score);

    expect(scores).toEqual([...scores].sort((a, b) => b - a));
  });

  test("harvests demonstrations into the demo components", async () => {
    const result = await optimizer({
      strategies: ["appendDemo"] as const,
    }).optimize({
      ...ruleTask(),
      seedCandidate: SEED_WITH_DEMOS,
      demoComponents: ["demos"] as const,
    });

    expect(result.bestCandidate.demos).toContain("<demo>");
  });

  test("leaves the instruction alone when only demos are being appended", async () => {
    const result = await optimizer({
      strategies: ["appendDemo"] as const,
    }).optimize({
      ...ruleTask(),
      seedCandidate: SEED_WITH_DEMOS,
      demoComponents: ["demos"] as const,
    });

    expect(result.bestCandidate.instruction).toBe(SEED.instruction);
  });

  test("keeps what a component already said when a demo lands in it", async () => {
    const result = await optimizer({
      strategies: ["appendDemo"] as const,
    }).optimize({
      ...ruleTask(),
      demoComponents: ["instruction"] as const,
    });

    expect(result.bestCandidate.instruction).toContain("<demo>");
    expect(result.bestCandidate.instruction.startsWith(SEED.instruction)).toBe(
      true,
    );
  });

  test("writes rules into the only component even when it holds the demos", async () => {
    const result = await optimizer({
      strategies: ["appendDemo", "appendRule"] as const,
      maxSteps: 4,
    }).optimize({
      ...ruleTask(),
      demoComponents: ["instruction"] as const,
    });

    expect(result.reflectionCalls).toBeGreaterThan(0);
  });

  test("calls no reflection model when only demos are being appended", async () => {
    let calls = 0;
    const result = await optimizer({
      strategies: ["appendDemo"] as const,
    }).optimize({
      ...ruleTask(),
      seedCandidate: SEED_WITH_DEMOS,
      demoComponents: ["demos"] as const,
      reflect: async () => {
        calls += 1;
        return "";
      },
    });

    expect(calls).toBe(0);
    expect(result.reflectionCalls).toBe(0);
  });

  test("writes no demos when no demo component was named", async () => {
    const result = await optimizer().optimize(ruleTask());

    expect(Object.values(result.bestCandidate).join(" ")).not.toContain(
      "<demo>",
    );
  });

  test("stops calling the reflection model once its call budget is spent", async () => {
    let calls = 0;
    const advise = createAdviceReflector();

    await optimizer({ maxSteps: 8, maxReflectionCalls: 1 }).optimize({
      ...ruleTask(),
      reflect: async (args) => {
        calls += 1;
        return advise(args);
      },
    });

    expect(calls).toBe(1);
  });

  test("reports the reflection budget as the reason it stopped", async () => {
    const result = await optimizer({
      maxSteps: 8,
      maxReflectionCalls: 1,
    }).optimize(ruleTask());

    expect(result.stopReason).toBe("reflectionBudgetExhausted");
  });

  test("stops after the configured number of steps", async () => {
    const steps: number[] = [];
    const result = await optimizer().optimize({
      ...ruleTask(),
      reporters: [
        {
          onEvent: (event: SimbaEvent) => {
            if (event.type === "stepStart") {
              steps.push(event.step);
            }
          },
        },
      ],
    });

    expect(steps).toEqual([0, 1]);
    expect(result.stopReason).toBe("maxSteps");
  });

  test("stops when the rollout budget cannot cover another step", async () => {
    const result = await optimizer().optimize({
      ...ruleTask(),
      maxMetricCalls: 14,
    });

    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("never spends more rollouts than the budget allows", async () => {
    const result = await optimizer({ maxSteps: 20 }).optimize({
      ...ruleTask(),
      maxMetricCalls: 60,
    });

    expect(result.metricCalls).toBeLessThanOrEqual(60);
  });

  test("rejects a minibatch larger than the training set", async () => {
    await expect(
      optimizer({ minibatchSize: 99 }).optimize(ruleTask()),
    ).rejects.toThrow(/minibatchSize/);
  });

  test("rejects an empty strategy list", async () => {
    await expect(
      optimizer({ strategies: [] }).optimize(ruleTask()),
    ).rejects.toThrow(/at least one strategy/);
  });

  test("rejects appendDemo without a demo component to write into", async () => {
    await expect(
      optimizer({ strategies: ["appendDemo"] as const }).optimize(ruleTask()),
    ).rejects.toThrow(/demoComponents/);
  });

  test("scores the winner on a held-out test set", async () => {
    const result = await optimizer().optimize({
      ...ruleTask(),
      validationSet: KEYWORD_EXAMPLES.slice(0, 2),
      testSet: KEYWORD_EXAMPLES.slice(2),
    });

    expect(result.testScore).toBeDefined();
    expect(result.testMetricCalls).toBe(2);
  });

  test("keeps the held-out sweep out of the run's usage", async () => {
    // No ceiling bounds the held-out sweep: it runs once the search has already
    // stopped. Counting it in `usage` would describe a run that honoured
    // `maxCostUsd` as having overrun it.
    const result = await optimizer().optimize({
      ...ruleTask(),
      adapter: pricedAdapter(),
      validationSet: KEYWORD_EXAMPLES.slice(0, 2),
      testSet: KEYWORD_EXAMPLES.slice(2),
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
  test("produces the same winner from the same seed", async () => {
    const first = await optimizer({ seed: 7 }).optimize(ruleTask());
    const second = await optimizer({ seed: 7 }).optimize(ruleTask());

    expect(second.bestCandidate).toEqual(first.bestCandidate);
    expect(second.bestScore).toBe(first.bestScore);
  });

  test("stops when the caller aborts", async () => {
    const controller = new AbortController();
    const result = await optimizer({ maxSteps: 20 }).optimize({
      ...ruleTask(),
      signal: controller.signal,
      reporters: [
        {
          onEvent: (event: SimbaEvent) => {
            if (event.type === "stepStart" && event.step === 1) {
              controller.abort();
            }
          },
        },
      ],
    });

    expect(result.stopReason).toBe("aborted");
  });

  test("continues from a checkpoint instead of starting over", async () => {
    const snapshots: SimbaSnapshot[] = [];
    await optimizer({ maxSteps: 1 }).optimize({
      ...ruleTask(),
      onCheckpoint: (snapshot) => {
        snapshots.push(snapshot);
      },
    });
    const checkpoint = JSON.parse(
      JSON.stringify(snapshots[snapshots.length - 1]),
    ) as SimbaSnapshot;

    const resumed = await optimizer({ maxSteps: 2 }).optimize({
      ...ruleTask(),
      resumeFrom: checkpoint,
    });

    expect(resumed.steps).toBe(2);
    expect(
      resumed.snapshot.programs.slice(0, checkpoint.programs.length),
    ).toEqual(checkpoint.programs);
    expect(resumed.snapshot.programs.length).toBeGreaterThan(
      checkpoint.programs.length,
    );
  });

  test("carries usage already spent into a resumed run", async () => {
    // `maxCostUsd` is a ceiling on the run, not on the segment. A resumed run
    // that restarts its token accounting at zero lets an interrupted-and-
    // resumed loop spend the ceiling over and over.
    const priced = {
      ...ruleTask(),
      cache: false as const,
      adapter: pricedAdapter(),
    };
    const snapshots: SimbaSnapshot[] = [];

    const interrupted = await optimizer({ maxSteps: 1 }).optimize({
      ...priced,
      onCheckpoint: (snapshot) => {
        snapshots.push(snapshot);
      },
    });
    const checkpoint = JSON.parse(
      JSON.stringify(snapshots[snapshots.length - 1]),
    ) as SimbaSnapshot;

    const resumed = await optimizer({ maxSteps: 2 }).optimize({
      ...priced,
      resumeFrom: checkpoint,
    });

    expect(interrupted.usage.inputTokens).toBeGreaterThan(0);
    // Ten input tokens a rollout, uncached, over both segments.
    expect(resumed.usage.inputTokens).toBe(resumed.metricCalls * 10);
  });

  test("re-scores nothing the checkpointed run already paid for", async () => {
    const snapshots: SimbaSnapshot[] = [];
    const first = await optimizer({ maxSteps: 1 }).optimize({
      ...ruleTask(),
      onCheckpoint: (snapshot) => {
        snapshots.push(snapshot);
      },
    });
    const checkpoint = JSON.parse(
      JSON.stringify(snapshots[snapshots.length - 1]),
    ) as SimbaSnapshot;

    const resumed = await optimizer({ maxSteps: 2 }).optimize({
      ...ruleTask(),
      resumeFrom: checkpoint,
    });

    expect(resumed.metricCalls).toBeGreaterThan(first.metricCalls);
  });

  test("refuses a checkpoint from a different run", async () => {
    const result = await optimizer({ maxSteps: 1 }).optimize(ruleTask());

    await expect(
      optimizer().optimize({
        ...ruleTask(),
        seedCandidate: { instruction: "A different seed entirely." },
        resumeFrom: result.snapshot,
      }),
    ).rejects.toThrow(/checkpoint does not belong to this run/);
  });
});

describe("SimbaOptimizer concurrency", () => {
  test("scores a step's candidates at the same time", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await optimizer({ candidates: 4, concurrency: 4 }).optimize({
      ...ruleTask(),
      adapter: tracked.adapter,
    });

    expect(tracked.maxInFlight()).toBeGreaterThan(1);
  });

  test("scores them one at a time by default", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await optimizer({ candidates: 4 }).optimize({
      ...ruleTask(),
      adapter: tracked.adapter,
    });

    expect(tracked.maxInFlight()).toBe(1);
  });

  test("reaches the same finalists whether or not the evaluations overlap", async () => {
    const serial = await optimizer({ candidates: 4, maxSteps: 3 }).optimize(
      ruleTask(),
    );
    const concurrent = await optimizer({
      candidates: 4,
      maxSteps: 3,
      concurrency: 4,
    }).optimize(ruleTask());

    expect(concurrent.finalists).toEqual(serial.finalists);
    expect(concurrent.bestCandidate).toEqual(serial.bestCandidate);
    expect(concurrent.bestScore).toBe(serial.bestScore);
    expect(concurrent.seedScore).toBe(serial.seedScore);
    expect(concurrent.metricCalls).toBe(serial.metricCalls);
    expect(concurrent.snapshot.programs).toEqual(serial.snapshot.programs);
    expect(concurrent.snapshot.programScores).toEqual(
      serial.snapshot.programScores,
    );
  });

  test("builds the same program pool whichever order the evaluations finish in", async () => {
    // A step's candidates enter the pool at the index the next step's softmax
    // will sample them by, so the pool has to be built in the order the step
    // proposed them rather than the order the network returned them in.
    const run = async (pace: (candidate: string) => number) => {
      const result = await optimizer({
        candidates: 4,
        maxSteps: 3,
        concurrency: 4,
      }).optimize({
        ...ruleTask(),
        adapter: withPacing(baseAdapter(), pace),
      });

      return result.snapshot;
    };

    const shortestFirst = await run((candidate) => candidate.length % 7);
    const longestFirst = await run((candidate) => 7 - (candidate.length % 7));

    expect(shortestFirst.programs).toEqual(longestFirst.programs);
    expect(shortestFirst.programScores).toEqual(longestFirst.programScores);
    expect(shortestFirst.winners).toEqual(longestFirst.winners);
  });

  test("sweeps the finalists at the same time", async () => {
    const phases: string[] = [];
    const adapter = baseAdapter();
    let inFlight = 0;
    let peak = 0;

    await optimizer({ candidates: 4, concurrency: 4 }).optimize({
      ...ruleTask(),
      adapter: {
        evaluate: async (args) => {
          const validation = args.run.split === "val";
          inFlight += validation ? 1 : 0;
          peak = Math.max(peak, inFlight);
          phases.push(args.run.phase);
          try {
            await Promise.resolve();
            return await adapter.evaluate(args);
          } finally {
            inFlight -= validation ? 1 : 0;
          }
        },
      },
    });

    expect(phases).toContain("validation");
    expect(peak).toBeGreaterThan(1);
  });

  test("never spends past the budget when the evaluations overlap", async () => {
    const result = await optimizer({
      candidates: 4,
      maxSteps: 10,
      concurrency: 4,
    }).optimize({ ...ruleTask(), maxMetricCalls: 60 });

    expect(result.metricCalls).toBeLessThanOrEqual(60);
  });

  test("rejects a concurrency below one", () => {
    expect(() => new SimbaOptimizer({ concurrency: 0 })).toThrow(/concurrency/);
  });
});

/** Wraps an adapter to observe how many evaluations are ever in flight at once. */
function withOverlapTracking<Datum, Trajectory, Output>(
  adapter: Adapter<Datum, Trajectory, Output>,
): {
  adapter: Adapter<Datum, Trajectory, Output>;
  maxInFlight: () => number;
} {
  let inFlight = 0;
  let peak = 0;

  return {
    maxInFlight: () => peak,
    adapter: {
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

/** Settles each evaluation after a delay the candidate's own text decides. */
function withPacing<Datum, Trajectory, Output>(
  adapter: Adapter<Datum, Trajectory, Output>,
  pace: (candidate: string) => number,
): Adapter<Datum, Trajectory, Output> {
  return {
    evaluate: async (args) => {
      const ticks = pace(Object.values(args.candidate).join(" "));
      for (let tick = 0; tick < ticks; tick += 1) {
        await Promise.resolve();
      }
      return adapter.evaluate(args);
    },
  };
}

describe("SimbaOptimizer reporting", () => {
  test("reports the seed as candidate 0, before any improvement", async () => {
    // The seed is what every later candidate is read against. A report that
    // starts at the first improvement has nothing to compare it to.
    const accepted: { id: number; candidate: Record<string, string> }[] = [];

    await optimizer().optimize({
      ...ruleTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              accepted.push({
                id: event.candidateId,
                candidate: { ...event.candidate },
              });
            }
          },
        },
      ],
    });

    expect(accepted[0]?.id).toBe(0);
    expect(accepted[0]?.candidate).toEqual(SEED);
  });

  test("reports an acceptance with the text that scored", async () => {
    const accepted: Record<string, string>[] = [];

    await optimizer().optimize({
      ...ruleTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              accepted.push({ ...event.candidate });
            }
          },
        },
      ],
    });

    expect(accepted.length).toBeGreaterThan(0);
  });

  test("reports a per-instance row aligned with the validation set", async () => {
    const rows: (number | undefined)[][] = [];

    await optimizer().optimize({
      ...ruleTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              rows.push([...event.instanceScores]);
            }
          },
        },
      ],
    });

    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row).toHaveLength(KEYWORD_EXAMPLES.length);
    }
  });

  test("names the winner in finish with an id an acceptance carried", async () => {
    const acceptedIds: number[] = [];
    let bestCandidateId: number | undefined;

    await optimizer().optimize({
      ...ruleTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              acceptedIds.push(event.candidateId);
            }
            if (event.type === "finish") {
              bestCandidateId = event.bestCandidateId;
            }
          },
        },
      ],
    });

    expect(acceptedIds).toContain(bestCandidateId);
  });

  test("flushes every reporter once the run ends", async () => {
    const flushed: string[] = [];

    await optimizer().optimize({
      ...ruleTask(),
      reporters: [
        {
          flush: async () => {
            flushed.push("first");
          },
        },
        {
          flush: async () => {
            flushed.push("second");
          },
        },
      ],
    });

    expect(flushed.toSorted()).toEqual(["first", "second"]);
  });

  test("reaches the same result with a cache as without one", async () => {
    const uncached = await optimizer({
      minibatchSize: 2,
      maxSteps: 3,
      seed: 7,
    }).optimize({ ...ruleTask(), cache: false });
    const cached = await optimizer({
      minibatchSize: 2,
      maxSteps: 3,
      seed: 7,
    }).optimize(ruleTask());

    expect(cached.bestCandidate).toEqual(uncached.bestCandidate);
    expect(cached.bestScore).toBe(uncached.bestScore);
  });

  test("caches a training score against the row that produced it", async () => {
    // Scores here depend on the row and nothing else, so a score filed under
    // another row's id is one the search will later read for an instance that
    // never earned it.
    const weights = new Map(
      KEYWORD_EXAMPLES.map((datum, index) => [datum, (index + 1) / 10]),
    );
    const keyword = createKeywordAdapter();
    const cache = createMemoryCache();

    await optimizer({ minibatchSize: 2, maxSteps: 3, seed: 7 }).optimize({
      ...ruleTask(),
      adapter: {
        evaluate: (args) => ({
          ...keyword.evaluate(args),
          scores: args.batch.map((datum) => weights.get(datum) ?? 0),
        }),
      },
      cache,
    });

    const trainEntries = (cache.entries?.() ?? []).filter(([key]) =>
      key.startsWith("train:"),
    );

    expect(trainEntries.length).toBeGreaterThan(0);
    for (const [key, cached] of trainEntries) {
      const id = key.slice(key.lastIndexOf(":") + 1);
      const row = KEYWORD_EXAMPLES.find((datum) => stableHash(datum) === id);
      expect(cached.score).toBe(
        weights.get(row as (typeof KEYWORD_EXAMPLES)[number]),
      );
    }
  });

  test("keys the training cache by dataset row, not minibatch position", async () => {
    const seen: string[] = [];
    const cache = createMemoryCache();

    await optimizer({ minibatchSize: 2, maxSteps: 3, seed: 7 }).optimize({
      ...ruleTask(),
      cache: {
        get: (key) => {
          seen.push(key);
          return cache.get(key);
        },
        set: (key, value) => cache.set(key, value),
      },
    });

    const trainKeys = seen.filter((key) => key.startsWith("train:"));
    const instanceIds = new Set(
      trainKeys.map((key) => key.slice(key.lastIndexOf(":") + 1)),
    );

    expect(instanceIds.size).toBeGreaterThan(2);
  });
});
