import { afterEach, describe, expect, test, vi } from "vitest";
import type { Optimizer, OptimizerResult } from "../optimizer.js";
import {
  KEYWORD_EXAMPLES,
  createKeywordAdapter,
  createSamplingReflector,
} from "../testing.js";
import type { Adapter } from "../types.js";
import { RandomSearchOptimizer } from "./optimize.js";
import type {
  RandomSearchSnapshot,
  RandomSearchStopReason,
} from "./optimize.js";

const SEED = { instruction: "Answer the user question." };

/** The adapter contract without any reflective surface on it at all. */
function baseAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const keyword = createKeywordAdapter();
  return { evaluate: (args) => keyword.evaluate(args) };
}

function task() {
  return {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: baseAdapter(),
    reflect: createSamplingReflector(),
    maxMetricCalls: 200,
  };
}

describe("RandomSearchOptimizer", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("stops once the wall clock deadline passes", async () => {
    // Rollout and cost ceilings bound what a run spends, not how long it
    // takes: a run stuck behind a rate limit costs nothing and runs forever.
    vi.useFakeTimers();

    const result = await new RandomSearchOptimizer({
      variants: 1,
      maxRounds: 20,
    }).optimize({
      ...task(),
      adapter: {
        evaluate: ({ batch }) => {
          vi.advanceTimersByTime(400);
          return {
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0.5),
          };
        },
      },
      maxWallClockMs: 1000,
    });

    expect(result.stopReason).toBe("deadlineReached");
  });
  test("stops once the reported cost reaches the ceiling", async () => {
    const result = await new RandomSearchOptimizer({
      variants: 1,
      maxRounds: 5,
    }).optimize({
      ...task(),
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 0.5),
          usage: batch.map(() => ({ costUsd: 1 })),
        }),
      },
      maxCostUsd: 4,
    });

    expect(result.stopReason).toBe("costExhausted");
    expect(result.usage.costUsd).toBe(4);
  });

  test("leaves a transiently failed rollout out of the score it reports", async () => {
    // A rate limit measured the provider, not the candidate, and a baseline
    // that averages its zero in is one no rollout ever produced.
    const result = await new RandomSearchOptimizer({
      variants: 1,
      maxRounds: 1,
    }).optimize({
      ...task(),
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map((_, index) => (index === 2 ? 0 : 1)),
          transient: batch.map((_, index) => index === 2),
        }),
      },
      retry: { attempts: 0 },
    });

    expect(result.seedScore).toBe(1);
  });

  test("satisfies the Optimizer contract", async () => {
    const search = new RandomSearchOptimizer({ variants: 2, maxRounds: 2 });
    const contract: Optimizer<RandomSearchStopReason> = search;

    const result = await search.optimize(task());
    const outcome: OptimizerResult<"instruction", RandomSearchStopReason> =
      result;

    expect(contract).toBe(search);
    expect(outcome.bestScore).toBeGreaterThanOrEqual(0);
  });

  test("runs against an adapter with no reflective dataset", async () => {
    const result = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 2,
    }).optimize(task());

    expect(result.bestCandidate.instruction).toContain(SEED.instruction);
  });

  test("improves on the seed candidate by sampling alone", async () => {
    const result = await new RandomSearchOptimizer({
      variants: 4,
      maxRounds: 8,
    }).optimize({ ...task(), maxMetricCalls: 400 });

    expect(result.seedScore).toBe(0);
    expect(result.bestScore).toBeGreaterThan(0);
  });

  test("never spends more than the metric call budget", async () => {
    const result = await new RandomSearchOptimizer({
      variants: 4,
    }).optimize({ ...task(), maxMetricCalls: 37 });

    expect(result.metricCalls).toBeLessThanOrEqual(37);
    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("stops once the round limit is reached", async () => {
    const result = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 3,
    }).optimize({ ...task(), maxMetricCalls: 10_000 });

    expect(result.rounds).toBe(3);
    expect(result.stopReason).toBe("maxRounds");
  });

  test("keeps a variant only when it beats the incumbent", async () => {
    const scores: number[] = [];

    await new RandomSearchOptimizer({ variants: 3, maxRounds: 6 }).optimize({
      ...task(),
      maxMetricCalls: 400,
      onEvent: (event) => {
        if (event.type === "candidateAccepted") {
          scores.push(event.score);
        }
      },
    });

    expect(scores.length).toBeGreaterThan(0);
    expect([...scores].sort((a, b) => a - b)).toEqual(scores);
  });

  test("shows the proposer no score, feedback or evidence", async () => {
    const prompts: string[] = [];

    await new RandomSearchOptimizer({ variants: 2, maxRounds: 3 }).optimize({
      ...task(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nAnswer the user question. hold\n```";
      },
    });

    expect(prompts.length).toBeGreaterThan(0);
    for (const prompt of prompts) {
      expect(prompt).not.toMatch(/score|feedback|Missing required/i);
    }
  });

  test("scores the winner on a held-out testSet", async () => {
    const result = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 2,
    }).optimize({
      ...task(),
      testSet: [
        { question: "held out, satisfied", required: ["answer"] },
        { question: "held out, unsatisfiable", required: ["zzz-never"] },
      ],
    });

    expect(result.testScore).toBe(0.5);
    expect(result.testMetricCalls).toBe(2);
  });

  test("stops when the signal is aborted", async () => {
    const controller = new AbortController();
    let evaluations = 0;

    const keyword = createKeywordAdapter();
    const result = await new RandomSearchOptimizer({
      variants: 2,
    }).optimize({
      ...task(),
      maxMetricCalls: 10_000,
      adapter: {
        evaluate: (args) => {
          evaluations += 1;
          if (evaluations === 3) {
            controller.abort();
          }
          return keyword.evaluate(args);
        },
      },
      signal: controller.signal,
    });

    expect(result.stopReason).toBe("aborted");
  });

  test("walks the components in turn on a multi-component candidate", async () => {
    const components: string[] = [];

    await new RandomSearchOptimizer({ variants: 1, maxRounds: 4 }).optimize({
      ...task(),
      seedCandidate: { intro: "Answer.", outro: "Be brief." },
      maxMetricCalls: 400,
      onEvent: (event) => {
        if (event.type === "roundStart") {
          components.push(event.component);
        }
      },
    });

    expect(components).toEqual(["intro", "outro", "intro", "outro"]);
  });
});

describe("RandomSearchOptimizer stalls", () => {
  test("stops when a full cycle of components proposes nothing new", async () => {
    // A proposer that keeps returning the incumbent costs no rollouts, so
    // neither the metric budget nor the cost ceiling can end the run: without
    // this the loop spins forever making reflection calls nobody bounded.
    const result = await new RandomSearchOptimizer({ variants: 2 }).optimize({
      ...task(),
      seedCandidate: { instruction: "Answer the user question." },
      reflect: async () => "```\nAnswer the user question.\n```",
    });

    expect(result.stopReason).toBe("proposerStalled");
  });
});

describe("RandomSearchOptimizer checkpoints", () => {
  test("survives a round trip through JSON", async () => {
    let snapshot: RandomSearchSnapshot | undefined;

    await new RandomSearchOptimizer({ variants: 2, maxRounds: 2 }).optimize({
      ...task(),
      onCheckpoint: (taken) => {
        snapshot = taken;
      },
    });

    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  test("checkpoints after the seed sweep and after every round", async () => {
    const rounds: number[] = [];

    await new RandomSearchOptimizer({ variants: 2, maxRounds: 3 }).optimize({
      ...task(),
      onCheckpoint: (taken) => {
        rounds.push(taken.round);
      },
    });

    expect(rounds).toEqual([0, 1, 2, 3]);
  });

  test("resumes without re-scoring the seed candidate", async () => {
    const interrupted = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 1,
    }).optimize({ ...task(), cache: false });

    const resumed = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 3,
    }).optimize({
      ...task(),
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.rounds).toBe(3);
    expect(resumed.seedScore).toBe(interrupted.seedScore);
    expect(resumed.bestScore).toBeGreaterThanOrEqual(interrupted.bestScore);
  });

  test("charges a resumed run for what the checkpoint already spent", async () => {
    const interrupted = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 1,
    }).optimize({ ...task(), cache: false });

    const resumed = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 3,
    }).optimize({
      ...task(),
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.metricCalls).toBeGreaterThan(interrupted.metricCalls);
  });

  test("re-sweeps nothing when it resumes at the round it stopped on", async () => {
    const interrupted = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 1,
    }).optimize({ ...task(), cache: false });

    const resumed = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 1,
    }).optimize({
      ...task(),
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.metricCalls).toBe(interrupted.metricCalls);
  });

  test("refuses a checkpoint taken against a different seed candidate", async () => {
    const interrupted = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 1,
    }).optimize(task());

    await expect(
      new RandomSearchOptimizer({ variants: 2, maxRounds: 1 }).optimize({
        ...task(),
        seedCandidate: { instruction: "Something else entirely." },
        resumeFrom: interrupted.snapshot,
      }),
    ).rejects.toThrow("does not belong to this run");
  });

  test("carries the evaluation cache in the checkpoint", async () => {
    const interrupted = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 1,
    }).optimize(task());

    expect(interrupted.snapshot.cache?.length).toBeGreaterThan(0);
  });

  test("leaves the cache out when checkpointing it was turned off", async () => {
    const interrupted = await new RandomSearchOptimizer({
      variants: 2,
      maxRounds: 1,
      checkpointCache: false,
    }).optimize(task());

    expect(interrupted.snapshot.cache).toBeUndefined();
  });
});

describe("RandomSearchOptimizer concurrency", () => {
  test("scores a round's variants at the same time", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await new RandomSearchOptimizer({
      variants: 4,
      concurrency: 4,
      maxRounds: 2,
    }).optimize({ ...task(), adapter: tracked.adapter });

    expect(tracked.maxInFlight()).toBeGreaterThan(1);
  });

  test("scores them one at a time by default", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await new RandomSearchOptimizer({ variants: 4, maxRounds: 2 }).optimize({
      ...task(),
      adapter: tracked.adapter,
    });

    expect(tracked.maxInFlight()).toBe(1);
  });

  test("reaches the same incumbent whether or not the sweeps overlap", async () => {
    const serial = await new RandomSearchOptimizer({
      variants: 4,
      maxRounds: 3,
    }).optimize(task());
    const concurrent = await new RandomSearchOptimizer({
      variants: 4,
      concurrency: 4,
      maxRounds: 3,
    }).optimize(task());

    expect(concurrent.bestCandidate).toEqual(serial.bestCandidate);
    expect(concurrent.bestScore).toBe(serial.bestScore);
    expect(concurrent.metricCalls).toBe(serial.metricCalls);
    expect(concurrent.variantsEvaluated).toBe(serial.variantsEvaluated);
    expect(concurrent.stopReason).toBe(serial.stopReason);
  });

  test("reports the same acceptances whichever order the sweeps finish in", async () => {
    // The incumbent ratchet is the one piece of a round that is order
    // dependent: a variant is accepted only if it beats every variant before
    // it. Deciding that on whichever sweep returned first would make the
    // reported lineage a property of the network rather than of the search.
    const run = async (pace: (candidate: string) => number) => {
      const events: string[] = [];

      await new RandomSearchOptimizer({
        variants: 4,
        concurrency: 4,
        maxRounds: 3,
      }).optimize({
        ...task(),
        adapter: withPacing(baseAdapter(), pace),
        onEvent: (event) => {
          if (event.type === "candidateAccepted") {
            events.push(`${event.round}:${event.score}`);
          }
        },
      });

      return events;
    };

    const shortestFirst = await run((candidate) => candidate.length);
    const longestFirst = await run((candidate) => 100 - candidate.length);

    expect(shortestFirst).toEqual(longestFirst);
    expect(shortestFirst.length).toBeGreaterThan(0);
  });

  test("never spends past the budget when the sweeps overlap", async () => {
    const result = await new RandomSearchOptimizer({
      variants: 4,
      concurrency: 4,
      maxRounds: 20,
    }).optimize({ ...task(), maxMetricCalls: 30 });

    expect(result.metricCalls).toBeLessThanOrEqual(30);
  });

  test("rejects a concurrency below one", () => {
    expect(() => new RandomSearchOptimizer({ concurrency: 0 })).toThrow(
      /concurrency/,
    );
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
