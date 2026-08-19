import { describe, expect, test } from "vitest";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "../testing.js";
import type { Adapter } from "../types.js";
import { BootstrapSearchOptimizer } from "./optimize.js";

// The instruction covers some required terms, so rollouts score above zero
// and there is something for the bootstrapper to harvest.
const SEED = {
  instruction: "Answer the user question. hold ten seconds thirty days",
  demos: "",
};

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
    demoComponents: ["demos"] as const,
    maxMetricCalls: 600,
  };
}

describe("BootstrapSearchOptimizer", () => {
  test("needs no reflection model at all", async () => {
    // The whole point of this search: demos are harvested from rollouts the
    // metric already rewarded, so nothing here writes text.
    const result = await new BootstrapSearchOptimizer({
      candidates: 2,
      seed: 1,
    }).optimize(task());

    expect(result.bestCandidate.instruction).toBe(SEED.instruction);
  });

  test("always keeps the zero-shot candidate in the running", async () => {
    // Demonstrations can hurt, so a search that cannot return "no demos" can
    // only ever report a candidate it has no baseline for.
    const result = await new BootstrapSearchOptimizer({
      candidates: 1,
      seed: 1,
    }).optimize(task());

    expect(result.candidates[0]?.source).toBe("zeroShot");
  });

  test("harvests demos into every component that holds them", async () => {
    const result = await new BootstrapSearchOptimizer({
      candidates: 3,
      seed: 1,
    }).optimize(task());

    const bootstrapped = result.candidates.filter(
      (entry) => entry.source === "bootstrapped",
    );

    expect(bootstrapped.length).toBeGreaterThan(0);
    expect(
      bootstrapped.some((entry) => entry.candidate.demos.includes("<demo>")),
    ).toBe(true);
  });

  test("builds a labels-only candidate when gold outputs are available", async () => {
    // It costs no rollouts at all: the answer is known rather than produced.
    const result = await new BootstrapSearchOptimizer({
      candidates: 1,
      seed: 1,
    }).optimize({
      ...task(),
      goldOutput: (datum) => datum.required.join(" "),
    });

    expect(result.candidates.some((entry) => entry.source === "labeled")).toBe(
      true,
    );
  });

  test("reports the best candidate by validation score", async () => {
    const result = await new BootstrapSearchOptimizer({
      candidates: 3,
      seed: 1,
    }).optimize(task());

    const scores = result.candidates.map((entry) => entry.score);

    expect(result.bestScore).toBe(Math.max(...scores));
  });

  test("stops early once a candidate reaches the target score", async () => {
    const result = await new BootstrapSearchOptimizer({
      candidates: 16,
      seed: 1,
      stopAtScore: 0,
    }).optimize(task());

    expect(result.stopReason).toBe("scoreReached");
    expect(result.candidates.length).toBe(1);
  });

  test("stops when the rollout budget cannot fund another candidate", async () => {
    const result = await new BootstrapSearchOptimizer({
      candidates: 16,
      seed: 1,
    }).optimize({ ...task(), maxMetricCalls: 20 });

    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("stops when every candidate has been evaluated", async () => {
    const result = await new BootstrapSearchOptimizer({
      candidates: 2,
      seed: 1,
    }).optimize(task());

    expect(result.stopReason).toBe("candidatesExhausted");
  });

  test("counts the rollouts harvesting spent separately", async () => {
    const result = await new BootstrapSearchOptimizer({
      candidates: 2,
      seed: 1,
    }).optimize(task());

    expect(result.bootstrapMetricCalls).toBeGreaterThan(0);
    expect(result.metricCalls).toBeGreaterThan(result.bootstrapMetricCalls);
  });

  test("refuses a seed candidate that holds no demo component", async () => {
    await expect(
      new BootstrapSearchOptimizer({}).optimize({
        ...task(),
        demoComponents: [] as const,
      }),
    ).rejects.toThrow("at least one demoComponent");
  });

  test("resumes from a checkpoint without re-scoring what it already tried", async () => {
    const interrupted = await new BootstrapSearchOptimizer({
      candidates: 1,
      seed: 1,
    }).optimize({ ...task(), cache: false });

    const resumed = await new BootstrapSearchOptimizer({
      candidates: 3,
      seed: 1,
    }).optimize({
      ...task(),
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.candidates.slice(0, interrupted.candidates.length)).toEqual(
      interrupted.candidates,
    );
  });

  test("refuses a checkpoint taken against a different seed candidate", async () => {
    const interrupted = await new BootstrapSearchOptimizer({
      candidates: 1,
      seed: 1,
    }).optimize(task());

    await expect(
      new BootstrapSearchOptimizer({ candidates: 1, seed: 1 }).optimize({
        ...task(),
        seedCandidate: { instruction: "Different.", demos: "" },
        resumeFrom: interrupted.snapshot,
      }),
    ).rejects.toThrow("does not belong to this run");
  });
});

describe("BootstrapSearchOptimizer concurrency", () => {
  test("sweeps several candidates at the same time", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await new BootstrapSearchOptimizer({
      candidates: 4,
      concurrency: 4,
      seed: 1,
    }).optimize({ ...task(), adapter: tracked.adapter });

    expect(tracked.maxInFlight()).toBeGreaterThan(1);
  });

  test("sweeps them one at a time by default", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await new BootstrapSearchOptimizer({ candidates: 4, seed: 1 }).optimize({
      ...task(),
      adapter: tracked.adapter,
    });

    expect(tracked.maxInFlight()).toBe(1);
  });

  test("tries the same candidates whether or not the sweeps overlap", async () => {
    const serial = await new BootstrapSearchOptimizer({
      candidates: 6,
      seed: 1,
    }).optimize(task());
    const concurrent = await new BootstrapSearchOptimizer({
      candidates: 6,
      concurrency: 3,
      seed: 1,
    }).optimize(task());

    expect(concurrent.candidates).toEqual(serial.candidates);
    expect(concurrent.bestCandidate).toEqual(serial.bestCandidate);
    expect(concurrent.bestScore).toBe(serial.bestScore);
    expect(concurrent.metricCalls).toBe(serial.metricCalls);
    expect(concurrent.bootstrapMetricCalls).toBe(serial.bootstrapMetricCalls);
    expect(concurrent.stopReason).toBe(serial.stopReason);
  });

  test("harvests in plan order however the sweeps interleave", async () => {
    // Every harvest draws from the same random stream, so a run whose harvests
    // ran in a different order is a different run at the same seed.
    const run = async (pace: (candidate: string) => number) => {
      const result = await new BootstrapSearchOptimizer({
        candidates: 6,
        concurrency: 3,
        seed: 1,
      }).optimize({ ...task(), adapter: withPacing(baseAdapter(), pace) });

      return result.snapshot;
    };

    const shortestFirst = await run((candidate) => candidate.length % 5);
    const longestFirst = await run((candidate) => 5 - (candidate.length % 5));

    expect(shortestFirst.rngState).toBe(longestFirst.rngState);
    expect(shortestFirst.candidates).toEqual(longestFirst.candidates);
  });

  test("never spends past the budget when the sweeps overlap", async () => {
    const result = await new BootstrapSearchOptimizer({
      candidates: 8,
      concurrency: 4,
      seed: 1,
    }).optimize({ ...task(), maxMetricCalls: 40 });

    expect(result.metricCalls).toBeLessThanOrEqual(40);
  });

  test("stops at the target score without sweeping past it", async () => {
    // A target score is a serial stopping condition: a wave that had already
    // bought the candidates behind the winner would pay for readings the
    // search asked it not to take.
    const tracked = withOverlapTracking(baseAdapter());

    const result = await new BootstrapSearchOptimizer({
      candidates: 8,
      concurrency: 4,
      stopAtScore: 0.1,
      seed: 1,
    }).optimize({ ...task(), adapter: tracked.adapter });

    expect(result.stopReason).toBe("scoreReached");
    expect(tracked.maxInFlight()).toBe(1);
  });

  test("rejects a concurrency below one", () => {
    expect(() => new BootstrapSearchOptimizer({ concurrency: 0 })).toThrow(
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
