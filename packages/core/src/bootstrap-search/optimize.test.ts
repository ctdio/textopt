import { describe, expect, test } from "vitest";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "../testing.js";
import type { Adapter } from "../types.js";
import { BootstrapSearchOptimizer } from "./optimize.js";

/** An instance that names the split it belongs to, so an adapter can tell them apart. */
interface Split {
  id: number;
  kind: "train" | "validate";
}

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
  test("counts the tokens harvesting spent in the run's usage", async () => {
    // Harvesting runs the candidate over the training set, which is most of
    // what a demo-heavy run spends. Usage that omits it makes `maxCostUsd` a
    // ceiling on part of the run.
    const keyword = createKeywordAdapter();
    const adapter: Adapter<(typeof KEYWORD_EXAMPLES)[number], unknown, string> =
      {
        evaluate: async (args) => {
          const evaluation = await keyword.evaluate(args);
          return {
            ...evaluation,
            usage: args.batch.map(() => ({ inputTokens: 10, outputTokens: 5 })),
          };
        },
      };

    const result = await new BootstrapSearchOptimizer({
      candidates: 2,
      seed: 1,
    }).optimize({ ...task(), adapter });

    // One rollout is 10 input tokens, so usage must cover every rollout the
    // run paid for, harvesting included.
    expect(result.usage.inputTokens).toBe(result.metricCalls * 10);
  });

  test("keeps the held-out sweep out of the run's usage", async () => {
    // No ceiling bounds the held-out sweep: it runs once the search has already
    // stopped. Counting it in `usage` would describe a run that honoured
    // `maxCostUsd` as having overrun it.
    const result = await new BootstrapSearchOptimizer({
      candidates: 2,
      seed: 1,
    }).optimize({
      ...task(),
      adapter: pricedAdapter(),
      testSet: [
        { question: "held out, satisfied", required: ["answer"] },
        { question: "held out, unsatisfiable", required: ["zzz-never"] },
      ],
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

  test("does not restart candidate ids after a resume", async () => {
    // Reporters key rows by candidateId. Restarting the counter at zero makes a
    // resumed run's candidates collide with the interrupted run's in whatever
    // store the reporter is writing to.
    let before = 0;
    let after = 0;

    const interrupted = await new BootstrapSearchOptimizer({
      candidates: 4,
      seed: 1,
    }).optimize({
      ...harvestTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              before = event.bestCandidateId;
            }
          },
        },
      ],
    });

    await new BootstrapSearchOptimizer({ candidates: 8, seed: 1 }).optimize({
      ...harvestTask(),
      resumeFrom: interrupted.snapshot,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              after = event.bestCandidateId;
            }
          },
        },
      ],
    });

    expect(before).toBeGreaterThan(0);
    expect(after).toBeGreaterThanOrEqual(before);
  });

  test("carries usage already spent into a resumed run", async () => {
    // `maxCostUsd` is a ceiling on the run, not on the segment. A resumed run
    // that restarts its token accounting at zero lets an interrupted-and-
    // resumed loop spend the ceiling over and over.
    const priced = {
      ...task(),
      cache: false as const,
      adapter: pricedAdapter(),
    };

    const interrupted = await new BootstrapSearchOptimizer({
      candidates: 1,
      seed: 1,
    }).optimize(priced);

    const resumed = await new BootstrapSearchOptimizer({
      candidates: 3,
      seed: 1,
    }).optimize({ ...priced, resumeFrom: interrupted.snapshot });

    expect(interrupted.usage.inputTokens).toBeGreaterThan(0);
    // Ten input tokens a rollout, uncached, harvesting included, over both
    // segments.
    expect(resumed.usage.inputTokens).toBe(resumed.metricCalls * 10);
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

  test("waits for the sweeps it dispatched before reporting a failure", async () => {
    // A wave dispatches its sweeps and then reads them in order. Leaving on the
    // first failure abandons the ones behind it: they keep calling the adapter,
    // and keep spending, after the caller has already been handed the error.
    const TRAIN: Split[] = Array.from({ length: 4 }, (_, id) => ({
      id,
      kind: "train",
    }));
    const VALIDATE: Split[] = Array.from({ length: 2 }, (_, id) => ({
      id,
      kind: "validate",
    }));

    let sweeps = 0;
    let inFlight = 0;

    const running = new BootstrapSearchOptimizer({
      candidates: 4,
      concurrency: 4,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: TRAIN,
      validationSet: VALIDATE,
      demoComponents: ["demos"] as const,
      maxMetricCalls: 600,
      adapter: {
        evaluate: async ({ batch }) => {
          const scored = {
            outputs: batch.map((datum) => `answer ${datum.id}`),
            scores: batch.map(() => 0.5),
          };
          if (batch[0]?.kind !== "validate") {
            return scored;
          }

          sweeps += 1;
          const mine = sweeps;
          inFlight += 1;
          try {
            await new Promise((resolve) => setTimeout(resolve, mine * 5));
            if (mine === 2) {
              throw new Error("boom");
            }
            return scored;
          } finally {
            inFlight -= 1;
          }
        },
      },
    });

    await expect(running).rejects.toThrow("boom");
    expect(inFlight).toBe(0);
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

/**
 * Scores the demo block rather than the instruction. The keyword fixture
 * grades a candidate's own text, which leaves a demonstration search nothing
 * to harvest and no acceptance to report.
 */
function harvestableAdapter(): Adapter<{ id: number }, unknown, string> {
  return {
    evaluate: ({ batch, candidate }) => ({
      outputs: batch.map((datum) => `answer ${datum.id}`),
      scores: batch.map((datum) =>
        candidate.demos.includes("<demo>") || datum.id % 2 === 0 ? 1 : 0,
      ),
    }),
  };
}

const HARVEST_TRAINSET = Array.from({ length: 8 }, (_, id) => ({ id }));

function harvestTask() {
  return {
    seedCandidate: { instruction: "Answer.", demos: "" },
    trainingSet: HARVEST_TRAINSET,
    adapter: harvestableAdapter(),
    demoComponents: ["demos"] as const,
    maxMetricCalls: 600,
  };
}

describe("BootstrapSearchOptimizer reporting", () => {
  test("reports the seed as candidate 0, before any improvement", async () => {
    // The seed is what every later candidate is read against. A report that
    // starts at the first improvement has nothing to compare it to.
    const accepted: { id: number; candidate: Record<string, string> }[] = [];

    await new BootstrapSearchOptimizer({ candidates: 2, seed: 1 }).optimize({
      ...task(),
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
    const accepted: string[] = [];

    await new BootstrapSearchOptimizer({ candidates: 4, seed: 1 }).optimize({
      ...harvestTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              accepted.push(event.candidate.demos);
            }
          },
        },
      ],
    });

    expect(accepted.length).toBeGreaterThan(0);
    expect(accepted.at(-1)).toContain("<demo>");
  });

  test("reports a per-instance row aligned with the validation set", async () => {
    const rows: (number | undefined)[][] = [];

    await new BootstrapSearchOptimizer({ candidates: 4, seed: 1 }).optimize({
      ...harvestTask(),
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
      expect(row).toHaveLength(HARVEST_TRAINSET.length);
    }
  });

  test("names the winner in finish with an id an acceptance carried", async () => {
    const acceptedIds: number[] = [];
    let bestCandidateId: number | undefined;

    await new BootstrapSearchOptimizer({ candidates: 4, seed: 1 }).optimize({
      ...harvestTask(),
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

    await new BootstrapSearchOptimizer({ candidates: 2, seed: 1 }).optimize({
      ...task(),
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
});

/** The base adapter, reporting every rollout as a failure it caught. */
function failingAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const keyword = createKeywordAdapter();
  return {
    evaluate: (args) => ({
      ...keyword.evaluate(args),
      scores: args.batch.map(() => 0),
      failed: args.batch.map(() => true),
    }),
  };
}

describe("BootstrapSearchOptimizer failure reporting", () => {
  test("reports the failures nothing classified as infrastructure", async () => {
    // Every optimizer shares one evaluator, and every one of them has to carry
    // what it saw onto the result. A search that drops the count reports a
    // run of zeros with nothing to say where they came from.
    const result = await new BootstrapSearchOptimizer({
      candidates: 2,
      seed: 1,
    }).optimize({
      ...task(),
      adapter: failingAdapter(),
    });

    expect(result.warnings.map((warning) => warning.code)).toContain(
      "unclassifiedFailures",
    );
  });
});
