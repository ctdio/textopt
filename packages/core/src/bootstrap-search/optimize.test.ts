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
