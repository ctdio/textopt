import { describe, expect, test } from "vitest";
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

/**
 * A stand-in for the advice model: it reads the missing terms out of whichever
 * trajectory the prompt carries and hands each component the terms it lacks.
 */
function createAdviceReflector(): TextModel {
  return async ({ prompt }) => {
    const components = (
      prompt.match(/<components>\n([\s\S]*?)\n<\/components>/)?.[1] ?? ""
    )
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

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
      onEvent: (event: SimbaEvent) => {
        if (event.type === "stepStart") {
          steps.push(event.step);
        }
      },
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
      onEvent: (event: SimbaEvent) => {
        if (event.type === "stepStart" && event.step === 1) {
          controller.abort();
        }
      },
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
