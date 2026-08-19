import { describe, expect, test } from "vitest";
import type { Optimizer, OptimizerResult } from "../optimizer.js";
import {
  KEYWORD_EXAMPLES,
  createKeywordAdapter,
  createSamplingReflector,
} from "../testing.js";
import type { Adapter } from "../types.js";
import { RandomSearchOptimizer } from "./optimize.js";
import type { RandomSearchStopReason } from "./optimize.js";

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
