import { describe, expect, test } from "vitest";
import {
  formatDemos,
  harvestFewShotExamples,
  parseDemos,
  replaceDemos,
} from "./demos.js";
import type { Demo } from "./demos.js";
import { createSeededRng } from "./rng.js";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "./testing.js";
import type { Adapter } from "./types.js";

const DEMOS = [
  {
    input: { question: "How do I reset a device?" },
    output: "Hold for ten seconds.",
  },
  { input: { question: "What is the refund window?" }, output: "Thirty days." },
];

describe("formatDemos", () => {
  test("renders every demo's input and output", () => {
    const block = formatDemos(DEMOS);

    expect(block).toContain("How do I reset a device?");
    expect(block).toContain("Hold for ten seconds.");
    expect(block).toContain("Thirty days.");
  });

  test("returns an empty string for no demos", () => {
    expect(formatDemos([])).toBe("");
  });

  test("uses a custom renderer when given one", () => {
    const block = formatDemos(DEMOS, {
      render: ({ index }) => `demo ${index}`,
    });

    expect(block).toContain("demo 0");
    expect(block).toContain("demo 1");
  });
});

describe("parseDemos", () => {
  test("recovers the demos a formatted block was built from", () => {
    expect(parseDemos(formatDemos(DEMOS))).toEqual(DEMOS);
  });

  test("round trips an output that itself contains a demo block", () => {
    const echoed = formatDemos([{ input: { q: "inner" }, output: "answer" }]);
    const demos = [{ input: { q: "outer" }, output: echoed }];

    expect(parseDemos(formatDemos(demos))).toEqual(demos);
  });

  test("reads exactly one demo from a block whose output echoes one", () => {
    const echoed = formatDemos([{ input: { q: "inner" }, output: "answer" }]);
    const block = formatDemos([{ input: { q: "outer" }, output: echoed }]);

    expect(parseDemos(block)).toHaveLength(1);
  });

  test("returns nothing for text that holds no demos", () => {
    expect(parseDemos("Just an ordinary instruction.")).toEqual([]);
  });

  test("survives text written around the block", () => {
    const text = `Follow these examples.\n\n${formatDemos(DEMOS)}\n\nBe brief.`;

    expect(parseDemos(text)).toHaveLength(2);
  });
});

describe("replaceDemos", () => {
  test("keeps the text written around the block it replaces", () => {
    const text = `Follow these examples.\n\n${formatDemos([DEMOS[0] as Demo])}\n\nBe brief.`;

    const next = replaceDemos({ text, demos: DEMOS });

    expect(next).toContain("Follow these examples.");
    expect(next).toContain("Be brief.");
    expect(parseDemos(next)).toEqual(DEMOS);
  });

  test("appends to text that holds no demos yet", () => {
    const next = replaceDemos({ text: "Be brief.", demos: DEMOS });

    expect(next).toContain("Be brief.");
    expect(parseDemos(next)).toEqual(DEMOS);
  });

  test("leaves only the surrounding text when given no demos", () => {
    const text = `Be brief.\n\n${formatDemos(DEMOS)}`;

    const next = replaceDemos({ text, demos: [] });

    expect(next).toBe("Be brief.");
    expect(parseDemos(next)).toEqual([]);
  });

  test("collapses demos scattered through the text into one block", () => {
    const text = [
      formatDemos([DEMOS[0] as Demo]),
      "Be brief.",
      formatDemos([DEMOS[1] as Demo]),
    ].join("\n\n");

    const next = replaceDemos({ text, demos: DEMOS });

    expect(next.match(/<demo>/g)).toHaveLength(2);
    expect(next).toContain("Be brief.");
  });

  test("renders replacements with a custom renderer", () => {
    const next = replaceDemos({
      text: "",
      demos: DEMOS,
      render: ({ index }) => `demo ${index}`,
    });

    expect(next).toContain("demo 0");
  });
});

describe("harvestFewShotExamples", () => {
  const adapter = (): Adapter<
    (typeof KEYWORD_EXAMPLES)[number],
    unknown,
    string
  > => createKeywordAdapter();

  test("keeps only rollouts that clear the score threshold", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      // Answers two of the four instances perfectly and neither of the others.
      candidate: { instruction: "hold ten seconds ticket portal" },
      trainingSet: KEYWORD_EXAMPLES,
      minScore: 1,
    });

    expect(result.demos).toHaveLength(2);
    for (const demo of result.demos) {
      expect(demo.score).toBe(1);
    }
  });

  test("keeps every rollout the metric rewarded when given no threshold", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      // Half credit on three instances and nothing on the fourth.
      candidate: { instruction: "hold ticket billing" },
      trainingSet: KEYWORD_EXAMPLES,
    });

    expect(result.demos).toHaveLength(3);
    for (const demo of result.demos) {
      expect(demo.score).toBe(0.5);
    }
  });

  test("keeps nothing the metric scored at zero when given no threshold", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      candidate: { instruction: "nothing useful here" },
      trainingSet: KEYWORD_EXAMPLES,
    });

    expect(result.demos).toEqual([]);
  });

  test("keeps nothing when the candidate never clears the threshold", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      candidate: { instruction: "nothing useful here" },
      trainingSet: KEYWORD_EXAMPLES,
      minScore: 1,
    });

    expect(result.demos).toEqual([]);
    expect(result.block).toBe("");
  });

  test("stops once it has enough demos", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      candidate: {
        instruction:
          "hold ten seconds ticket portal thirty days billing prorated",
      },
      trainingSet: KEYWORD_EXAMPLES,
      minScore: 1,
      maxDemos: 2,
      batchSize: 1,
    });

    expect(result.demos).toHaveLength(2);
    // Two instances answered, two never attempted.
    expect(result.metricCalls).toBe(2);
  });

  test("reports what it spent", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      candidate: { instruction: "nothing useful here" },
      trainingSet: KEYWORD_EXAMPLES,
    });

    expect(result.metricCalls).toBe(KEYWORD_EXAMPLES.length);
    expect(result.attempted).toBe(KEYWORD_EXAMPLES.length);
  });

  test("stops at the metric call ceiling", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      candidate: { instruction: "nothing useful here" },
      trainingSet: KEYWORD_EXAMPLES,
      maxMetricCalls: 2,
      batchSize: 1,
    });

    expect(result.metricCalls).toBe(2);
  });

  test("produces a block the seed candidate can carry directly", async () => {
    const result = await harvestFewShotExamples({
      adapter: adapter(),
      candidate: { instruction: "hold ten seconds" },
      trainingSet: KEYWORD_EXAMPLES,
      minScore: 1,
    });

    expect(parseDemos(result.block)).toHaveLength(result.demos.length);
  });

  test("samples the trainingSet in a reproducible order when given an rng", async () => {
    const run = async () =>
      (
        await harvestFewShotExamples({
          adapter: adapter(),
          candidate: { instruction: "hold ten seconds ticket portal" },
          trainingSet: KEYWORD_EXAMPLES,
          minScore: 1,
          maxDemos: 1,
          batchSize: 1,
          rng: createSeededRng(7),
        })
      ).demos.map((demo) => JSON.stringify(demo.input));

    expect(await run()).toEqual(await run());
  });

  test("refuses an empty trainingSet", async () => {
    await expect(
      harvestFewShotExamples({
        adapter: adapter(),
        candidate: { instruction: "x" },
        trainingSet: [],
      }),
    ).rejects.toThrow(/trainingSet/);
  });
});
