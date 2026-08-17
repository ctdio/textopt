import type { generateText } from "ai";
import { describe, expect, test } from "vitest";
import type { AiSdkResultLike } from "./adapter.js";
import { createAiSdkAdapter, summarizeRun } from "./adapter.js";

interface Question {
  question: string;
  answer: string;
}

const QUESTIONS: Question[] = [
  { question: "capital of france?", answer: "paris" },
  { question: "capital of japan?", answer: "tokyo" },
];

// Compile-time proof that a real AI SDK generateText result satisfies the
// structural type this adapter accepts. Fails typecheck if the SDK drifts.
type GenerateTextIsCompatible =
  Awaited<ReturnType<typeof generateText>> extends AiSdkResultLike
    ? true
    : "generateText result no longer satisfies AiSdkResultLike";

function resultFor(text: string): AiSdkResultLike {
  return {
    text,
    finishReason: "stop",
    usage: { inputTokens: 10, outputTokens: 4, totalTokens: 14 },
    steps: [
      {
        text,
        finishReason: "stop",
        toolCalls: [{ toolName: "lookup", input: { q: text } }],
        toolResults: [{ toolName: "lookup", output: { hit: true } }],
      },
    ],
  };
}

describe("createAiSdkAdapter", () => {
  test("accepts the shape a real generateText call returns", () => {
    const compatible: GenerateTextIsCompatible = true;

    expect(compatible).toBe(true);
  });

  test("uses result.text as the output by default", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => resultFor(datum.answer),
      score: ({ datum, output }) => ({
        score: output === datum.answer ? 1 : 0,
      }),
    });

    const result = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "be terse" },
      captureTraces: false,
    });

    expect(result.outputs).toEqual(["paris", "tokyo"]);
    expect(result.scores).toEqual([1, 1]);
  });

  test("passes the candidate to the run function", async () => {
    const seen: string[] = [];
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ candidate, datum }) => {
        seen.push(candidate.system as string);
        return resultFor(datum.answer);
      },
      score: () => ({ score: 1 }),
    });

    await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "revision-3" },
      captureTraces: false,
    });

    expect(seen).toEqual(["revision-3", "revision-3"]);
  });

  test("supports a custom output extractor for structured generation", async () => {
    const adapter = createAiSdkAdapter<Question, { city: string }>({
      run: async ({ datum }) => ({
        text: "",
        object: { city: datum.answer },
      }),
      toOutput: (result) => (result as { object: { city: string } }).object,
      score: ({ datum, output }) => ({
        score: output?.city === datum.answer ? 1 : 0,
      }),
    });

    const result = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: false,
    });

    expect(result.outputs).toEqual([{ city: "paris" }, { city: "tokyo" }]);
  });

  test("captures a step trace including tool calls", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => resultFor(datum.answer),
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: true,
    });

    const trace = result.trajectories?.[0];

    expect(trace?.steps).toHaveLength(1);
    expect(trace?.steps[0]?.toolCalls).toEqual([
      { toolName: "lookup", input: { q: "paris" } },
    ]);
    expect(trace?.usage?.totalTokens).toBe(14);
  });

  test("scores a failed run as zero with the error as feedback", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async () => {
        throw new Error("rate limited");
      },
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: true,
    });

    expect(result.scores).toEqual([0, 0]);
    expect(result.feedback?.[0]).toContain("rate limited");
    expect(result.outputs).toEqual([null, null]);
  });

  test("respects the concurrency limit", async () => {
    let active = 0;
    let peak = 0;

    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => {
        active += 1;
        peak = Math.max(peak, active);
        await new Promise((resolve) => setTimeout(resolve, 5));
        active -= 1;
        return resultFor(datum.answer);
      },
      score: () => ({ score: 1 }),
      concurrency: 2,
    });

    await adapter.evaluate({
      batch: Array.from({ length: 8 }, () => QUESTIONS[0] as Question),
      candidate: { system: "x" },
      captureTraces: false,
    });

    expect(peak).toBeLessThanOrEqual(2);
  });

  test("keeps results aligned with the batch when items finish out of order", async () => {
    // Every item is distinct and the first one finishes last, so a result
    // array assembled in completion order would be visibly wrong.
    const batch = [
      { question: "slow", answer: "slow-answer" },
      { question: "medium", answer: "medium-answer" },
      { question: "fast", answer: "fast-answer" },
    ];
    const delays: Record<string, number> = { slow: 30, medium: 15, fast: 1 };

    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => {
        await new Promise((resolve) =>
          setTimeout(resolve, delays[datum.question]),
        );
        return resultFor(datum.answer);
      },
      score: ({ datum, output }) => ({
        score: output === datum.answer ? 1 : 0,
      }),
      concurrency: 3,
    });

    const result = await adapter.evaluate({
      batch,
      candidate: { system: "x" },
      captureTraces: false,
    });

    expect(result.outputs).toEqual([
      "slow-answer",
      "medium-answer",
      "fast-answer",
    ]);
    expect(result.scores).toEqual([1, 1, 1]);
  });

  test("scores a failing scorer as zero without aborting the batch", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => resultFor(datum.answer),
      score: ({ datum }) => {
        if (datum.answer === "paris") {
          throw new Error("judge rate limited");
        }
        return { score: 1 };
      },
    });

    const result = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: false,
    });

    expect(result.scores).toEqual([0, 1]);
    expect(result.feedback?.[0]).toContain("Scoring failed");
    expect(result.feedback?.[0]).toContain("judge rate limited");
  });

  test("marks a run failure transient when isTransient says so", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async () => {
        throw new Error("429 rate limit exceeded");
      },
      score: () => ({ score: 1 }),
      isTransient: (err) => (err as Error).message.includes("429"),
    });

    const result = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: false,
    });

    expect(result.transient).toEqual([true, true]);
    expect(result.scores).toEqual([0, 0]);
  });

  test("leaves a run failure non-transient by default", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async () => {
        throw new Error("429 rate limit exceeded");
      },
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: false,
    });

    expect(result.transient).toBeUndefined();
  });

  test("propagates an abort instead of scoring it as a failed run", async () => {
    const controller = new AbortController();
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ signal }) => {
        controller.abort();
        signal?.throwIfAborted();
        return resultFor("unreachable");
      },
      score: () => ({ score: 1 }),
    });

    await expect(
      adapter.evaluate({
        batch: QUESTIONS,
        candidate: { system: "x" },
        captureTraces: false,
        signal: controller.signal,
      }),
    ).rejects.toThrow(/abort/i);
  });

  test("builds reflective records carrying feedback and the trace", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => resultFor(datum.answer),
      score: ({ datum }) => ({
        score: 0,
        feedback: `The answer must be ${datum.answer}`,
      }),
    });

    const evaluation = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: true,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { system: "x" },
      batch: QUESTIONS,
      evaluation,
      componentsToUpdate: ["system"],
    });

    expect(dataset.system).toHaveLength(2);
    expect(dataset.system?.[0]?.feedback).toBe("The answer must be paris");
    expect(dataset.system?.[0]?.inputs).toEqual(QUESTIONS[0]);
  });
});

describe("summarizeRun", () => {
  test("reads v4-style tool call fields as well as v5+", () => {
    const trace = summarizeRun({
      text: "done",
      steps: [
        {
          text: "done",
          toolCalls: [{ toolName: "search", args: { q: "x" } }],
          toolResults: [{ toolName: "search", result: "found" }],
        },
      ],
    });

    expect(trace.steps[0]?.toolCalls).toEqual([
      { toolName: "search", input: { q: "x" } },
    ]);
    expect(trace.steps[0]?.toolResults).toEqual([
      { toolName: "search", output: "found" },
    ]);
  });

  test("falls back to a single synthetic step when steps are absent", () => {
    const trace = summarizeRun({ text: "just text" });

    expect(trace.steps).toHaveLength(1);
    expect(trace.steps[0]?.text).toBe("just text");
  });

  test("falls back to a synthetic step when steps is an empty array", () => {
    const trace = summarizeRun({ text: "the final answer", steps: [] });

    expect(trace.steps).toHaveLength(1);
    expect(trace.steps[0]?.text).toBe("the final answer");
  });
});
