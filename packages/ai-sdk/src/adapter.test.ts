import type { EvaluationContext } from "textopt";
import type { GepaAdapter } from "textopt/gepa";
import { GepaOptimizer } from "textopt/gepa";
import { createKeywordReflector } from "textopt/testing";
import { generateText } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { describe, expect, test } from "vitest";
import type { AiSdkResultLike, AiSdkTrace } from "./adapter.js";
import { createAiSdkAdapter, summarizeRun } from "./adapter.js";

interface Question {
  question: string;
  answer: string;
}

const QUESTIONS: Question[] = [
  { question: "capital of france?", answer: "paris" },
  { question: "capital of japan?", answer: "tokyo" },
];

const RUN: EvaluationContext = {
  iteration: 0,
  phase: "minibatch",
  split: "train",
  candidateId: 0,
};

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
  test("reports token usage per rollout, priced when a price list is given", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => resultFor(datum.answer),
      score: () => ({ score: 1 }),
      pricing: { inputPerMillionUsd: 3, outputPerMillionUsd: 15 },
    });

    const evaluation = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "answer" },
      captureTraces: false,
      run: RUN,
    });

    expect(evaluation.usage).toEqual([
      { inputTokens: 10, outputTokens: 4, totalTokens: 14, costUsd: 0.00009 },
      { inputTokens: 10, outputTokens: 4, totalTokens: 14, costUsd: 0.00009 },
    ]);
  });

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
      run: RUN,
    });

    expect(result.outputs).toEqual(["paris", "tokyo"]);
    expect(result.scores).toEqual([1, 1]);
  });

  test("passes the candidate to the run function", async () => {
    const seen: string[] = [];
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ candidate, datum }) => {
        seen.push(candidate.system);
        return resultFor(datum.answer);
      },
      score: () => ({ score: 1 }),
    });

    await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "revision-3" },
      captureTraces: false,
      run: RUN,
    });

    expect(seen).toEqual(["revision-3", "revision-3"]);
  });

  test("passes the run context to the run function", async () => {
    // Without it a run is thousands of indistinguishable rollouts, and no
    // trace can be tied back to the iteration whose score moved.
    const seen: EvaluationContext[] = [];
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum, run }) => {
        seen.push(run);
        return resultFor(datum.answer);
      },
      score: () => ({ score: 1 }),
    });

    await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: false,
      run: { iteration: 4, phase: "validation", split: "val", candidateId: 7 },
    });

    expect(seen).toEqual([
      { iteration: 4, phase: "validation", split: "val", candidateId: 7 },
      { iteration: 4, phase: "validation", split: "val", candidateId: 7 },
    ]);
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
        run: RUN,
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
      run: RUN,
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

  test("nests a multi-step trace under the record's evidence slot", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => ({
        text: datum.answer,
        steps: [{ text: "thinking" }, { text: datum.answer }],
      }),
      score: () => ({ score: 0 }),
    });

    const evaluation = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { system: "x" },
      batch: QUESTIONS,
      evaluation,
      componentsToUpdate: ["system"],
    });
    expect(dataset.system?.[0]?.evidence).toEqual({
      trace: [
        { index: 0, text: "thinking" },
        { index: 1, text: "paris" },
      ],
    });
  });

  test("nests a run failure's message under the record's evidence slot", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async () => {
        throw new Error("rate limited");
      },
      score: () => ({ score: 1 }),
    });

    const evaluation = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { system: "x" },
      batch: QUESTIONS,
      evaluation,
      componentsToUpdate: ["system"],
    });
    expect(dataset.system?.[0]?.evidence).toEqual({
      trace: [],
      error: "rate limited",
    });
  });

  test("omits evidence for a single-step run that did not fail", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => resultFor(datum.answer),
      score: () => ({ score: 0 }),
    });

    const evaluation = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { system: "x" },
      batch: QUESTIONS,
      evaluation,
      componentsToUpdate: ["system"],
    });

    expect(dataset.system?.[0]?.evidence).toBeUndefined();
  });

  test("uses a custom buildRecord when one is supplied", async () => {
    const adapter = createAiSdkAdapter<Question, string>({
      run: async ({ datum }) => resultFor(datum.answer),
      score: () => ({ score: 0.5 }),
      buildRecord: ({ datum, output, score, component }) => ({
        inputs: { asked: datum.question, for: component },
        generatedOutputs: output,
        feedback: "custom",
        score,
      }),
    });

    const evaluation = await adapter.evaluate({
      batch: QUESTIONS,
      candidate: { system: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { system: "x" },
      batch: QUESTIONS,
      evaluation,
      componentsToUpdate: ["system"],
    });

    expect(dataset.system?.[0]).toEqual({
      inputs: { asked: "capital of france?", for: "system" },
      generatedOutputs: "paris",
      feedback: "custom",
      score: 0.5,
    });
  });

  test("plugs into a component-keyed GEPA task while staying key-agnostic", async () => {
    // The annotation is the point: `createAiSdkAdapter` never sees the
    // component names, so its adapter must still satisfy a keyed task's slot.
    const keyed: GepaAdapter<
      Question,
      AiSdkTrace,
      string | null,
      "system" | "style"
    > = createAiSdkAdapter<Question, string>({
      run: async ({ candidate, datum }) =>
        resultFor(`${candidate.system}:${datum.answer}`),
      score: () => ({ score: 1 }),
    });

    const result = await keyed.evaluate({
      batch: QUESTIONS,
      candidate: { system: "terse", style: "plain" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.outputs).toEqual(["terse:paris", "terse:tokyo"]);
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

/**
 * Everything above builds its own `AiSdkResultLike`. These run the real
 * `generateText` against the SDK's own mock provider instead, so the trace the
 * adapter extracts is taken from what the SDK actually returns rather than
 * from this file's idea of it.
 */
describe("against a real generateText call", () => {
  function echoingModel(): MockLanguageModelV4 {
    return new MockLanguageModelV4({
      // The system prompt is the candidate, so echoing it back is what gives
      // the optimizer a gradient to climb.
      doGenerate: async ({ prompt }) => {
        const system = prompt.find((message) => message.role === "system");
        const content = system?.content;
        const text =
          typeof content === "string" ? content : JSON.stringify(content ?? "");

        return {
          content: [{ type: "text" as const, text }],
          // Provider-level shapes, which are nested and differ from the flat
          // ones `generateText` reports back to a caller.
          finishReason: { unified: "stop" as const, raw: "stop" },
          usage: {
            inputTokens: { total: 8, noCache: 8, cacheRead: 0, cacheWrite: 0 },
            outputTokens: { total: 4, text: 4, reasoning: 0 },
          },
          warnings: [],
        };
      },
    });
  }

  test("traces what the SDK actually returned", async () => {
    const adapter = createAiSdkAdapter<Question>({
      run: ({ candidate, datum, signal }) =>
        generateText({
          model: echoingModel(),
          system: candidate.system ?? "",
          prompt: datum.question,
          abortSignal: signal,
        }),
      score: ({ output }) => ({ score: output === "" ? 0 : 1 }),
    });

    const evaluation = await adapter.evaluate({
      batch: [QUESTIONS[0] as Question],
      candidate: { system: "answer plainly" },
      captureTraces: true,
      run: RUN,
    });

    expect(evaluation.outputs[0]).toContain("answer plainly");
    expect(evaluation.scores).toEqual([1]);

    // The SDK reports these back flattened, whatever shape the provider used.
    const trace = evaluation.trajectories?.[0];
    expect(trace?.steps).toHaveLength(1);
    expect(trace?.steps[0]?.finishReason).toBe("stop");
    expect(trace?.usage).toMatchObject({
      inputTokens: 8,
      outputTokens: 4,
      totalTokens: 12,
    });
  });

  test("improves the seed candidate over a full optimization run", async () => {
    const adapter = createAiSdkAdapter<Question>({
      run: ({ candidate, datum }) =>
        generateText({
          model: echoingModel(),
          system: candidate.instruction ?? "",
          prompt: datum.question,
        }),
      score: ({ datum, output }) => {
        const covered = (output ?? "").toLowerCase().includes(datum.answer);

        return covered
          ? { score: 1, feedback: "All required terms present." }
          : { score: 0, feedback: `Missing required terms: ${datum.answer}` };
      },
    });

    const result = await new GepaOptimizer({
      minibatchSize: 2,
      seed: 7,
    }).optimize({
      seedCandidate: { instruction: "answer the question" },
      trainingSet: QUESTIONS,
      adapter,
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });

    // The seed answers nothing, so a perfect score is movement rather than a
    // starting point the run never had to earn.
    expect(result.candidates[0]?.aggregateScore).toBe(0);
    expect(result.bestScore).toBe(1);
    expect(result.bestCandidate.instruction).toContain("paris");
    expect(result.bestCandidate.instruction).toContain("tokyo");
  });
});
