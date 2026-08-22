import type { EvaluationContext } from "textopt";
import { GepaOptimizer } from "textopt/gepa";
import { createKeywordReflector } from "textopt/testing";
import type { CallbackManager } from "@langchain/core/callbacks/manager";
import { AIMessage } from "@langchain/core/messages";
import type { ChatGeneration } from "@langchain/core/outputs";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import type { RunnableConfig } from "@langchain/core/runnables";
import { RunnableLambda } from "@langchain/core/runnables";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { describe, expect, test } from "vitest";
import { createLangChainAdapter } from "./adapter.js";

interface Ticket {
  text: string;
  expected: string;
}

const TICKETS: Ticket[] = [
  { text: "my printer is on fire", expected: "hardware" },
  { text: "charged twice this month", expected: "billing" },
];

const RUN: EvaluationContext = {
  iteration: 0,
  phase: "minibatch",
  split: "train",
  candidateId: 0,
};

function echoRunnable(instruction: string) {
  return RunnableLambda.from((input: { text: string }) =>
    `${instruction}|${input.text}`.trim(),
  );
}

/** A chain whose model call carries a run name, so a trace step can be attributed. */
function tracedClassifier() {
  return ChatPromptTemplate.fromMessages([
    ["system", "Classify the ticket."],
    ["human", "{text}"],
  ])
    .pipe(
      new FakeListChatModel({ responses: ["hardware", "billing"] }).withConfig({
        runName: "classifier",
      }),
    )
    .pipe(new StringOutputParser());
}

/**
 * Opens two tool spans through LangChain's own callback manager and only closes
 * one, reproducing a chain torn down mid-tool.
 */
function buildRunnableWithOrphanedToolSpan() {
  return RunnableLambda.from(
    async (input: { text: string }, config?: { callbacks?: unknown }) => {
      const manager = config?.callbacks as CallbackManager;

      await manager.handleToolStart(
        { lc: 1, type: "not_implemented", id: ["abandoned"] },
        input.text,
        undefined,
        undefined,
        undefined,
        undefined,
        "abandoned",
      );

      const finished = await manager.handleToolStart(
        { lc: 1, type: "not_implemented", id: ["finished"] },
        input.text,
        undefined,
        undefined,
        undefined,
        undefined,
        "finished",
      );
      await finished.handleToolEnd("ok");

      return "done";
    },
  );
}

describe("createLangChainAdapter", () => {
  test("reports the tokens its model spans accounted for, priced when asked", async () => {
    const runnable = RunnableLambda.from(
      async (input: { text: string }, config?: { callbacks?: unknown }) => {
        const manager = config?.callbacks as CallbackManager;
        const run = await manager.handleLLMStart(
          { lc: 1, type: "not_implemented", id: ["counter"] },
          [input.text],
        );
        await run[0]?.handleLLMEnd({
          generations: [[{ text: "hardware" }]],
          llmOutput: {
            tokenUsage: { promptTokens: 10, completionTokens: 4 },
          },
        });
        return "hardware";
      },
    );

    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => runnable,
      score: () => ({ score: 1 }),
      pricing: { inputPerMillionUsd: 3, outputPerMillionUsd: 15 },
    });

    const evaluation = await adapter.evaluate({
      batch: [TICKETS[0] as Ticket],
      candidate: { system: "classify" },
      captureTraces: false,
      run: RUN,
    });

    expect(evaluation.usage).toEqual([
      { inputTokens: 10, outputTokens: 4, totalTokens: 14, costUsd: 0.00009 },
    ]);
  });

  test("falls back to the legacy total when the message-level shape counts nothing", async () => {
    // Some integrations attach a zeroed `usage_metadata` to a generation whose
    // real total only ever reaches `llmOutput`. Reading its presence as a count
    // reports zero for a call that spent.
    const generation: ChatGeneration = {
      text: "hardware",
      message: new AIMessage({
        content: "hardware",
        usage_metadata: {
          input_tokens: 0,
          output_tokens: 0,
          total_tokens: 0,
        },
      }),
    };
    const runnable = RunnableLambda.from(
      async (input: { text: string }, config?: { callbacks?: unknown }) => {
        const manager = config?.callbacks as CallbackManager;
        const run = await manager.handleLLMStart(
          { lc: 1, type: "not_implemented", id: ["counter"] },
          [input.text],
        );
        await run[0]?.handleLLMEnd({
          generations: [[generation]],
          llmOutput: {
            tokenUsage: {
              promptTokens: 10,
              completionTokens: 4,
              totalTokens: 14,
            },
          },
        });
        return "hardware";
      },
    );

    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => runnable,
      score: () => ({ score: 1 }),
    });

    const evaluation = await adapter.evaluate({
      batch: [TICKETS[0] as Ticket],
      candidate: { system: "classify" },
      captureTraces: false,
      run: RUN,
    });

    expect(evaluation.usage).toEqual([
      { inputTokens: 10, outputTokens: 4, totalTokens: 14 },
    ]);
  });

  test("counts a provider reporting both token shapes only once", async () => {
    // Newer chat models populate `usage_metadata` while the integration still
    // fills the legacy `llmOutput.tokenUsage` for the same call. Reading both
    // is what makes this portable; adding both bills the run twice.
    const generation: ChatGeneration = {
      text: "hardware",
      message: new AIMessage({
        content: "hardware",
        usage_metadata: {
          input_tokens: 10,
          output_tokens: 4,
          total_tokens: 14,
        },
      }),
    };
    const runnable = RunnableLambda.from(
      async (input: { text: string }, config?: { callbacks?: unknown }) => {
        const manager = config?.callbacks as CallbackManager;
        const run = await manager.handleLLMStart(
          { lc: 1, type: "not_implemented", id: ["counter"] },
          [input.text],
        );
        await run[0]?.handleLLMEnd({
          generations: [[generation]],
          llmOutput: {
            tokenUsage: {
              promptTokens: 10,
              completionTokens: 4,
              totalTokens: 14,
            },
          },
        });
        return "hardware";
      },
    );

    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => runnable,
      score: () => ({ score: 1 }),
    });

    const evaluation = await adapter.evaluate({
      batch: [TICKETS[0] as Ticket],
      candidate: { system: "classify" },
      captureTraces: false,
      run: RUN,
    });

    expect(evaluation.usage).toEqual([
      { inputTokens: 10, outputTokens: 4, totalTokens: 14 },
    ]);
  });

  test("reports usage a scorer accounted for when the run itself counted none", async () => {
    // A judge called from `score` is the expensive half of some setups, and
    // the model under optimization may report nothing at all.
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => RunnableLambda.from(async () => "hardware"),
      score: () => ({
        score: 1,
        usage: { inputTokens: 10, outputTokens: 4, costUsd: 0.25 },
      }),
    });

    const evaluation = await adapter.evaluate({
      batch: [TICKETS[0] as Ticket],
      candidate: { system: "classify" },
      captureTraces: false,
      run: RUN,
    });

    expect(evaluation.usage).toEqual([
      { inputTokens: 10, outputTokens: 4, costUsd: 0.25 },
    ]);
  });

  test("scores every item in the batch in order", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: (candidate) => echoRunnable(candidate.instruction ?? ""),
      toInput: (datum) => ({ text: datum.text }),
      score: ({ datum, output }) => ({
        score: output?.includes(datum.text) === true ? 1 : 0,
      }),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "classify" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.scores).toEqual([1, 1]);
    expect(result.outputs).toEqual([
      "classify|my printer is on fire",
      "classify|charged twice this month",
    ]);
  });

  test("reports each rollout as it settles", async () => {
    const settled: number[] = [];
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: (candidate) => echoRunnable(candidate.instruction ?? ""),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "classify" },
      captureTraces: false,
      run: RUN,
      onRollout: () => settled.push(settled.length + 1),
    });

    expect(settled).toEqual([1, 2]);
  });

  test("passes the candidate text into the runnable", async () => {
    const seen: string[] = [];
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: (candidate) => {
        seen.push(candidate.instruction as string);
        return echoRunnable(candidate.instruction as string);
      },
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "version-two" },
      captureTraces: false,
      run: RUN,
    });

    expect(seen).toContain("version-two");
  });

  test("records feedback returned by the scorer", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => echoRunnable("x"),
      toInput: (datum) => ({ text: datum.text }),
      score: ({ datum }) => ({
        score: 0,
        feedback: `Expected ${datum.expected}`,
      }),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.feedback).toEqual(["Expected hardware", "Expected billing"]);
  });

  test("omits trajectories when traces are not requested", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => echoRunnable("x"),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.trajectories).toBeUndefined();
  });

  test("captures model calls in the trajectory when traces are requested", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: (candidate) =>
        ChatPromptTemplate.fromMessages([
          ["system", candidate.instruction as string],
          ["human", "{text}"],
        ])
          .pipe(new FakeListChatModel({ responses: ["hardware", "billing"] }))
          .pipe(new StringOutputParser()),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "Classify the ticket." },
      captureTraces: true,
      run: RUN,
    });

    const steps = result.trajectories?.[0]?.steps ?? [];

    expect(steps.length).toBeGreaterThan(0);
    expect(steps.some((step) => step.type === "llm")).toBe(true);
    expect(JSON.stringify(steps)).toContain("Classify the ticket.");
  });

  test("scores a failing run as zero instead of throwing", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(() => {
          throw new Error("chain blew up");
        }),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: true,
      run: RUN,
    });

    expect(result.scores).toEqual([0, 0]);
    expect(result.feedback?.[0]).toContain("chain blew up");
    expect(result.outputs).toEqual([null, null]);
  });

  test("builds one reflective record per batch item for each component", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => echoRunnable("x"),
      toInput: (datum) => ({ text: datum.text }),
      score: ({ datum }) => ({
        score: 0,
        feedback: `Expected ${datum.expected}`,
      }),
    });

    const evaluation = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { instruction: "x" },
      batch: TICKETS,
      evaluation,
      componentsToUpdate: ["instruction"],
    });

    expect(dataset.instruction).toHaveLength(2);
    expect(dataset.instruction?.[0]?.feedback).toBe("Expected hardware");
    expect(dataset.instruction?.[0]?.inputs).toEqual({
      text: "my printer is on fire",
    });
  });

  test("nests the captured trace steps under the record's evidence", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => tracedClassifier(),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    const evaluation = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { instruction: "x" },
      batch: TICKETS,
      evaluation,
      componentsToUpdate: ["instruction"],
    });

    expect(evaluation.trajectories?.[0]?.steps).not.toHaveLength(0);
    expect(dataset.instruction?.[0]?.evidence).toEqual({
      trace: evaluation.trajectories?.[0]?.steps,
    });
  });

  test("keeps only the named run in the evidence when componentRunNames names one", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => tracedClassifier(),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
      componentRunNames: { instruction: "classifier" },
    });

    const evaluation = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { instruction: "x" },
      batch: TICKETS,
      evaluation,
      componentsToUpdate: ["instruction"],
    });

    expect(dataset.instruction?.[0]?.evidence).toEqual({
      trace: [expect.objectContaining({ name: "classifier", type: "llm" })],
    });
  });

  test("omits the evidence when no trace step belongs to the component", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => tracedClassifier(),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
      componentRunNames: { instruction: "some-other-run" },
    });

    const evaluation = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: true,
      run: RUN,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: { instruction: "x" },
      batch: TICKETS,
      evaluation,
      componentsToUpdate: ["instruction"],
    });

    expect(dataset.instruction?.[0]).not.toHaveProperty("evidence");
  });

  test("respects the concurrency limit", async () => {
    let active = 0;
    let peak = 0;

    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(async () => {
          active += 1;
          peak = Math.max(peak, active);
          await new Promise((resolve) => setTimeout(resolve, 5));
          active -= 1;
          return "done";
        }),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
      concurrency: 2,
    });

    await adapter.evaluate({
      batch: Array.from({ length: 8 }, () => TICKETS[0] as Ticket),
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(peak).toBeLessThanOrEqual(2);
  });

  test("keeps results aligned with the batch when items finish out of order", async () => {
    // Every item is distinct and the first one finishes last, so a result
    // array assembled in completion order would be visibly wrong.
    const batch: Ticket[] = [
      { text: "slow", expected: "slow" },
      { text: "medium", expected: "medium" },
      { text: "fast", expected: "fast" },
    ];
    const delays: Record<string, number> = { slow: 30, medium: 15, fast: 1 };

    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(async (input: { text: string }) => {
          await new Promise((resolve) =>
            setTimeout(resolve, delays[input.text]),
          );
          return input.text;
        }),
      toInput: (datum) => ({ text: datum.text }),
      score: ({ datum, output }) => ({
        score: output === datum.expected ? 1 : 0,
      }),
      concurrency: 3,
    });

    const result = await adapter.evaluate({
      batch,
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.outputs).toEqual(["slow", "medium", "fast"]);
    expect(result.scores).toEqual([1, 1, 1]);
  });

  test("scores a failing scorer as zero instead of aborting the run", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => echoRunnable(""),
      toInput: (datum) => ({ text: datum.text }),
      score: ({ datum }) => {
        if (datum.expected === "hardware") {
          throw new Error("judge rate limited");
        }
        return { score: 1 };
      },
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.scores).toEqual([0, 1]);
    expect(result.feedback?.[0]).toContain("Scoring failed");
    expect(result.feedback?.[0]).toContain("judge rate limited");
  });

  test("marks a run failure transient when isTransient says so", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(() => {
          throw new Error("429 rate limit exceeded");
        }),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
      isTransient: (err) => (err as Error).message.includes("429"),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.transient).toEqual([true, true]);
    expect(result.scores).toEqual([0, 0]);
  });

  test("leaves a run failure non-transient by default", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(() => {
          throw new Error("429 rate limit exceeded");
        }),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.transient).toBeUndefined();
  });

  test("marks a step that never completed rather than leaving it look empty", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => buildRunnableWithOrphanedToolSpan(),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: [TICKETS[0] as Ticket],
      candidate: { instruction: "x" },
      captureTraces: true,
      run: RUN,
    });

    const orphan = result.trajectories?.[0]?.steps.find(
      (step) => step.name === "abandoned",
    );

    expect(orphan?.error).toMatch(/did not complete/i);
  });

  test("leaves a step that completed unmarked", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => buildRunnableWithOrphanedToolSpan(),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    const result = await adapter.evaluate({
      batch: [TICKETS[0] as Ticket],
      candidate: { instruction: "x" },
      captureTraces: true,
      run: RUN,
    });

    const completed = result.trajectories?.[0]?.steps.find(
      (step) => step.name === "finished",
    );

    expect(completed?.error).toBeUndefined();
  });

  test("propagates an abort instead of scoring it as a failed run", async () => {
    const controller = new AbortController();
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(() => {
          controller.abort();
          controller.signal.throwIfAborted();
          return "unreachable";
        }),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    await expect(
      adapter.evaluate({
        batch: TICKETS,
        candidate: { instruction: "x" },
        captureTraces: false,
        run: RUN,
        signal: controller.signal,
      }),
    ).rejects.toThrow(/abort/i);
  });

  test("surfaces objective scores from the scorer", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () => echoRunnable("x"),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({
        score: 0.5,
        objectiveScores: { accuracy: 0.5, latencyMs: 120 },
      }),
    });

    const result = await adapter.evaluate({
      batch: TICKETS,
      candidate: { instruction: "x" },
      captureTraces: false,
      run: RUN,
    });

    expect(result.objectiveScores?.[0]).toEqual({
      accuracy: 0.5,
      latencyMs: 120,
    });
  });

  test("attaches the run context to every invocation as tracing metadata", async () => {
    const seen: (Record<string, unknown> | undefined)[] = [];
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(
          (input: { text: string }, config: RunnableConfig) => {
            seen.push(config.metadata);
            return input.text;
          },
        ),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    await adapter.evaluate({
      batch: TICKETS.slice(0, 1),
      candidate: { instruction: "x" },
      captureTraces: false,
      run: {
        iteration: 7,
        phase: "validation",
        split: "val",
        candidateId: 3,
      },
    });

    expect(seen[0]).toMatchObject({
      textopt_iteration: 7,
      textopt_phase: "validation",
      textopt_split: "val",
      textopt_candidate_id: 3,
    });
  });

  test("marks a proposal being screened as having no candidate id", async () => {
    const seen: (Record<string, unknown> | undefined)[] = [];
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: () =>
        RunnableLambda.from(
          (input: { text: string }, config: RunnableConfig) => {
            seen.push(config.metadata);
            return input.text;
          },
        ),
      toInput: (datum) => ({ text: datum.text }),
      score: () => ({ score: 1 }),
    });

    await adapter.evaluate({
      batch: TICKETS.slice(0, 1),
      candidate: { instruction: "x" },
      captureTraces: false,
      run: { ...RUN, candidateId: null },
    });

    expect(seen[0]?.textopt_candidate_id).toBeNull();
  });
});

describe("createLangChainAdapter driven by GepaOptimizer", () => {
  test("optimizes the components the seed candidate names", async () => {
    const adapter = createLangChainAdapter<Ticket, string>({
      buildRunnable: (candidate) => echoRunnable(candidate.instruction),
      toInput: (datum) => ({ text: datum.text }),
      score: ({ datum, output }) => ({
        score: Number(output?.includes(datum.expected) === true),
        feedback: `Missing required terms: ${datum.expected}`,
      }),
    });

    const gepa = new GepaOptimizer({ minibatchSize: 2, seed: 7 });
    const result = await gepa.optimize({
      seedCandidate: { instruction: "classify" },
      trainingSet: TICKETS,
      adapter,
      reflect: createKeywordReflector(),
      maxMetricCalls: 60,
    });

    expect(result.bestScore).toBe(1);
    expect(result.bestCandidate.instruction).toContain("hardware");
    expect(result.bestCandidate.instruction).toContain("billing");
    // The adapter is K-agnostic; this fails to compile if that widens the
    // component keys the seed candidate inferred back to `string`.
    // @ts-expect-error `instructions` is not a component of the seed candidate
    expect(result.bestCandidate.instructions).toBeUndefined();
  });
});
