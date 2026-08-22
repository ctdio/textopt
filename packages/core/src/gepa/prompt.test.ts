import { describe, expect, test } from "vitest";
import { createPromptAdapter } from "./prompt.js";
import type { PipelineTrace } from "./pipeline.js";
import type { Adapter } from "../types.js";

interface Question {
  id: string;
  text: string;
}

const BATCH: Question[] = [
  { id: "a", text: "how long is the hold" },
  { id: "b", text: "where is my refund" },
];

describe("createPromptAdapter", () => {
  test("runs the candidate's only component as the instruction", async () => {
    const adapter = buildAdapter();

    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: { system: "answer briefly" },
      captureTraces: false,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    expect(evaluation.outputs[0]).toBe("answer briefly | how long is the hold");
  });

  test("scores each rollout's output", async () => {
    const adapter = buildAdapter();

    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: { system: "answer briefly" },
      captureTraces: false,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    expect(evaluation.scores).toEqual([1, 0]);
  });

  test("gives the component records of what the prompt received and produced", async () => {
    const adapter = buildAdapter();
    const candidate = { system: "answer briefly" };
    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate,
      captureTraces: true,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    const dataset = await adapter.makeReflectiveDataset({
      candidate,
      batch: BATCH,
      evaluation,
      componentsToUpdate: ["system"],
    });

    expect(dataset.system?.[0]?.inputs).toBe("how long is the hold");
    expect(dataset.system?.[0]?.generatedOutputs).toBe(
      "answer briefly | how long is the hold",
    );
  });

  test("defaults the prompt's input to the datum itself", async () => {
    const adapter = createPromptAdapter<Question, string>({
      run: ({ input }) => JSON.stringify(input),
      score: () => ({ score: 1 }),
    });

    const evaluation = await adapter.evaluate({
      batch: [BATCH[0] as Question],
      candidate: { system: "answer briefly" },
      captureTraces: false,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    expect(evaluation.outputs[0]).toBe(
      '{"id":"a","text":"how long is the hold"}',
    );
  });

  test("refuses a candidate whose second component nothing would read", async () => {
    const adapter = buildAdapter();

    await expect(
      adapter.evaluate({
        batch: BATCH,
        candidate: { router: "route it", replyStyle: "be brief" },
        captureTraces: false,
        run: {
          iteration: 0,
          phase: "validation",
          split: "val",
          candidateId: 1,
        },
      }),
    ).rejects.toThrow(/router, replyStyle.*createPipelineAdapter/s);
  });

  test("refuses a candidate with no components at all", async () => {
    const adapter = buildAdapter();

    await expect(
      adapter.evaluate({
        batch: BATCH,
        candidate: {},
        captureTraces: false,
        run: {
          iteration: 0,
          phase: "validation",
          split: "val",
          candidateId: 1,
        },
      }),
    ).rejects.toThrow(/exactly one component/);
  });
});

/**
 * The adapter is a `GepaAdapter`, and every other optimizer takes the base
 * `Adapter` it extends. This assignment is what keeps the documented claim —
 * write the adapter once, choose the optimizer later — from being prose. It is
 * never called; the compiler is the assertion.
 */
export function promptAdapterFitsEveryOptimizer(): Adapter<
  Question,
  PipelineTrace,
  string,
  string
> {
  return buildAdapter();
}

function buildAdapter() {
  return createPromptAdapter<Question, string>({
    input: (datum) => datum.text,
    run: ({ instruction, input }) => `${instruction} | ${String(input)}`,
    score: ({ datum, output }) => ({
      score: output.includes("hold") ? 1 : 0,
      feedback: `answered ${datum.id}`,
    }),
  });
}
