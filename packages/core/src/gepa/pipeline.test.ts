import { describe, expect, test } from "vitest";
import { createPipelineAdapter } from "./pipeline.js";

interface Question {
  id: string;
  text: string;
}

const BATCH: Question[] = [
  { id: "a", text: "how long is the hold" },
  { id: "b", text: "where is my refund" },
];

const CANDIDATE = { retrieve: "find docs", answer: "reply briefly" };

describe("createPipelineAdapter", () => {
  test("feeds each module the previous module's output", async () => {
    const adapter = buildAdapter();

    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: CANDIDATE,
      captureTraces: true,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    expect(evaluation.outputs[0]).toBe(
      "reply briefly | find docs | how long is the hold",
    );
  });

  test("scores the pipeline's final output", async () => {
    const adapter = buildAdapter();

    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: CANDIDATE,
      captureTraces: false,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    expect(evaluation.scores).toEqual([1, 0]);
  });

  test("gives a component records of what its own module saw and produced", async () => {
    const adapter = buildAdapter();
    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: CANDIDATE,
      captureTraces: true,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    const dataset = await adapter.makeReflectiveDataset({
      candidate: CANDIDATE,
      batch: BATCH,
      evaluation,
      componentsToUpdate: ["retrieve"],
    });

    expect(dataset.retrieve?.[0]?.inputs).toBe("how long is the hold");
    expect(dataset.retrieve?.[0]?.generatedOutputs).toBe(
      "find docs | how long is the hold",
    );
  });

  test("builds records only for the components asked for", async () => {
    const adapter = buildAdapter();
    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: CANDIDATE,
      captureTraces: true,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    const dataset = await adapter.makeReflectiveDataset({
      candidate: CANDIDATE,
      batch: BATCH,
      evaluation,
      componentsToUpdate: ["answer"],
    });

    expect(Object.keys(dataset)).toEqual(["answer"]);
  });

  test("attaches the instance's end-to-end feedback to every module", async () => {
    const adapter = buildAdapter();
    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: CANDIDATE,
      captureTraces: true,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    const dataset = await adapter.makeReflectiveDataset({
      candidate: CANDIDATE,
      batch: BATCH,
      evaluation,
      componentsToUpdate: ["retrieve", "answer"],
    });

    expect(dataset.retrieve?.[1]?.feedback).toBe("wrong answer for b");
    expect(dataset.answer?.[1]?.feedback).toBe("wrong answer for b");
  });

  test("names the module a record came from, so a merged trace stays readable", async () => {
    const adapter = buildAdapter();
    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: CANDIDATE,
      captureTraces: true,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    const dataset = await adapter.makeReflectiveDataset({
      candidate: CANDIDATE,
      batch: BATCH,
      evaluation,
      componentsToUpdate: ["answer"],
    });

    expect(dataset.answer?.[0]?.evidence).toEqual({ component: "answer" });
  });

  test("reports no trajectories when traces were not asked for", async () => {
    const adapter = buildAdapter();

    const evaluation = await adapter.evaluate({
      batch: BATCH,
      candidate: CANDIDATE,
      captureTraces: false,
      run: { iteration: 0, phase: "validation", split: "val", candidateId: 1 },
    });

    expect(evaluation.trajectories).toBeUndefined();
  });
});

function buildAdapter() {
  return createPipelineAdapter<Question, string, "retrieve" | "answer">({
    modules: [
      {
        component: "retrieve",
        run: async ({ instruction, input }) => `${instruction} | ${input}`,
      },
      {
        component: "answer",
        run: async ({ instruction, input }) => `${instruction} | ${input}`,
      },
    ],
    input: (datum) => datum.text,
    score: ({ datum, output }) => ({
      score: output.includes("hold") ? 1 : 0,
      feedback: output.includes("hold")
        ? `answered ${datum.id}`
        : `wrong answer for ${datum.id}`,
    }),
  });
}
