import { GepaOptimizer } from "textopt/gepa";
import { subsampledEvaluationPolicy } from "textopt/gepa";
import type { KeywordExample } from "textopt/testing";
import {
  KEYWORD_EXAMPLES,
  createKeywordAdapter,
  createKeywordReflector,
} from "textopt/testing";
import { afterEach, describe, expect, test, vi } from "vitest";
import type {
  LangSmithClientLike,
  LangSmithExample,
  LangSmithRun,
} from "./reporter.js";
import { createLangSmithReporter } from "./reporter.js";

interface RecordedProject {
  id: string;
  name: string;
  referenceDatasetId?: string;
  metadata?: Record<string, unknown>;
}

interface RecordingClient extends LangSmithClientLike {
  datasets: { id: string; name: string }[];
  examples: LangSmithExample[];
  projects: RecordedProject[];
  runs: LangSmithRun[];
  feedback: { runId: string; key: string; score: number }[];
}

function createRecordingClient(): RecordingClient {
  const datasets: { id: string; name: string }[] = [];
  const examples: LangSmithExample[] = [];
  const projects: RecordedProject[] = [];
  const runs: LangSmithRun[] = [];
  const feedback: { runId: string; key: string; score: number }[] = [];

  return {
    datasets,
    examples,
    projects,
    runs,
    feedback,
    hasDataset: async ({ datasetName }) =>
      datasets.some((dataset) => dataset.name === datasetName),
    readDataset: async ({ datasetName }) => {
      const found = datasets.find((dataset) => dataset.name === datasetName);
      if (found === undefined) {
        throw new Error(`no dataset named ${String(datasetName)}`);
      }
      return found;
    },
    createDataset: async (name) => {
      const dataset = { id: `dataset-${datasets.length}`, name };
      datasets.push(dataset);
      return dataset;
    },
    createExamples: async (uploads) => {
      const created = uploads.map((upload, index) => ({
        ...upload,
        id: upload.id ?? `example-${examples.length + index}`,
      }));
      examples.push(...created);
      return created;
    },
    createProject: async (args) => {
      const project = {
        ...args,
        id: `project-${projects.length}`,
        name: args.projectName,
      };
      projects.push(project);
      return project;
    },
    createRun: async (run) => {
      runs.push(run);
    },
    createFeedback: async (runId, key, options) => {
      feedback.push({ runId: runId ?? "", key, score: options.score ?? 0 });
      return { id: `feedback-${feedback.length}` };
    },
  };
}

const SEED = { instruction: "Answer the customer's question." };
const CONFIG = { minibatchSize: 2, seed: 1, maxIterations: 3 };
const HELD_OUT: KeywordExample[] = [
  { question: "held out, satisfied", required: ["answer"] },
  { question: "held out, unsatisfiable", required: ["zzz-never-proposed"] },
];

function createTask() {
  return {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: createKeywordAdapter(),
    reflect: createKeywordReflector(),
    maxMetricCalls: 200,
  };
}

function experimentsOf(client: RecordingClient): RecordedProject[] {
  return client.projects.filter(
    (project) => !project.name.includes("held-out"),
  );
}

describe("createLangSmithReporter", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("uploads the validation set once, as one dataset", async () => {
    const client = createRecordingClient();

    await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
        }),
      ],
    });

    expect(client.datasets).toHaveLength(1);
    expect(client.datasets[0]?.name).toBe("keyword-val");
    expect(client.examples).toHaveLength(KEYWORD_EXAMPLES.length);
  });

  test("opens an experiment for every accepted candidate, the seed included", async () => {
    // The seed is the baseline every later candidate is read against. An
    // experiment list that starts at the first improvement has nothing to
    // compare the improvement to.
    const client = createRecordingClient();

    const result = await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
        }),
      ],
    });

    expect(experimentsOf(client).map((project) => project.name)).toEqual(
      result.candidates.map((record) => `run-1/cand-${record.id}`),
    );
  });

  test("points every experiment at the same dataset", async () => {
    // Comparing two experiments in LangSmith means comparing them over shared
    // examples: a dataset per candidate would compare nothing.
    const client = createRecordingClient();

    await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
        }),
      ],
    });

    const referenced = new Set(
      experimentsOf(client).map((project) => project.referenceDatasetId),
    );

    expect([...referenced]).toEqual([client.datasets[0]?.id]);
  });

  test("scores one run per measured instance against its dataset example", async () => {
    const client = createRecordingClient();

    const result = await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
        }),
      ],
    });

    const measured = result.candidates.flatMap((record) =>
      record.instanceScores.filter((score) => score !== undefined),
    );
    const exampleIds = new Set(client.examples.map((example) => example.id));

    expect(client.runs).toHaveLength(measured.length);
    expect(client.feedback).toHaveLength(measured.length);
    for (const run of client.runs) {
      expect(exampleIds.has(run.reference_example_id ?? "")).toBe(true);
    }
  });

  test("leaves out the instances the evaluation policy never measured", async () => {
    // Writing a row for an unscored instance shows as a regression that never
    // happened.
    const client = createRecordingClient();

    const result = await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      valEvaluationPolicy: subsampledEvaluationPolicy<KeywordExample>({
        size: 2,
      }),
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
        }),
      ],
    });

    const unmeasured = result.candidates.flatMap((record) =>
      record.instanceScores.filter((score) => score === undefined),
    );

    expect(unmeasured.length).toBeGreaterThan(0);
    expect(client.runs.length).toBeLessThan(
      result.candidates.length * KEYWORD_EXAMPLES.length,
    );
  });

  test("carries the text that scored into the experiment", async () => {
    const client = createRecordingClient();

    const result = await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
        }),
      ],
    });

    const winner = client.projects.find(
      (project) => project.name === `run-1/cand-${result.bestCandidateId}`,
    );

    expect(winner?.metadata?.["instruction"]).toBe(
      result.bestCandidate.instruction,
    );
    expect(winner?.metadata?.["textopt_candidate_id"]).toBe(
      result.bestCandidateId,
    );
  });

  test("reports the held-out sweep as its own experiment on its own dataset", async () => {
    // Held-out instances are not validation instances, and an experiment that
    // shares their comparison view invites selecting on them by eye.
    const client = createRecordingClient();

    const result = await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      testSet: HELD_OUT,
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
          testSet: HELD_OUT,
        }),
      ],
    });

    const heldOut = client.projects.filter((project) =>
      project.name.includes("held-out"),
    );

    expect(heldOut).toHaveLength(1);
    expect(heldOut[0]?.name).toBe(
      `run-1/held-out-cand-${result.bestCandidateId}`,
    );
    expect(heldOut[0]?.referenceDatasetId).not.toBe(client.datasets[0]?.id);
    expect(client.datasets.map((dataset) => dataset.name)).toEqual([
      "keyword-val",
      "keyword-val-held-out",
    ]);
  });

  test("opens no held-out experiment when the run had no testSet", async () => {
    const client = createRecordingClient();

    await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      reporters: [
        createLangSmithReporter({
          client,
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
          testSet: HELD_OUT,
        }),
      ],
    });

    expect(client.datasets).toHaveLength(1);
    expect(
      client.projects.filter((project) => project.name.includes("held-out")),
    ).toHaveLength(0);
  });

  test("keeps the search running when LangSmith is unreachable", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const client = createRecordingClient();

    const result = await new GepaOptimizer(CONFIG).optimize({
      ...createTask(),
      reporters: [
        createLangSmithReporter({
          client: {
            ...client,
            createProject: async () => {
              throw new Error("langsmith is down");
            },
          },
          dataset: "keyword-val",
          experimentPrefix: "run-1",
          validationSet: KEYWORD_EXAMPLES,
        }),
      ],
    });

    expect(result.stopReason).toBe("maxIterations");
    expect(warn).toHaveBeenCalled();
  });

  test("keys dataset examples the same way on every run", async () => {
    // Server-assigned ids would give a resumed run a fresh set of rows, and
    // the comparison against everything logged before the restart breaks.
    const runOnce = async () => {
      const client = createRecordingClient();
      await new GepaOptimizer(CONFIG).optimize({
        ...createTask(),
        reporters: [
          createLangSmithReporter({
            client,
            dataset: "keyword-val",
            experimentPrefix: "run-1",
            validationSet: KEYWORD_EXAMPLES,
            instanceId: ({ datum }) => datum.question,
          }),
        ],
      });
      return client.examples.map((example) => example.id);
    };

    expect(await runOnce()).toEqual(await runOnce());
  });
});
