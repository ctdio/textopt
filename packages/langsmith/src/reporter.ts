import { mapWithConcurrency } from "textopt";
import type { Candidate } from "textopt";
import type { GepaEvent, GepaReporter } from "textopt/gepa";

export interface LangSmithDataset {
  id: string;
}

export interface LangSmithProject {
  id: string;
  /** Optional because LangSmith's own `TracerSession` declares it so. */
  name?: string;
}

export interface LangSmithExample {
  id?: string;
  dataset_id?: string;
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface LangSmithRun {
  id?: string;
  name: string;
  run_type: string;
  inputs: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  project_name?: string;
  reference_example_id?: string;
  start_time?: number;
  end_time?: number;
}

/**
 * The part of LangSmith's `Client` this reporter uses. Structural rather than
 * imported so the package carries no runtime dependency on the SDK, and so a
 * test can hand it a recording client instead of a network.
 */
export interface LangSmithClientLike {
  hasDataset(args: { datasetName: string }): Promise<boolean>;
  readDataset(args: { datasetName: string }): Promise<LangSmithDataset>;
  createDataset(
    name: string,
    options?: { description?: string },
  ): Promise<LangSmithDataset>;
  createExamples(uploads: LangSmithExample[]): Promise<LangSmithExample[]>;
  createProject(args: {
    projectName: string;
    referenceDatasetId?: string;
    metadata?: Record<string, unknown>;
  }): Promise<LangSmithProject>;
  createRun(run: LangSmithRun): Promise<unknown>;
  createFeedback(
    runId: string | null,
    key: string,
    options: { score?: number; sessionId?: string },
  ): Promise<unknown>;
}

export interface LangSmithReporterOptions<Datum> {
  client: LangSmithClientLike;
  /** Dataset holding the validation split. Created once, reused across runs. */
  dataset: string;
  /** Experiments are named `<experimentPrefix>/cand-<candidateId>`. */
  experimentPrefix: string;
  /** The validation set the run was given, in the same order. */
  validationSet: readonly Datum[];
  /**
   * The held-out set the run was given. Uploaded as its own dataset, and only
   * once the winner has been chosen — see the note on the reporter itself.
   */
  testSet?: readonly Datum[];
  /** Defaults to `<dataset>-held-out`. */
  testDataset?: string;
  /**
   * Names a dataset row. Defaults to its position, which is enough for a run
   * whose validation set is stable; give a real id if the set is ever
   * reordered or re-generated, since a row's identity is what a later run
   * matches against when it writes to the same dataset.
   */
  instanceId?: (args: { datum: Datum; index: number }) => string;
  toInput?: (datum: Datum) => Record<string, unknown>;
  toExpected?: (datum: Datum) => Record<string, unknown>;
  /** Rows uploaded at once within one experiment. Default 8. */
  concurrency?: number;
}

const DEFAULT_CONCURRENCY = 8;

/**
 * A UUIDv5 namespace for this library, so a dataset row's id is a pure
 * function of the dataset it belongs to and the instance it names. Fixed
 * forever: changing it renames every row textopt has ever written.
 */
const TEXTOPT_NAMESPACE = "6f9619ff-8b86-d011-b42d-00c04fc964ff";

/**
 * Reports a GEPA run to LangSmith as one experiment per accepted candidate
 * over one fixed dataset — which is what makes the candidates comparable.
 * Selecting several in LangSmith's comparison view renders the candidate x
 * instance score matrix the Pareto frontier is chosen on: which instances a
 * candidate won, and which it paid for.
 *
 * Minibatch rollouts are deliberately absent. They are small random subsets of
 * the *training* set that differ every iteration, so they cannot be compared
 * across candidates and would bury the experiments that can. They remain
 * traceable through the adapter, which tags every rollout with its iteration,
 * phase, split and candidate id.
 *
 * The held-out sweep gets exactly one experiment, for the winner, on its own
 * dataset. There is no option to log one per candidate: GEPA evaluates the
 * held-out set only after selection is over precisely so that nothing can be
 * chosen against it, and a per-candidate view hands that back the moment
 * someone reads it.
 */
export function createLangSmithReporter<Datum>(
  options: LangSmithReporterOptions<Datum>,
): GepaReporter {
  const {
    client,
    dataset,
    experimentPrefix,
    validationSet,
    testSet,
    testDataset = `${dataset}-held-out`,
    instanceId = ({ index }: { datum: Datum; index: number }) => String(index),
    toInput = (datum: Datum) => datum as Record<string, unknown>,
    toExpected,
    concurrency = DEFAULT_CONCURRENCY,
  } = options;

  // Uploads are chained rather than awaited: `onEvent` runs on the search's
  // hot path, and a run should not pay LangSmith's latency once per candidate.
  // Serialized rather than parallel because the dataset has to exist before
  // the first experiment can reference it.
  let pending: Promise<void> = Promise.resolve();
  let validationDatasetId: Promise<string> | undefined;
  let heldOutDatasetId: Promise<string> | undefined;

  function enqueue(work: () => Promise<void>): void {
    pending = pending.then(work).catch((err: unknown) => {
      // The run buys the rollouts; a logging endpoint being down does not get
      // to spend them again.
      console.warn("[textopt-langsmith] upload failed", { err });
    });
  }

  async function ensureDataset(args: {
    name: string;
    rows: readonly Datum[];
  }): Promise<string> {
    if (await client.hasDataset({ datasetName: args.name })) {
      // Its rows already carry the ids `exampleIdFor` derives, so there is
      // nothing to reconcile — and re-uploading would duplicate them.
      return (await client.readDataset({ datasetName: args.name })).id;
    }

    const created = await client.createDataset(args.name, {
      description: `textopt: ${String(args.rows.length)} instances`,
    });

    await client.createExamples(
      await Promise.all(
        args.rows.map(async (datum, index) => ({
          id: await exampleIdFor({
            dataset: args.name,
            instance: instanceId({ datum, index }),
          }),
          dataset_id: created.id,
          inputs: toInput(datum),
          ...(toExpected === undefined ? {} : { outputs: toExpected(datum) }),
          metadata: {
            textopt_instance_id: instanceId({ datum, index }),
          },
        })),
      ),
    );

    return created.id;
  }

  async function writeExperiment(args: {
    projectName: string;
    datasetId: string;
    datasetName: string;
    rows: readonly Datum[];
    scores: readonly (number | undefined)[];
    outputs: readonly unknown[] | undefined;
    metadata: Record<string, unknown>;
  }): Promise<void> {
    const project = await client.createProject({
      projectName: args.projectName,
      referenceDatasetId: args.datasetId,
      metadata: args.metadata,
    });

    const scored = args.rows.flatMap((datum, index) => {
      const score = args.scores[index];
      // An instance nothing measured is unknown, not zero. A row written for
      // it reads as a regression that never happened.
      return score === undefined ? [] : [{ datum, index, score }];
    });

    await mapWithConcurrency({
      items: scored,
      limit: concurrency,
      task: async ({ datum, index, score }) => {
        const runId = crypto.randomUUID();
        const now = Date.now();

        await client.createRun({
          id: runId,
          name: args.projectName,
          run_type: "chain",
          inputs: toInput(datum),
          outputs: { output: args.outputs?.[index] ?? null },
          project_name: args.projectName,
          reference_example_id: await exampleIdFor({
            dataset: args.datasetName,
            instance: instanceId({ datum, index }),
          }),
          start_time: now,
          end_time: now,
        });

        await client.createFeedback(runId, "score", {
          score,
          sessionId: project.id,
        });
      },
    });
  }

  return {
    onEvent: (event: GepaEvent) => {
      if (event.type === "candidateAccepted") {
        enqueue(async () => {
          validationDatasetId ??= ensureDataset({
            name: dataset,
            rows: validationSet,
          });

          await writeExperiment({
            projectName: `${experimentPrefix}/cand-${String(event.candidateId)}`,
            datasetId: await validationDatasetId,
            datasetName: dataset,
            rows: validationSet,
            scores: event.instanceScores,
            outputs: event.outputs,
            metadata: experimentMetadata({
              candidate: event.candidate,
              candidateId: event.candidateId,
              iteration: event.iteration,
              parentIds: event.parentIds,
              source: event.source,
              aggregateScore: event.aggregateScore,
            }),
          });
        });
      }

      if (
        event.type === "finish" &&
        event.testInstanceScores !== undefined &&
        testSet !== undefined
      ) {
        const heldOut = event.testInstanceScores;
        const outputs = event.testOutputs;
        const { bestCandidateId, testScore } = event;

        enqueue(async () => {
          heldOutDatasetId ??= ensureDataset({
            name: testDataset,
            rows: testSet,
          });

          await writeExperiment({
            projectName: `${experimentPrefix}/held-out-cand-${String(bestCandidateId)}`,
            datasetId: await heldOutDatasetId,
            datasetName: testDataset,
            rows: testSet,
            scores: heldOut,
            outputs,
            metadata: {
              textopt_candidate_id: bestCandidateId,
              textopt_split: "test",
              ...(testScore === undefined ? {} : { textopt_score: testScore }),
            },
          });
        });
      }
    },

    flush: async () => {
      await pending;
    },
  };
}

function experimentMetadata(args: {
  candidate: Candidate;
  candidateId: number;
  iteration: number;
  parentIds: readonly number[];
  source: string;
  aggregateScore: number;
}): Record<string, unknown> {
  return {
    // The text first, because it is what someone opens an experiment to read.
    ...args.candidate,
    textopt_candidate_id: args.candidateId,
    textopt_iteration: args.iteration,
    textopt_parent_ids: [...args.parentIds],
    textopt_source: args.source,
    textopt_score: args.aggregateScore,
  };
}

/**
 * A dataset row's id, derived from the dataset and the instance it names
 * rather than assigned by the server.
 *
 * Server-assigned ids would give a resumed or repeated run a fresh set of
 * rows, and every experiment logged before the restart would have nothing left
 * to compare against.
 */
async function exampleIdFor(args: {
  dataset: string;
  instance: string;
}): Promise<string> {
  return uuidV5({
    namespace: TEXTOPT_NAMESPACE,
    name: `${args.dataset}:${args.instance}`,
  });
}

/** RFC 4122 name-based UUID, SHA-1 flavour. LangSmith ids must be UUIDs. */
async function uuidV5(args: {
  namespace: string;
  name: string;
}): Promise<string> {
  const namespaceBytes = uuidToBytes(args.namespace);
  const nameBytes = new TextEncoder().encode(args.name);
  const payload = new Uint8Array(namespaceBytes.length + nameBytes.length);
  payload.set(namespaceBytes);
  payload.set(nameBytes, namespaceBytes.length);

  const digest = new Uint8Array(await crypto.subtle.digest("SHA-1", payload));
  const bytes = digest.slice(0, 16);
  bytes[6] = ((bytes[6] as number) & 0x0f) | 0x50;
  bytes[8] = ((bytes[8] as number) & 0x3f) | 0x80;

  const hex = [...bytes]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-");
}

function uuidToBytes(uuid: string): Uint8Array {
  const hex = uuid.replaceAll("-", "");
  const bytes = new Uint8Array(16);

  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}
