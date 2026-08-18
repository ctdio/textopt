import { mapWithConcurrency } from "@ctdio/textopt";
import type { Candidate, EvaluationBatch, ScoreResult } from "@ctdio/textopt";
import type {
  GepaAdapter,
  ReflectiveDataset,
  ReflectiveRecord,
} from "@ctdio/textopt/gepa";
import type { CallbackHandlerMethods } from "@langchain/core/callbacks/base";
import type { Runnable } from "@langchain/core/runnables";

export type LangChainStepType = "llm" | "chain" | "tool" | "retriever";

export interface LangChainTraceStep {
  type: LangChainStepType;
  name: string;
  runId: string;
  parentRunId?: string;
  inputs?: unknown;
  outputs?: unknown;
  error?: string;
}

export interface LangChainTrace {
  steps: LangChainTraceStep[];
  durationMs: number;
  error?: string;
}

export type LangChainScore = ScoreResult;

/** What the reflection model sees beyond the score and feedback of a run. */
export interface LangChainEvidence {
  trace: LangChainTraceStep[];
}

export interface LangChainAdapterOptions<Datum, Out> {
  /** Rebuild the chain with the candidate's text injected into its prompts. */
  buildRunnable: (candidate: Candidate) => Runnable<never, Out>;
  /**
   * Per-instance score plus the textual feedback the reflection model reads.
   * Feedback is what separates GEPA from blind search — say what went wrong,
   * not just how wrong it was.
   */
  score: (args: {
    datum: Datum;
    output: Out | null;
    trace: LangChainTrace;
  }) => LangChainScore | Promise<LangChainScore>;
  /** Map a dataset row to the chain's input. Defaults to the row itself. */
  toInput?: (datum: Datum) => unknown;
  concurrency?: number;
  /** Include LangChain's per-runnable chain spans in the trace. Noisy. */
  includeChainSteps?: boolean;
  /** Component name -> LangChain run name, used to highlight per-component IO. */
  componentRunNames?: Record<string, string>;
  /**
   * Classify a thrown error as infrastructure (rate limit, 5xx, network) so
   * its zero is not cached against the candidate. Defaults to treating every
   * failure as the candidate's, which is the safe assumption.
   */
  isTransient?: (err: unknown) => boolean;
  buildRecord?: (args: {
    datum: Datum;
    output: Out | null;
    trace: LangChainTrace;
    score: number;
    feedback: string;
    component: string;
  }) => ReflectiveRecord;
}

const DEFAULT_CONCURRENCY = 8;

/**
 * Runs a LangChain runnable (chain, agent, or LangGraph graph) as the system
 * under optimization. The candidate is injected by rebuilding the runnable, so
 * anything that reads candidate text — prompts, tool descriptions, routing
 * instructions — is optimizable.
 */
export function createLangChainAdapter<Datum, Out>(
  options: LangChainAdapterOptions<Datum, Out>,
): GepaAdapter<Datum, LangChainTrace, Out | null> {
  const {
    buildRunnable,
    score,
    toInput = (datum: Datum) => datum,
    concurrency = DEFAULT_CONCURRENCY,
    includeChainSteps = false,
    componentRunNames,
    isTransient = () => false,
    buildRecord,
  } = options;

  return {
    evaluate: async ({ batch, candidate, captureTraces, run, signal }) => {
      const runnable = buildRunnable(candidate);
      // Inert without a tracer configured, and the difference between a
      // LangSmith project full of anonymous rollouts and one you can filter
      // down to the iteration whose score moved.
      const metadata = {
        textopt_iteration: run.iteration,
        textopt_phase: run.phase,
        textopt_split: run.split,
        textopt_candidate_id: run.candidateId,
      };

      const results = await mapWithConcurrency({
        items: batch,
        limit: concurrency,
        task: async (datum) => {
          const collector = createTraceCollector({ includeChainSteps });
          const startedAt = Date.now();
          let output: Out | null = null;
          let failure: string | undefined;
          let transient = false;

          try {
            output = await runnable.invoke(toInput(datum) as never, {
              callbacks: captureTraces ? [collector.handler] : undefined,
              metadata,
              signal,
            });
          } catch (err) {
            // A cancelled run is not a failed rollout: scoring it zero would
            // poison the candidate's record and the evaluation cache.
            signal?.throwIfAborted();
            failure = err instanceof Error ? err.message : String(err);
            transient = isTransient(err);
          }

          const trace: LangChainTrace = {
            steps: collector.finish(),
            durationMs: Date.now() - startedAt,
            ...(failure === undefined ? {} : { error: failure }),
          };

          if (failure !== undefined) {
            return {
              output: null,
              trace,
              scored: {
                score: 0,
                feedback: `Run failed: ${failure}`,
                transient,
              },
            };
          }

          try {
            return {
              output,
              trace,
              scored: await score({ datum, output, trace }),
            };
          } catch (err) {
            signal?.throwIfAborted();
            const message = err instanceof Error ? err.message : String(err);
            return {
              output,
              trace,
              scored: {
                score: 0,
                feedback: `Scoring failed: ${message}`,
                transient: isTransient(err),
              },
            };
          }
        },
        signal,
      });

      const evaluation: EvaluationBatch<LangChainTrace, Out | null> = {
        outputs: results.map((result) => result.output),
        scores: results.map((result) => result.scored.score),
        feedback: results.map((result) => result.scored.feedback ?? ""),
      };

      if (captureTraces) {
        evaluation.trajectories = results.map((result) => result.trace);
      }
      if (
        results.some((result) => result.scored.objectiveScores !== undefined)
      ) {
        evaluation.objectiveScores = results.map(
          (result) => result.scored.objectiveScores ?? {},
        );
      }
      if (results.some((result) => result.scored.transient === true)) {
        evaluation.transient = results.map(
          (result) => result.scored.transient === true,
        );
      }

      return evaluation;
    },

    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) => {
      const dataset: ReflectiveDataset = {};

      for (const component of componentsToUpdate) {
        dataset[component] = batch.map((datum, index) => {
          const trace = evaluation.trajectories?.[index] ?? {
            steps: [],
            durationMs: 0,
          };
          const output = evaluation.outputs[index] ?? null;
          const scoreValue = evaluation.scores[index] ?? 0;
          const feedback = evaluation.feedback?.[index] ?? "";

          if (buildRecord !== undefined) {
            return buildRecord({
              datum,
              output,
              trace,
              score: scoreValue,
              feedback,
              component,
            });
          }

          const componentSteps = selectComponentSteps({
            trace,
            runName: componentRunNames?.[component],
          });

          const record: ReflectiveRecord<LangChainEvidence> = {
            inputs: toInput(datum),
            generatedOutputs: output,
            feedback,
            score: scoreValue,
            ...(componentSteps.length > 0
              ? { evidence: { trace: componentSteps } }
              : {}),
          };

          return record;
        });
      }

      return dataset;
    },
  };
}

function selectComponentSteps(args: {
  trace: LangChainTrace;
  runName: string | undefined;
}): LangChainTraceStep[] {
  const { trace, runName } = args;

  if (runName === undefined) {
    return trace.steps;
  }
  return trace.steps.filter((step) => step.name === runName);
}

/**
 * LangChain accepts plain handler objects, so trace capture needs no subclass
 * and no LangSmith account — the same events LangSmith records land here.
 */
function createTraceCollector(args: { includeChainSteps: boolean }): {
  handler: CallbackHandlerMethods;
  finish: () => LangChainTraceStep[];
} {
  const { includeChainSteps } = args;
  const steps: LangChainTraceStep[] = [];
  const openSteps = new Map<string, LangChainTraceStep>();

  function open(step: LangChainTraceStep): void {
    openSteps.set(step.runId, step);
    steps.push(step);
  }

  /**
   * Any span still open when the run ends never got its end callback — the
   * chain was torn down first. Left alone it is indistinguishable in the trace
   * from a step that legitimately produced nothing, which quietly misleads the
   * reflection model.
   */
  function finish(): LangChainTraceStep[] {
    for (const step of openSteps.values()) {
      step.error = "Step did not complete before the run ended";
    }
    openSteps.clear();
    return steps;
  }

  function close(args: {
    runId: string;
    outputs: unknown;
    error?: string;
  }): void {
    const step = openSteps.get(args.runId);
    if (step === undefined) {
      return;
    }
    step.outputs = args.outputs;
    if (args.error !== undefined) {
      step.error = args.error;
    }
    openSteps.delete(args.runId);
  }

  const handler: CallbackHandlerMethods = {
    handleLLMStart: (
      llm,
      prompts,
      runId,
      parentRunId,
      _extra,
      _tags,
      _meta,
      runName,
    ) => {
      open({
        type: "llm",
        name: runName ?? nameOf(llm),
        runId,
        ...(parentRunId === undefined ? {} : { parentRunId }),
        inputs: prompts,
      });
    },
    handleChatModelStart: (
      llm,
      messages,
      runId,
      parentRunId,
      _extra,
      _tags,
      _meta,
      runName,
    ) => {
      open({
        type: "llm",
        name: runName ?? nameOf(llm),
        runId,
        ...(parentRunId === undefined ? {} : { parentRunId }),
        inputs: messages.flat().map((message) => ({
          role: message.getType(),
          content: message.content,
        })),
      });
    },
    handleLLMEnd: (output, runId) => {
      close({
        runId,
        outputs: output.generations.flat().map((generation) => generation.text),
      });
    },
    handleLLMError: (err, runId) => {
      close({ runId, outputs: null, error: messageOf(err) });
    },
    handleToolStart: (
      tool,
      input,
      runId,
      parentRunId,
      _tags,
      _meta,
      runName,
    ) => {
      open({
        type: "tool",
        name: runName ?? nameOf(tool),
        runId,
        ...(parentRunId === undefined ? {} : { parentRunId }),
        inputs: input,
      });
    },
    handleToolEnd: (output, runId) => {
      close({ runId, outputs: output });
    },
    handleToolError: (err, runId) => {
      close({ runId, outputs: null, error: messageOf(err) });
    },
    handleRetrieverStart: (
      retriever,
      query,
      runId,
      parentRunId,
      _tags,
      _meta,
      runName,
    ) => {
      open({
        type: "retriever",
        name: runName ?? nameOf(retriever),
        runId,
        ...(parentRunId === undefined ? {} : { parentRunId }),
        inputs: query,
      });
    },
    handleRetrieverEnd: (documents, runId) => {
      close({
        runId,
        outputs: documents.map((document) => document.pageContent),
      });
    },
  };

  if (includeChainSteps) {
    handler.handleChainStart = (
      chain,
      inputs,
      runId,
      parentRunId,
      _tags,
      _meta,
      _runType,
      runName,
    ) => {
      open({
        type: "chain",
        name: runName ?? nameOf(chain),
        runId,
        ...(parentRunId === undefined ? {} : { parentRunId }),
        inputs,
      });
    };
    handler.handleChainEnd = (outputs, runId) => {
      close({ runId, outputs });
    };
  }

  return { handler, finish };
}

function nameOf(serialized: { id?: string[] } | undefined): string {
  return serialized?.id?.at(-1) ?? "unknown";
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
