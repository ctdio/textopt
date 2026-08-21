import { mapWithConcurrency, priceUsage } from "textopt";
import type {
  Candidate,
  EvaluationContext,
  RolloutUsage,
  ScoreResult,
  TokenPricing,
} from "textopt";
import type {
  GepaAdapter,
  ReflectiveBatch,
  ReflectiveDataset,
  ReflectiveRecord,
} from "textopt/gepa";

/**
 * Structural types matching the AI SDK's `generateText` / `generateObject`
 * results. Declaring them structurally rather than importing from `ai` keeps
 * this package free of a hard dependency and tolerant of SDK version drift.
 */
export interface AiSdkToolCallLike {
  toolName: string;
  input?: unknown;
  /** AI SDK v4 name for `input`. */
  args?: unknown;
}

export interface AiSdkToolResultLike {
  toolName: string;
  output?: unknown;
  /** AI SDK v4 name for `output`. */
  result?: unknown;
}

export interface AiSdkStepLike {
  text?: string;
  finishReason?: string;
  toolCalls?: readonly AiSdkToolCallLike[];
  toolResults?: readonly AiSdkToolResultLike[];
  usage?: AiSdkUsageLike;
}

export interface AiSdkUsageLike {
  inputTokens?: number | undefined;
  outputTokens?: number | undefined;
  totalTokens?: number | undefined;
}

export interface AiSdkResultLike {
  text?: string;
  finishReason?: string;
  steps?: readonly AiSdkStepLike[];
  usage?: AiSdkUsageLike;
}

export interface AiSdkTraceStep {
  index: number;
  text: string;
  finishReason?: string;
  toolCalls?: { toolName: string; input: unknown }[];
  toolResults?: { toolName: string; output: unknown }[];
}

export interface AiSdkTrace {
  steps: AiSdkTraceStep[];
  usage?: AiSdkUsageLike;
  durationMs: number;
  error?: string;
}

/**
 * What this adapter puts in a reflective record's `evidence` slot: the step
 * detail a reflection model needs to diagnose a multi-step run, and the error
 * that ended it. Only present when there is something to diagnose — a
 * single-step run that did not fail is fully described by its output.
 */
export interface AiSdkEvidence {
  trace: AiSdkTraceStep[];
  error?: string;
}

export interface AiSdkAdapterOptions<Datum, Output> {
  /**
   * Execute the system for one dataset row. Return the AI SDK result directly:
   * `run: ({ candidate, datum }) => generateText({ model, system: candidate.system, prompt: datum.q })`
   *
   * `run` (the context, not this option) says where in the optimization this
   * rollout sits. Forward it to whatever tracing the system already has, or a
   * run is thousands of indistinguishable calls.
   *
   * `candidate` is keyed by `string`: this factory never sees the seed
   * candidate, so a component read here is not checked against the ones the
   * optimizer was actually given.
   */
  run: (args: {
    candidate: Candidate;
    datum: Datum;
    run: EvaluationContext;
    signal?: AbortSignal;
  }) => Promise<AiSdkResultLike>;
  /** Defaults to `result.text`. Required when optimizing structured output. */
  toOutput?: (result: AiSdkResultLike) => Output;
  score: (args: {
    datum: Datum;
    output: Output | null;
    result: AiSdkResultLike | null;
    trace: AiSdkTrace;
  }) => ScoreResult | Promise<ScoreResult>;
  concurrency?: number;
  /**
   * Converts the tokens a rollout reported into dollars. Without it usage is
   * still reported in tokens; with it a run can be given a spend ceiling.
   */
  pricing?: TokenPricing;
  /**
   * Classify a thrown error as infrastructure (rate limit, 5xx, network) so
   * its zero is not cached against the candidate. Defaults to treating every
   * failure as the candidate's, which is the safe assumption.
   */
  isTransient?: (err: unknown) => boolean;
  /**
   * Replaces the default reflective record wholesale, `evidence` included.
   * Return `ReflectiveRecord<YourEvidence>` to type your own slot.
   */
  buildRecord?: (args: {
    datum: Datum;
    output: Output | null;
    trace: AiSdkTrace;
    score: number;
    feedback: string;
    component: string;
  }) => ReflectiveRecord;
}

const DEFAULT_CONCURRENCY = 8;

/**
 * Optimizes the text components of a Vercel AI SDK call — system prompts, tool
 * descriptions, output instructions — by re-running `run` with each candidate.
 */
export function createAiSdkAdapter<Datum, Output = string>(
  options: AiSdkAdapterOptions<Datum, Output>,
): GepaAdapter<Datum, AiSdkTrace, Output | null> {
  const {
    run,
    toOutput,
    score,
    concurrency = DEFAULT_CONCURRENCY,
    pricing,
    isTransient = () => false,
    buildRecord,
  } = options;

  const extractOutput =
    toOutput ?? ((result: AiSdkResultLike) => (result.text ?? "") as Output);

  return {
    evaluate: async ({
      batch,
      candidate,
      captureTraces,
      run: context,
      signal,
    }) => {
      const results = await mapWithConcurrency({
        items: batch,
        limit: concurrency,
        task: async (datum) => {
          const startedAt = Date.now();
          let result: AiSdkResultLike;
          let trace: AiSdkTrace;

          try {
            result = await run({ candidate, datum, run: context, signal });
            trace = {
              ...summarizeRun(result),
              durationMs: Date.now() - startedAt,
            };
          } catch (err) {
            // A cancelled run is not a failed rollout: scoring it zero would
            // poison the candidate's record and the evaluation cache.
            signal?.throwIfAborted();
            const message = err instanceof Error ? err.message : String(err);
            const scored: ScoreResult = {
              score: 0,
              feedback: `Run failed: ${message}`,
              transient: isTransient(err),
            };
            const failed: AiSdkTrace = {
              steps: [],
              durationMs: Date.now() - startedAt,
              error: message,
            };
            return { output: null, result: null, trace: failed, scored };
          }

          const output = extractOutput(result);

          try {
            return {
              output,
              result,
              trace,
              scored: await score({ datum, output, result, trace }),
            };
          } catch (err) {
            signal?.throwIfAborted();
            const message = err instanceof Error ? err.message : String(err);
            const scored: ScoreResult = {
              score: 0,
              feedback: `Scoring failed: ${message}`,
              transient: isTransient(err),
            };
            return { output, result, trace, scored };
          }
        },
        signal,
      });

      const evaluation: ReflectiveBatch<AiSdkTrace, Output | null> = {
        outputs: results.map((result) => result.output),
        scores: results.map((result) => result.scored.score),
        feedback: results.map((result) => result.scored.feedback ?? ""),
      };

      if (
        results.some(
          (result) =>
            result.scored.usage !== undefined ||
            result.trace.usage !== undefined,
        )
      ) {
        evaluation.usage = results.map(
          (result) =>
            result.scored.usage ??
            usageOf({ usage: result.trace.usage, pricing }),
        );
      }

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

          const evidence: AiSdkEvidence = {
            trace: trace.steps,
            ...(trace.error === undefined ? {} : { error: trace.error }),
          };

          return {
            inputs: datum,
            generatedOutputs: output,
            feedback,
            score: scoreValue,
            ...(trace.steps.length > 1 || trace.error !== undefined
              ? { evidence }
              : {}),
          };
        });
      }

      return dataset;
    },
  };
}

/**
 * Flattens an AI SDK result into a compact trace. Multi-step agent runs are
 * where the reflection model earns its keep, so tool calls and their results
 * are preserved rather than collapsed into the final text.
 */
export function summarizeRun(result: AiSdkResultLike): AiSdkTrace {
  // An empty `steps` array is as much "no step detail" as an absent one, and
  // dropping the fallback there would discard the generated text entirely.
  const steps =
    result.steps === undefined || result.steps.length === 0
      ? [
          {
            text: result.text ?? "",
            ...(result.finishReason === undefined
              ? {}
              : { finishReason: result.finishReason }),
          },
        ]
      : result.steps;

  return {
    steps: steps.map((step, index) => ({
      index,
      text: step.text ?? "",
      ...(step.finishReason === undefined
        ? {}
        : { finishReason: step.finishReason }),
      ...(step.toolCalls === undefined || step.toolCalls.length === 0
        ? {}
        : {
            toolCalls: step.toolCalls.map((call) => ({
              toolName: call.toolName,
              input: call.input ?? call.args,
            })),
          }),
      ...(step.toolResults === undefined || step.toolResults.length === 0
        ? {}
        : {
            toolResults: step.toolResults.map((toolResult) => ({
              toolName: toolResult.toolName,
              output: toolResult.output ?? toolResult.result,
            })),
          }),
    })),
    ...(result.usage === undefined ? {} : { usage: result.usage }),
    durationMs: 0,
  };
}

/**
 * Turns an AI SDK usage reading into the engine's own. Providers report
 * different subsets of the token fields, so absent ones are left absent rather
 * than zeroed.
 */
function usageOf(args: {
  usage: AiSdkUsageLike | undefined;
  pricing: TokenPricing | undefined;
}): RolloutUsage {
  const { usage, pricing } = args;
  if (usage === undefined) {
    return {};
  }

  const { inputTokens, outputTokens, totalTokens } = usage;
  return priceUsage({
    usage: {
      ...(inputTokens === undefined ? {} : { inputTokens }),
      ...(outputTokens === undefined ? {} : { outputTokens }),
      ...(totalTokens === undefined ? {} : { totalTokens }),
    },
    ...(pricing === undefined ? {} : { pricing }),
  });
}
