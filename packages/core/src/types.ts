/**
 * A candidate is a map of named text components to their current text. This is
 * the unit of optimization — prompts, instructions, code, tool descriptions,
 * anything expressible as a named string.
 *
 * `K` is the union of component names, inferred from the seed candidate, so a
 * misspelled component is a compile error rather than a silent no-op.
 */
export type Candidate<K extends string = string> = Record<K, string>;

/**
 * What one rollout consumed. Every field is optional because providers report
 * different subsets, and a partial reading is still worth more than none.
 */
export interface RolloutUsage {
  inputTokens?: number;
  outputTokens?: number;
  /** Defaults to the sum of the two token counts when they are reported. */
  totalTokens?: number;
  costUsd?: number;
}

/** Usage summed over a run, alongside the rollouts that produced it. */
export interface UsageTotals {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  costUsd: number;
  /** Fresh rollouts counted here. Cached instances buy nothing. */
  rollouts: number;
}

/**
 * Result of running a candidate over a batch of data instances.
 *
 * `scores` is the load-bearing field: one number per instance, higher is
 * better. `feedback` is a per-instance textual diagnosis of what went wrong,
 * which a reflective optimizer reads to write a better candidate.
 */
export interface EvaluationBatch<Trajectory = unknown, Output = unknown> {
  outputs: Output[];
  scores: number[];
  /**
   * What each rollout consumed. Rollout counts are the budget, but they are a
   * poor proxy for spend: reflective search grows the text it optimizes, so
   * the same rollout costs more late in a run than early in it.
   */
  usage?: RolloutUsage[];
  feedback?: string[];
  trajectories?: Trajectory[];
  objectiveScores?: Record<string, number>[];
  /**
   * Per-instance: true when the score reflects an infrastructure failure
   * rather than the candidate's behaviour. Transient scores are never written
   * to the evaluation cache.
   */
  transient?: boolean[];
}

/**
 * What a per-instance scorer returns. Shared by every adapter so scorers are
 * portable between them — a Braintrust scorer works in a LangChain run.
 */
export interface ScoreResult {
  score: number;
  feedback?: string;
  objectiveScores?: Record<string, number>;
  /** What this rollout consumed, when the caller can see it. */
  usage?: RolloutUsage;
  /**
   * Marks a score produced by an infrastructure failure — a rate limit, a
   * network blip, a provider 5xx — rather than by the candidate. Without this
   * the engine cannot tell such a zero from a genuine one, and would cache it
   * permanently against the candidate.
   */
  transient?: boolean;
}

export interface EvaluateArgs<Datum, K extends string = string> {
  batch: readonly Datum[];
  candidate: Candidate<K>;
  captureTraces: boolean;
  /**
   * Where this batch sits in the run. Forward it to whatever tracing the
   * system under optimization already has — without it a run is thousands of
   * indistinguishable rollouts, and no trace can be tied back to the iteration
   * whose score moved.
   */
  run: EvaluationContext;
  /**
   * Call as each rollout settles. It is what turns a batch into progress: the
   * optimizer emits a `rollout` event per call, which on a slow model is the
   * only thing between `start` and the end of a validation sweep that says the
   * run is moving rather than hung.
   *
   * Optional, and absent when nothing is listening. An adapter that never
   * calls it simply reports no progress below the batch.
   */
  onRollout?: () => void;
  signal?: AbortSignal;
}

/**
 * Identifies one evaluation within a run. `candidateId` is null while the
 * candidate is still a proposal being screened on a minibatch: it has no
 * record, and inventing an id for it would collide with the one it gets if it
 * is accepted.
 */
export interface EvaluationContext {
  iteration: number;
  phase: EvaluationPhase;
  split: EvaluationSplit;
  candidateId: number | null;
}

export type EvaluationPhase = "seed" | "minibatch" | "validation" | "test";

/**
 * Which dataset an instance id was drawn from. Each split numbers its ids
 * independently, so the same id can name three different instances; the cache
 * key has to keep them apart.
 */
export type EvaluationSplit = "train" | "val" | "test";

/**
 * The single integration seam between an optimizer and a system under
 * optimization. Everything framework-specific — LangChain, the AI SDK,
 * Braintrust — lives in an implementation of this interface.
 */
export interface Adapter<
  Datum,
  Trajectory = unknown,
  Output = unknown,
  K extends string = string,
> {
  evaluate(
    args: EvaluateArgs<Datum, K>,
  ):
    | Promise<EvaluationBatch<Trajectory, Output>>
    | EvaluationBatch<Trajectory, Output>;
}

/** Provider-agnostic text model: text in, text out. */
export type TextModel = (args: {
  prompt: string;
  signal?: AbortSignal;
}) => Promise<string>;

/**
 * The component names of a candidate, as the union they were inferred from.
 *
 * `Object.keys` widens a closed key union back to `string`. This is the one
 * place that narrowing happens, so every other caller stays assertion-free.
 * Accepts a partial so it also names the components of a component patch.
 */
export function componentNames<K extends string>(
  candidate: Partial<Candidate<K>>,
): K[] {
  return Object.keys(candidate) as K[];
}
