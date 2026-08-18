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
 * Result of running a candidate over a batch of data instances.
 *
 * `scores` is the load-bearing field: one number per instance, higher is
 * better. `feedback` is a per-instance textual diagnosis of what went wrong,
 * which a reflective optimizer reads to write a better candidate.
 */
export interface EvaluationBatch<Traj = unknown, Out = unknown> {
  outputs: Out[];
  scores: number[];
  feedback?: string[];
  trajectories?: Traj[];
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

export type EvaluationPhase = "seed" | "minibatch" | "validation";

/**
 * Which dataset an instance id was drawn from. Train and val ids are numbered
 * independently, so the same id can name two different instances; the cache
 * key has to keep them apart.
 */
export type EvaluationSplit = "train" | "val";

/**
 * The single integration seam between an optimizer and a system under
 * optimization. Everything framework-specific — LangChain, the AI SDK,
 * Braintrust — lives in an implementation of this interface.
 */
export interface Adapter<
  Datum,
  Traj = unknown,
  Out = unknown,
  K extends string = string,
> {
  evaluate(
    args: EvaluateArgs<Datum, K>,
  ): Promise<EvaluationBatch<Traj, Out>> | EvaluationBatch<Traj, Out>;
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
