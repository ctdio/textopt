import type { Rng } from "./rng.js";

/**
 * A candidate is a map of named text components to their current text. This is
 * the only thing GEPA evolves — prompts, instructions, code, tool descriptions,
 * anything expressible as a named string.
 */
export type Candidate = Record<string, string>;

/** New text for the subset of components a proposal actually changed. */
export type ComponentPatch = Record<string, string>;

/**
 * Result of running a candidate over a batch of data instances.
 *
 * `scores` is the load-bearing field: one number per instance, higher is
 * better. `feedback` is what makes GEPA more than hill climbing — a textual
 * diagnosis per instance that the reflection model reads.
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

export interface EvaluateArgs<Datum> {
  batch: readonly Datum[];
  candidate: Candidate;
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

export interface ReflectiveRecord {
  inputs: unknown;
  generatedOutputs: unknown;
  feedback: string;
  score?: number;
  [key: string]: unknown;
}

/** Component name -> records shown to the reflection model for that component. */
export type ReflectiveDataset = Record<string, ReflectiveRecord[]>;

export interface MakeReflectiveDatasetArgs<Datum, Traj, Out> {
  candidate: Candidate;
  batch: readonly Datum[];
  evaluation: EvaluationBatch<Traj, Out>;
  componentsToUpdate: readonly string[];
}

/**
 * A component text that was proposed and lost on the minibatch. Showing these
 * back to the reflection model is what stops a run from re-deriving the same
 * dead end: only exact duplicates are filtered structurally, so without this
 * the model can spend the whole budget circling one bad idea.
 */
export interface RejectedProposal {
  text: string;
  parentScore: number;
  childScore: number;
}

export interface ProposeArgs {
  candidate: Candidate;
  reflectiveDataset: ReflectiveDataset;
  componentsToUpdate: readonly string[];
  /** Component name -> texts already tried and rejected for it. */
  rejectedProposals?: Record<string, RejectedProposal[]>;
  reflect: Reflector;
  signal?: AbortSignal;
}

/**
 * The single integration seam between GEPA and a system under optimization.
 * Everything framework-specific — LangChain, the AI SDK, Braintrust — lives in
 * an implementation of this interface.
 */
export interface Adapter<Datum, Traj = unknown, Out = unknown> {
  evaluate(
    args: EvaluateArgs<Datum>,
  ): Promise<EvaluationBatch<Traj, Out>> | EvaluationBatch<Traj, Out>;

  makeReflectiveDataset(
    args: MakeReflectiveDatasetArgs<Datum, Traj, Out>,
  ): Promise<ReflectiveDataset> | ReflectiveDataset;

  proposeNewTexts?(args: ProposeArgs): Promise<ComponentPatch> | ComponentPatch;
}

/** Provider-agnostic reflection model: text in, text out. */
export type Reflector = (args: {
  prompt: string;
  signal?: AbortSignal;
}) => Promise<string>;

export type CandidateSource = "seed" | "mutation" | "merge";

export interface CandidateRecord {
  id: number;
  candidate: Candidate;
  parentIds: number[];
  /**
   * One entry per validation instance. `undefined` marks an instance the
   * evaluation policy did not select for this candidate — unknown, not zero.
   */
  instanceScores: (number | undefined)[];
  /** Mean over the instances that were scored. */
  aggregateScore: number;
  /** Mean of each objective over the evaluated validation instances. */
  objectiveScores?: Record<string, number>;
  source: CandidateSource;
  updatedComponents: string[];
  iteration: number;
  /**
   * Round-robin position this candidate resumes from the next time it is
   * chosen as a parent. Per-candidate rather than global: a lineage that is
   * only reselected every fifth iteration must still walk its own components
   * one at a time instead of being pinned to whichever one the global counter
   * happens to land on.
   */
  componentCursor: number;
}

/** What a Pareto frontier is taken over. */
export type ParetoFrontier = "instance" | "objective" | "hybrid";

/** Read-only view handed to candidate selectors. */
export interface SelectionState {
  scoreMatrix: readonly (readonly (number | undefined)[])[];
  aggregateScores: readonly number[];
  /** Per-candidate objective means, absent when the adapter reports none. */
  objectiveScores?: readonly (Readonly<Record<string, number>> | undefined)[];
}

export type CandidateSelector = (args: {
  state: SelectionState;
  rng: Rng;
}) => number;

export type ComponentSelector = (args: {
  candidate: Candidate;
  /** This candidate's own round-robin position, advanced after each selection. */
  cursor: number;
  iteration: number;
  rng: Rng;
}) => string[];

export type BatchSampler<Datum> = ((args: {
  trainset: readonly Datum[];
  iteration: number;
  rng: Rng;
}) => number[]) & {
  /**
   * Position within the sampler's own schedule, checkpointed alongside the
   * random stream. Without it a resumed run restarts its epoch and re-walks
   * minibatches the interrupted run had already spent.
   */
  state?: () => unknown;
  restore?: (state: unknown) => void;
};

export type AcceptancePolicy = (args: {
  parentScores: readonly number[];
  childScores: readonly number[];
}) => boolean;

/**
 * Which validation instances a candidate is scored on, and how the best
 * candidate is read back out of possibly partial coverage.
 *
 * A full sweep per accepted candidate is the published behaviour and the
 * default. Swapping in a partial policy trades frontier fidelity for rollouts:
 * candidates scored on different instances are no longer strictly comparable,
 * which is why picking the best is the policy's job too.
 */
export interface ValEvaluationPolicy<Datum = unknown> {
  selectInstances(args: {
    valset: readonly Datum[];
    candidate: Candidate;
    records: readonly CandidateRecord[];
    iteration: number;
    rng: Rng;
  }): number[];
  bestCandidate(records: readonly CandidateRecord[]): number;
}

export interface Budget {
  readonly maxMetricCalls: number;
  spent(): number;
  remaining(): number;
  canAfford(calls: number): boolean;
  /** Debits `calls` atomically. False when the allowance cannot cover them. */
  reserve(calls: number): boolean;
  /** Credits back calls a reservation did not end up spending. */
  refund(calls: number): void;
}

/**
 * What the cache stores per (candidate, instance): the metric the frontier is
 * built from, plus the per-objective breakdown when the adapter reports one.
 * Both come from the same rollout, so caching the score without the objectives
 * would force a re-run to recover them.
 */
export interface CachedScore {
  score: number;
  objectiveScores?: Record<string, number>;
}

export interface EvaluationCache {
  get(key: string): CachedScore | undefined;
  set(key: string, cached: CachedScore): void;
  /** Entries for checkpointing. Omit on caches that are already durable. */
  entries?(): [string, CachedScore][];
}

/**
 * Which dataset an instance id was drawn from. Train and val ids are numbered
 * independently, so the same id can name two different instances; the cache
 * key has to keep them apart.
 */
export type EvaluationSplit = "train" | "val";

export type EvaluationPhase = "seed" | "minibatch" | "validation";

export type OptimizerEvent =
  | { type: "start"; components: string[]; valsetSize: number }
  | { type: "iterationStart"; iteration: number; parentIds: number[] }
  | {
      type: "evaluation";
      iteration: number;
      phase: EvaluationPhase;
      candidateId: number | null;
      metricCalls: number;
      cacheHits: number;
      meanScore: number;
    }
  | {
      type: "proposal";
      iteration: number;
      parentId: number;
      componentsToUpdate: string[];
      changed: boolean;
    }
  | {
      type: "candidateAccepted";
      iteration: number;
      candidateId: number;
      parentIds: number[];
      aggregateScore: number;
      source: CandidateSource;
    }
  | {
      type: "candidateRejected";
      iteration: number;
      parentId: number;
      parentScore: number;
      childScore: number;
      source: CandidateSource;
      /**
       * "worse" lost to its parent on the minibatch. "notSelected" beat its
       * parent but lost to a stronger proposal from the same iteration — it is
       * not a dead end, and is never fed back to reflection as one.
       */
      reason: "worse" | "notSelected";
    }
  | { type: "error"; iteration: number; err: unknown }
  | {
      type: "finish";
      reason: StopReason;
      bestCandidateId: number;
      metricCalls: number;
    };

export type StopReason =
  "budgetExhausted" | "reflectionBudgetExhausted" | "aborted" | "maxIterations";

/**
 * Everything needed to continue a run: the candidate pool with its scores, the
 * budget already spent, the position of the random stream, and the bookkeeping
 * that stops merges and proposals from being relitigated. Plain JSON — write it
 * wherever you like and hand it back as `resumeFrom`.
 *
 * A resumed run follows the same trajectory an uninterrupted one would, as long
 * as the batch sampler reports its state and the evaluation cache is either
 * checkpointed or disabled — a cache that is neither leaves the resumed run
 * paying again for rollouts the first one had already bought.
 */
export interface OptimizerSnapshot {
  version: 1;
  /**
   * Identifies the run this checkpoint came from — seed candidate, instance
   * ids and seed. Resuming against a different setup is refused rather than
   * silently scoring old candidates against new data.
   */
  fingerprint: string;
  records: CandidateRecord[];
  iteration: number;
  metricCalls: number;
  reflectionCalls: number;
  cacheHits: number;
  rngState: number;
  /** Whatever the batch sampler reports from `state()`, when it has one. */
  sampler?: unknown;
  rejectedProposals: Record<string, RejectedProposal[]>;
  merge: {
    attempts: string[];
    descriptions: string[];
    due: number;
    tested: number;
    lastIterationAccepted: boolean;
  };
  /** Cached instance scores, when the cache can enumerate them. */
  cache?: [string, CachedScore][];
}

export interface OptimizationResult<Out = unknown> {
  bestCandidate: Candidate;
  bestScore: number;
  bestCandidateId: number;
  /**
   * What the best candidate actually produced on each validation instance,
   * when `trackBestOutputs` is on. `undefined` where the score came from the
   * cache rather than from a rollout of this candidate.
   */
  bestOutputs?: (Out | undefined)[];
  candidates: CandidateRecord[];
  paretoFrontier: CandidateRecord[];
  /**
   * Per objective: the best value reached and every candidate that reached it.
   * Absent when the adapter reports no objective scores.
   */
  perObjectiveBest?: Record<string, { score: number; candidateIds: number[] }>;
  scoreMatrix: (number | undefined)[][];
  metricCalls: number;
  /** Calls made to the reflection model, which no metric budget covers. */
  reflectionCalls: number;
  cacheHits: number;
  iterations: number;
  stopReason: StopReason;
  /** State as of the last iteration, ready to hand back as `resumeFrom`. */
  snapshot: OptimizerSnapshot;
}
