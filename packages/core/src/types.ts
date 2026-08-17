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
  signal?: AbortSignal;
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

export interface ProposeArgs {
  candidate: Candidate;
  reflectiveDataset: ReflectiveDataset;
  componentsToUpdate: readonly string[];
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
  instanceScores: number[];
  aggregateScore: number;
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

/** Read-only view handed to candidate selectors. */
export interface SelectionState {
  scoreMatrix: readonly (readonly number[])[];
  aggregateScores: readonly number[];
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

export type BatchSampler<Datum> = (args: {
  trainset: readonly Datum[];
  iteration: number;
  rng: Rng;
}) => number[];

export type AcceptancePolicy = (args: {
  parentScores: readonly number[];
  childScores: readonly number[];
}) => boolean;

export interface Budget {
  readonly maxMetricCalls: number;
  spent(): number;
  remaining(): number;
  canAfford(calls: number): boolean;
  charge(calls: number): void;
}

export interface EvaluationCache {
  get(key: string): number | undefined;
  set(key: string, score: number): void;
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
  | { type: "iterationStart"; iteration: number; parentId: number }
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
    }
  | { type: "error"; iteration: number; err: unknown }
  | {
      type: "finish";
      reason: "budgetExhausted" | "aborted" | "maxIterations";
      bestCandidateId: number;
      metricCalls: number;
    };

export interface OptimizationResult {
  bestCandidate: Candidate;
  bestScore: number;
  bestCandidateId: number;
  candidates: CandidateRecord[];
  paretoFrontier: CandidateRecord[];
  scoreMatrix: number[][];
  metricCalls: number;
  cacheHits: number;
  iterations: number;
  stopReason: "budgetExhausted" | "aborted" | "maxIterations";
}
