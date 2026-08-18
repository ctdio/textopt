import type { CachedScore } from "../cache.js";
import type { Rng } from "../rng.js";
import type {
  Adapter,
  Candidate,
  EvaluationBatch,
  EvaluationPhase,
  TextModel,
} from "../types.js";

/**
 * One piece of evidence reflection reads: what went in, what came out, and how
 * it scored. `Evidence` is the adapter's own slot — trace steps, retrieved
 * documents, errors — typed rather than smuggled in through an index signature.
 */
export interface ReflectiveRecord<Evidence = unknown> {
  inputs: unknown;
  generatedOutputs: unknown;
  feedback: string;
  score?: number;
  evidence?: Evidence;
}

/**
 * Component name -> records shown to the reflection model for that component.
 *
 * Partial, not total: an adapter only fills the components it was asked to
 * update, so a component with nothing to diagnose is simply absent.
 */
export type ReflectiveDataset<K extends string = string> = Partial<
  Record<K, ReflectiveRecord[]>
>;

export interface MakeReflectiveDatasetArgs<
  Datum,
  Traj,
  Out,
  K extends string = string,
> {
  candidate: Candidate<K>;
  batch: readonly Datum[];
  evaluation: EvaluationBatch<Traj, Out>;
  componentsToUpdate: readonly K[];
}

/** New text for the subset of components a proposal actually changed. */
export type ComponentPatch<K extends string = string> = Partial<
  Record<K, string>
>;

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

export interface ProposeArgs<K extends string = string> {
  candidate: Candidate<K>;
  reflectiveDataset: ReflectiveDataset<K>;
  componentsToUpdate: readonly K[];
  /** Component name -> texts already tried and rejected for it. */
  rejectedProposals?: Partial<Record<K, RejectedProposal[]>>;
  reflect: TextModel;
  signal?: AbortSignal;
}

/**
 * An adapter GEPA can reflect against: evaluation, plus the traces reflection
 * reads. `makeReflectiveDataset` is what turns a scored batch into the
 * per-component evidence a reflection call is written from.
 */
export interface GepaAdapter<
  Datum,
  Traj = unknown,
  Out = unknown,
  K extends string = string,
> extends Adapter<Datum, Traj, Out, K> {
  makeReflectiveDataset(
    args: MakeReflectiveDatasetArgs<Datum, Traj, Out, K>,
  ): Promise<ReflectiveDataset<K>> | ReflectiveDataset<K>;

  proposeNewTexts?(
    args: ProposeArgs<K>,
  ): Promise<ComponentPatch<K>> | ComponentPatch<K>;
}

export type CandidateSource = "seed" | "mutation" | "merge";

export interface CandidateRecord<K extends string = string> {
  id: number;
  candidate: Candidate<K>;
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
  updatedComponents: K[];
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

export type ComponentSelector<K extends string = string> = (args: {
  candidate: Candidate<K>;
  /** This candidate's own round-robin position, advanced after each selection. */
  cursor: number;
  iteration: number;
  rng: Rng;
}) => K[];

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
export interface ValEvaluationPolicy<
  Datum = unknown,
  K extends string = string,
> {
  selectInstances(args: {
    valset: readonly Datum[];
    candidate: Candidate<K>;
    records: readonly CandidateRecord<K>[];
    iteration: number;
    rng: Rng;
  }): number[];
  bestCandidate(records: readonly CandidateRecord<K>[]): number;
}

export type GepaStopReason =
  "budgetExhausted" | "reflectionBudgetExhausted" | "aborted" | "maxIterations";

export type GepaEvent<K extends string = string> =
  | { type: "start"; components: K[]; valsetSize: number }
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
      componentsToUpdate: K[];
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
      reason: GepaStopReason;
      bestCandidateId: number;
      metricCalls: number;
    };

/**
 * Everything needed to continue a run: the candidate pool with its scores, the
 * budget already spent, the position of the random stream, and the bookkeeping
 * that stops merges and proposals from being relitigated. Plain JSON — write it
 * wherever you like and hand it back as `resumeFrom`.
 *
 * Deliberately not generic over component names. It leaves the process and
 * comes back through `JSON.parse` with plain string keys, so the narrowing back
 * to a run's own components happens once inside the engine, guarded by the
 * fingerprint, rather than being a type the caller has to reconstruct.
 *
 * A resumed run follows the same trajectory an uninterrupted one would, as long
 * as the batch sampler reports its state and the evaluation cache is either
 * checkpointed or disabled — a cache that is neither leaves the resumed run
 * paying again for rollouts the first one had already bought.
 */
export interface GepaSnapshot {
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
