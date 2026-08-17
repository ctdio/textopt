export { optimize } from "./optimize.js";
export type { OptimizeOptions } from "./optimize.js";

export { createBudget } from "./budget.js";
export {
  candidateHash,
  createMemoryCache,
  evaluationCacheKey,
  stableHash,
} from "./cache.js";
export { mapWithConcurrency } from "./concurrency.js";
export { proposeMerge, selectMergeSubsample } from "./merge.js";
export type { MergeProposal, ProposeMergeArgs } from "./merge.js";
export {
  argmax,
  buildInstanceFronts,
  buildObjectiveFronts,
  computeInstanceBests,
  mean,
  objectiveBests,
  pruneDominatedFronts,
  selectParetoCandidate,
  sum,
} from "./pareto.js";
export {
  buildReflectionPrompt,
  createDefaultProposer,
  limitReflectiveRecords,
  parseProposedText,
} from "./reflection.js";
export type {
  ReflectionLimits,
  ReflectionPromptArgs,
  ReflectionPromptBuilder,
} from "./reflection.js";
export { createSeededRng } from "./rng.js";
export type { Rng } from "./rng.js";
export {
  allComponentsSelector,
  createEpochShuffledSampler,
  currentBestSelector,
  epsilonGreedySelector,
  fullEvaluationPolicy,
  improvementAcceptance,
  paretoSelector,
  roundRobinComponentSelector,
  subsampledEvaluationPolicy,
  topKParetoSelector,
} from "./strategies.js";
export type {
  AcceptancePolicy,
  Adapter,
  BatchSampler,
  Budget,
  CachedScore,
  Candidate,
  CandidateRecord,
  CandidateSelector,
  CandidateSource,
  ComponentPatch,
  ComponentSelector,
  EvaluateArgs,
  EvaluationBatch,
  EvaluationCache,
  EvaluationContext,
  EvaluationPhase,
  EvaluationSplit,
  MakeReflectiveDatasetArgs,
  OptimizationResult,
  OptimizerEvent,
  OptimizerSnapshot,
  ParetoFrontier,
  ProposeArgs,
  Reflector,
  ReflectiveDataset,
  ReflectiveRecord,
  RejectedProposal,
  ScoreResult,
  SelectionState,
  StopReason,
  ValEvaluationPolicy,
} from "./types.js";
