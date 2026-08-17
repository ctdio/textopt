export { optimize } from "./optimize.js";
export type { OptimizeOptions } from "./optimize.js";

export { createBudget } from "./budget.js";
export { createMemoryCache, evaluationCacheKey } from "./cache.js";
export { mapWithConcurrency } from "./concurrency.js";
export { proposeMerge, selectMergeSubsample } from "./merge.js";
export type { MergeProposal, ProposeMergeArgs } from "./merge.js";
export {
  argmax,
  buildInstanceFronts,
  computeInstanceBests,
  mean,
  pruneDominatedFronts,
  selectParetoCandidate,
  sum,
} from "./pareto.js";
export {
  buildReflectionPrompt,
  createDefaultProposer,
  parseProposedText,
} from "./reflection.js";
export type { ReflectionPromptArgs } from "./reflection.js";
export { createSeededRng } from "./rng.js";
export type { Rng } from "./rng.js";
export {
  allComponentsSelector,
  createEpochShuffledSampler,
  currentBestSelector,
  epsilonGreedySelector,
  improvementAcceptance,
  paretoSelector,
  roundRobinComponentSelector,
  topKParetoSelector,
} from "./strategies.js";
export type {
  AcceptancePolicy,
  Adapter,
  BatchSampler,
  Budget,
  Candidate,
  CandidateRecord,
  CandidateSelector,
  CandidateSource,
  ComponentPatch,
  ComponentSelector,
  EvaluateArgs,
  EvaluationBatch,
  EvaluationCache,
  EvaluationPhase,
  EvaluationSplit,
  MakeReflectiveDatasetArgs,
  OptimizationResult,
  OptimizerEvent,
  ProposeArgs,
  Reflector,
  ReflectiveDataset,
  ReflectiveRecord,
  ScoreResult,
  SelectionState,
} from "./types.js";
