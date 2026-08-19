import type { Budget } from "./budget.js";
import { candidateHash, evaluationCacheKey } from "./cache.js";
import type { CachedScore, EvaluationCache } from "./cache.js";
import { mean } from "./math.js";
import type {
  Adapter,
  Candidate,
  EvaluationBatch,
  EvaluationPhase,
  EvaluationSplit,
} from "./types.js";

/**
 * One evaluation, reported as it happens. Optimizer-agnostic: every search
 * pays for rollouts the same way, and a caller watching cost should not have
 * to know which algorithm is spending it.
 */
export interface EvaluationEvent {
  iteration: number;
  phase: EvaluationPhase;
  split: EvaluationSplit;
  candidateId: number | null;
  /** Rollouts this evaluation actually bought. Cached instances are not here. */
  metricCalls: number;
  cacheHits: number;
  meanScore: number;
}

/** Scores plus, when the adapter reports them, their per-objective breakdown. */
export interface ScoredBatch<Output> {
  scores: number[];
  objectiveScores: (Record<string, number> | undefined)[];
  /** Populated only under `trackOutputs`, and only for fresh rollouts. */
  outputs: (Output | undefined)[];
  /** Per instance: the score came from an infrastructure failure, not the candidate. */
  transient: boolean[];
}

export interface EvaluateBatchArgs<Datum, K extends string> {
  candidate: Candidate<K>;
  batch: readonly Datum[];
  /** Instance ids, aligned with `batch`, naming rows in the cache. */
  ids: readonly string[];
  split: EvaluationSplit;
  phase: EvaluationPhase;
  candidateId: number | null;
  iteration: number;
  /**
   * Whether these rollouts come out of the search budget. Measurement taken
   * after the search has chosen a winner passes false: charging it would let
   * the size of a held-out set change which candidate wins.
   */
  charge?: boolean;
}

export interface EvaluateTracedArgs<Datum, K extends string> {
  candidate: Candidate<K>;
  batch: readonly Datum[];
  split: EvaluationSplit;
  phase: EvaluationPhase;
  candidateId: number | null;
  iteration: number;
}

/**
 * The part of an optimizer that spends money. Every search built on this
 * package shares it, so caching, budgeting, transient-failure handling and
 * cost reporting behave identically whichever algorithm is running.
 */
export interface Evaluator<Datum, Trajectory, Output, K extends string> {
  evaluate(args: EvaluateBatchArgs<Datum, K>): Promise<ScoredBatch<Output>>;
  /**
   * A rollout with traces captured, always fresh. Reflection reads the traces,
   * and the cache stores scores rather than trajectories, so a cached instance
   * has nothing to reflect on — serving one here would silently hand the
   * reflection model an empty dataset.
   *
   * Returns null when the budget cannot cover the batch, rather than throwing:
   * a run that cannot afford to reflect is finished, not broken.
   */
  evaluateTraced(
    args: EvaluateTracedArgs<Datum, K>,
  ): Promise<EvaluationBatch<Trajectory, Output> | null>;
  /**
   * How many of `ids` this candidate has not been scored on yet — what a sweep
   * would actually cost. Lets a caller price an evaluation before committing
   * to it, instead of discovering the shortfall halfway through.
   */
  countUncached(args: {
    candidate: Candidate<K>;
    ids: readonly string[];
    split: EvaluationSplit;
  }): number;
  /** Instances served from the cache, which no budget was charged for. */
  cacheHits(): number;
  /** Rollouts made with `charge: false`, tracked apart from the budget. */
  unchargedCalls(): number;
  /** Cache contents for checkpointing, when the cache can enumerate them. */
  entries(): [string, CachedScore][] | undefined;
  restore(entries: Iterable<readonly [string, CachedScore]>): void;
}

/**
 * Raised when a reservation cannot be met mid-flight. A concurrent evaluation
 * cannot check the budget and then spend it — another may take the remainder
 * in between — so running out is reported where it happens and turned into a
 * stop reason by whichever loop is driving.
 */
export class BudgetExhausted extends Error {}

export function createEvaluator<
  Datum,
  Trajectory,
  Output,
  K extends string,
>(args: {
  adapter: Adapter<Datum, Trajectory, Output, K>;
  budget: Budget;
  /** Omit to run uncached; every instance is then a fresh rollout. */
  cache?: EvaluationCache;
  /** Keep what each rollout produced. Costs memory proportional to outputs. */
  trackOutputs?: boolean;
  onEvaluation?: (event: EvaluationEvent) => void;
  signal?: AbortSignal;
  /** Resumed counters, so a continued run reports totals rather than deltas. */
  cacheHits?: number;
}): Evaluator<Datum, Trajectory, Output, K> {
  const {
    adapter,
    budget,
    cache,
    trackOutputs = false,
    onEvaluation,
    signal,
    cacheHits: initialCacheHits = 0,
  } = args;

  let cacheHits = initialCacheHits;
  let unchargedCalls = 0;

  return {
    countUncached: ({ candidate, ids, split }) => {
      if (cache === undefined) {
        return ids.length;
      }
      const hash = candidateHash(candidate);
      return ids.filter(
        (id) =>
          cache.get(evaluationCacheKey({ hash, instanceId: id, split })) ===
          undefined,
      ).length;
    },

    evaluateTraced: async ({
      candidate,
      batch,
      split,
      phase,
      candidateId,
      iteration,
    }) => {
      if (!budget.reserve(batch.length)) {
        return null;
      }

      let evaluation: EvaluationBatch<Trajectory, Output>;
      try {
        evaluation = await adapter.evaluate({
          batch,
          candidate,
          captureTraces: true,
          run: { iteration, phase, split, candidateId },
          signal,
        });
        assertEvaluation({ evaluation, expected: batch.length });
      } catch (err) {
        budget.refund(batch.length);
        throw err;
      }

      onEvaluation?.({
        iteration,
        phase,
        split,
        candidateId,
        metricCalls: batch.length,
        cacheHits: 0,
        meanScore: mean(evaluation.scores),
      });

      return evaluation;
    },

    evaluate: async (call) => {
      const {
        candidate,
        batch,
        ids,
        split,
        phase,
        candidateId,
        iteration,
        charge = true,
      } = call;

      const hash = candidateHash(candidate);
      const scores = new Array<number>(batch.length);
      const objectiveScores = new Array<Record<string, number> | undefined>(
        batch.length,
      );
      const outputs = new Array<Output | undefined>(batch.length).fill(
        undefined,
      );
      const transient = new Array<boolean>(batch.length).fill(false);
      const pendingIndices: number[] = [];

      for (let index = 0; index < batch.length; index += 1) {
        const cached = cache?.get(
          evaluationCacheKey({
            hash,
            instanceId: ids[index] as string,
            split,
          }),
        );
        if (cached === undefined) {
          pendingIndices.push(index);
        } else {
          scores[index] = cached.score;
          objectiveScores[index] = cached.objectiveScores;
        }
      }

      cacheHits += batch.length - pendingIndices.length;

      if (pendingIndices.length > 0) {
        if (charge && !budget.reserve(pendingIndices.length)) {
          throw new BudgetExhausted();
        }
        if (!charge) {
          unchargedCalls += pendingIndices.length;
        }

        let evaluation: EvaluationBatch<Trajectory, Output>;
        try {
          evaluation = await adapter.evaluate({
            batch: pendingIndices.map((index) => batch[index] as Datum),
            candidate,
            captureTraces: false,
            run: { iteration, phase, split, candidateId },
            signal,
          });
          assertEvaluation({
            evaluation,
            expected: pendingIndices.length,
          });
        } catch (err) {
          // Nothing was measured, so nothing is owed.
          if (charge) {
            budget.refund(pendingIndices.length);
          } else {
            unchargedCalls -= pendingIndices.length;
          }
          throw err;
        }

        pendingIndices.forEach((batchIndex, resultIndex) => {
          const score = evaluation.scores[resultIndex] as number;
          const objectives = evaluation.objectiveScores?.[resultIndex];
          scores[batchIndex] = score;
          objectiveScores[batchIndex] = objectives;
          if (trackOutputs) {
            outputs[batchIndex] = evaluation.outputs[resultIndex];
          }

          // A transient score says nothing about the candidate, so caching it
          // would pin this instance to an infrastructure failure forever.
          if (evaluation.transient?.[resultIndex] === true) {
            transient[batchIndex] = true;
            return;
          }
          cache?.set(
            evaluationCacheKey({
              hash,
              instanceId: ids[batchIndex] as string,
              split,
            }),
            objectives === undefined
              ? { score }
              : { score, objectiveScores: objectives },
          );
        });
      }

      onEvaluation?.({
        iteration,
        phase,
        split,
        candidateId,
        metricCalls: pendingIndices.length,
        cacheHits: batch.length - pendingIndices.length,
        meanScore: mean(scores),
      });

      return { scores, objectiveScores, outputs, transient };
    },

    cacheHits: () => cacheHits,
    unchargedCalls: () => unchargedCalls,
    entries: () => cache?.entries?.(),
    restore: (entries) => {
      for (const [key, cached] of entries) {
        cache?.set(key, cached);
      }
    },
  };
}

/**
 * A NaN score never raises on its own: every comparison that decides fronts or
 * the best candidate is false for NaN, so the candidate silently becomes an
 * unselectable phantom that still consumed budget. Every other array is read
 * positionally against the batch, so a short one misattributes a diagnosis, an
 * objective or an infrastructure failure to the wrong instance. Catch both at
 * the boundary.
 */
function assertEvaluation(args: {
  evaluation: EvaluationBatch<unknown, unknown>;
  expected: number;
}): void {
  const { evaluation, expected } = args;
  const { scores } = evaluation;

  const aligned: [string, { length: number } | undefined][] = [
    ["scores", scores],
    ["outputs", evaluation.outputs],
    ["feedback", evaluation.feedback],
    ["objectiveScores", evaluation.objectiveScores],
    ["transient", evaluation.transient],
  ];
  for (const [name, values] of aligned) {
    if (values !== undefined && values.length !== expected) {
      throw new Error(
        `Adapter returned ${values.length} ${name} for a batch of ${expected}; ${name} must align one-to-one with the batch`,
      );
    }
  }

  for (let index = 0; index < scores.length; index += 1) {
    const score = scores[index];
    if (typeof score !== "number" || !Number.isFinite(score)) {
      throw new Error(
        `Adapter returned a non-finite score at index ${index}: ${String(score)}`,
      );
    }
  }
}
