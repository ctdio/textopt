import type { Budget } from "./budget.js";
import { candidateHash, evaluationCacheKey } from "./cache.js";
import type { CachedScore, EvaluationCache } from "./cache.js";
import type {
  Adapter,
  Candidate,
  EvaluationBatch,
  EvaluationPhase,
  EvaluationSplit,
  RolloutUsage,
  UsageTotals,
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
  /** Tokens and money the run has spent, as far as adapters have reported it. */
  usage(): UsageTotals;
  /** Cache contents for checkpointing, when the cache can enumerate them. */
  entries(): [string, CachedScore][] | undefined;
  restore(entries: Iterable<readonly [string, CachedScore]>): void;
}

/**
 * How often a rollout the adapter reported as infrastructure failure is tried
 * again before its instance is left unmeasured.
 */
export interface RetryPolicy {
  /** Extra attempts per instance, beyond the first. Zero disables retrying. */
  attempts?: number;
  /** Wait before the first retry. Doubled for each attempt after it. */
  delayMs?: number;
}

const DEFAULT_RETRY: Required<RetryPolicy> = { attempts: 2, delayMs: 500 };

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
  /**
   * Rate limits and 5xx responses are the common case in a long run, and a
   * transient row costs the instance whichever optimizer is driving: it is
   * either an unexplained zero or a hole in the candidate's coverage. Retrying
   * here fixes it once for every optimizer rather than in each search loop.
   */
  retry?: RetryPolicy;
  /** Scopes every cache key to the system these rollouts were measured under. */
  cacheNamespace?: string;
}): Evaluator<Datum, Trajectory, Output, K> {
  const {
    adapter,
    budget,
    cache,
    trackOutputs = false,
    onEvaluation,
    signal,
    cacheHits: initialCacheHits = 0,
    retry,
    cacheNamespace,
  } = args;

  const { attempts: retryAttempts, delayMs: retryDelayMs } = {
    ...DEFAULT_RETRY,
    ...retry,
  };

  let cacheHits = initialCacheHits;
  let unchargedCalls = 0;
  const usage: UsageTotals = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    costUsd: 0,
    rollouts: 0,
  };

  /** Folds one adapter call's reported usage into the run's totals. */
  function recordUsage(args: {
    evaluation: EvaluationBatch<Trajectory, Output>;
    rollouts: number;
  }): void {
    usage.rollouts += args.rollouts;
    for (const rollout of args.evaluation.usage ?? []) {
      addUsage({ totals: usage, rollout });
    }
  }

  /**
   * Runs `rows` of `batch`, retrying the instances the adapter reports as
   * infrastructure failures. Retries are charged like any other rollout —
   * they cost the same money — but they are never allowed to overdraw: a run
   * that cannot afford another attempt keeps the transient row instead.
   */
  async function runWithRetries(call: {
    candidate: Candidate<K>;
    rows: readonly Datum[];
    captureTraces: boolean;
    split: EvaluationSplit;
    phase: EvaluationPhase;
    candidateId: number | null;
    iteration: number;
    charge: boolean;
  }): Promise<{
    evaluation: EvaluationBatch<Trajectory, Output>;
    calls: number;
  }> {
    const { rows } = call;

    let merged = await runRows(call);
    let calls = rows.length;

    for (let attempt = 0; attempt < retryAttempts; attempt += 1) {
      const failed = transientIndices(merged);
      if (
        failed.length === 0 ||
        !affordable({ calls: failed.length, charge: call.charge })
      ) {
        return { evaluation: merged, calls };
      }

      await delay(retryDelayMs * 2 ** attempt);
      signal?.throwIfAborted();

      const retried = await runRows({
        ...call,
        rows: failed.map((index) => rows[index] as Datum),
      });
      calls += failed.length;
      merged = mergeRows({ base: merged, positions: failed, rows: retried });
    }
    return { evaluation: merged, calls };
  }

  /** One adapter call, charged up front and refunded if it never happened. */
  async function runRows(call: {
    candidate: Candidate<K>;
    rows: readonly Datum[];
    captureTraces: boolean;
    split: EvaluationSplit;
    phase: EvaluationPhase;
    candidateId: number | null;
    iteration: number;
    charge: boolean;
  }): Promise<EvaluationBatch<Trajectory, Output>> {
    const { candidate, rows, captureTraces, charge, ...context } = call;

    reserve({ calls: rows.length, charge });
    try {
      const evaluation = await adapter.evaluate({
        batch: rows,
        candidate,
        captureTraces,
        run: context,
        signal,
      });
      assertEvaluation({ evaluation, expected: rows.length });
      recordUsage({ evaluation, rollouts: rows.length });
      return evaluation;
    } catch (err) {
      release({ calls: rows.length, charge });
      throw err;
    }
  }

  function affordable(args: { calls: number; charge: boolean }): boolean {
    return !args.charge || budget.canAfford(args.calls);
  }

  function reserve(args: { calls: number; charge: boolean }): void {
    if (!args.charge) {
      unchargedCalls += args.calls;
      return;
    }
    if (!budget.reserve(args.calls)) {
      throw new BudgetExhausted();
    }
  }

  function release(args: { calls: number; charge: boolean }): void {
    if (args.charge) {
      budget.refund(args.calls);
    } else {
      unchargedCalls -= args.calls;
    }
  }

  return {
    countUncached: ({ candidate, ids, split }) => {
      if (cache === undefined) {
        return ids.length;
      }
      const hash = candidateHash(candidate);
      return ids.filter(
        (id) =>
          cache.get(
            evaluationCacheKey({
              hash,
              instanceId: id,
              split,
              ...(cacheNamespace === undefined
                ? {}
                : { namespace: cacheNamespace }),
            }),
          ) === undefined,
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
      let run: {
        evaluation: EvaluationBatch<Trajectory, Output>;
        calls: number;
      };
      try {
        run = await runWithRetries({
          candidate,
          rows: batch,
          captureTraces: true,
          split,
          phase,
          candidateId,
          iteration,
          charge: true,
        });
      } catch (err) {
        // A batch the budget cannot cover is the run ending, not a failure:
        // the caller decides whether it can still do anything useful.
        if (err instanceof BudgetExhausted) {
          return null;
        }
        throw err;
      }

      onEvaluation?.({
        iteration,
        phase,
        split,
        candidateId,
        metricCalls: run.calls,
        cacheHits: 0,
        meanScore: measuredMean(run.evaluation) ?? 0,
      });

      return run.evaluation;
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
            ...(cacheNamespace === undefined
              ? {}
              : { namespace: cacheNamespace }),
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

      let metricCalls = 0;

      if (pendingIndices.length > 0) {
        const run = await runWithRetries({
          candidate,
          rows: pendingIndices.map((index) => batch[index] as Datum),
          captureTraces: false,
          split,
          phase,
          candidateId,
          iteration,
          charge,
        });
        const evaluation = run.evaluation;
        metricCalls = run.calls;

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
              ...(cacheNamespace === undefined
                ? {}
                : { namespace: cacheNamespace }),
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
        metricCalls,
        cacheHits: batch.length - pendingIndices.length,
        meanScore: measuredMean({ scores, transient }) ?? 0,
      });

      return { scores, objectiveScores, outputs, transient };
    },

    cacheHits: () => cacheHits,
    unchargedCalls: () => unchargedCalls,
    usage: () => ({ ...usage }),
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
    ["usage", evaluation.usage],
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

/**
 * Mean over the rows that measured the candidate. Transient rows measured the
 * infrastructure instead, so averaging their zeros in would reject a candidate
 * for an outage; undefined when no row measured anything at all, which is a
 * batch that says nothing rather than a batch that says zero.
 */
export function measuredMean(batch: {
  scores: readonly number[];
  transient?: readonly boolean[];
}): number | undefined {
  const { scores, transient } = batch;

  let total = 0;
  let count = 0;
  for (let index = 0; index < scores.length; index += 1) {
    if (transient?.[index] === true) {
      continue;
    }
    total += scores[index] as number;
    count += 1;
  }
  return count === 0 ? undefined : total / count;
}

/**
 * Whether a run has spent what it was allowed to. Checked between evaluations,
 * because usage is only known once a rollout has been paid for.
 */
export function costExhausted(args: {
  usage: UsageTotals;
  maxCostUsd?: number;
}): boolean {
  return args.maxCostUsd !== undefined && args.usage.costUsd >= args.maxCostUsd;
}

/**
 * The measured mean of an evaluation a run cannot continue without — its seed
 * baseline, and the sweeps it compares everything against.
 *
 * Reporting zero for a batch in which nothing ran would set the search a
 * baseline no rollout produced, and every later comparison would be made
 * against it. Failing here names the provider outage instead.
 */
export function requireMeasuredMean(args: {
  batch: { scores: readonly number[]; transient?: readonly boolean[] };
  phase: string;
}): number {
  const value = measuredMean(args.batch);
  if (value === undefined) {
    throw new Error(
      `Every rollout in the ${args.phase} evaluation failed transiently; it measured the infrastructure rather than the candidate`,
    );
  }
  return value;
}

/**
 * Adds one rollout's reading to a running total. `totalTokens` is derived when
 * the provider reports only the two halves, because a caller comparing runs
 * should not have to know which providers report which fields.
 */
function addUsage(args: { totals: UsageTotals; rollout: RolloutUsage }): void {
  const { totals, rollout } = args;
  const { inputTokens = 0, outputTokens = 0, costUsd = 0 } = rollout;

  totals.inputTokens += inputTokens;
  totals.outputTokens += outputTokens;
  totals.totalTokens += rollout.totalTokens ?? inputTokens + outputTokens;
  totals.costUsd += costUsd;
}

function transientIndices(
  evaluation: EvaluationBatch<unknown, unknown>,
): number[] {
  const { transient } = evaluation;
  if (transient === undefined) {
    return [];
  }

  const indices: number[] = [];
  for (let index = 0; index < transient.length; index += 1) {
    if (transient[index] === true) {
      indices.push(index);
    }
  }
  return indices;
}

/**
 * Writes a re-run of some rows back over the batch they came from, field by
 * field. Aligned arrays are read positionally everywhere downstream, so a
 * retried row has to land in the position its instance occupies.
 */
function mergeRows<Trajectory, Output>(args: {
  base: EvaluationBatch<Trajectory, Output>;
  positions: readonly number[];
  rows: EvaluationBatch<Trajectory, Output>;
}): EvaluationBatch<Trajectory, Output> {
  const { base, positions, rows } = args;

  const merged: EvaluationBatch<Trajectory, Output> = {
    outputs: [...base.outputs],
    scores: [...base.scores],
    ...(base.feedback === undefined ? {} : { feedback: [...base.feedback] }),
    ...(base.trajectories === undefined
      ? {}
      : { trajectories: [...base.trajectories] }),
    ...(base.objectiveScores === undefined
      ? {}
      : { objectiveScores: [...base.objectiveScores] }),
    transient: base.transient === undefined ? [] : [...base.transient],
  };

  positions.forEach((position, row) => {
    merged.outputs[position] = rows.outputs[row] as Output;
    merged.scores[position] = rows.scores[row] as number;
    if (merged.feedback !== undefined) {
      merged.feedback[position] = rows.feedback?.[row] ?? "";
    }
    if (merged.trajectories !== undefined && rows.trajectories !== undefined) {
      merged.trajectories[position] = rows.trajectories[row] as Trajectory;
    }
    if (merged.objectiveScores !== undefined) {
      const objectives = rows.objectiveScores?.[row];
      if (objectives !== undefined) {
        merged.objectiveScores[position] = objectives;
      }
    }
    (merged.transient as boolean[])[position] = rows.transient?.[row] === true;
  });

  return merged;
}

function delay(milliseconds: number): Promise<void> {
  if (milliseconds <= 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}
