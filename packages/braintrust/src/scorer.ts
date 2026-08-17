import type { ScoreResult } from "@ctdio/gepa";

/** Matches the `Score` object returned by braintrust and autoevals scorers. */
export interface BraintrustScoreLike {
  name: string;
  score: number | null;
  metadata?: Record<string, unknown>;
}

export interface BraintrustScorerArgs<Output = string> {
  output: Output;
  expected?: Output;
  input?: unknown;
  metadata?: Record<string, unknown>;
}

export type BraintrustScorerFn<Output = string> = (
  args: BraintrustScorerArgs<Output>,
) => BraintrustScoreLike | number | Promise<BraintrustScoreLike | number>;

export interface BraintrustScorerOptions<Output> {
  scorers: readonly BraintrustScorerFn<Output>[];
  /** Relative weight per scorer name. Missing names default to 1. */
  weights?: Record<string, number>;
  /** Override how scorer output becomes reflection feedback. */
  buildFeedback?: (scores: readonly BraintrustScoreLike[]) => string;
  /**
   * Classify a thrown scorer error as infrastructure (rate limit, 5xx,
   * network) rather than as the candidate's doing. A scorer that failed this
   * way leaves the composite computed from whichever scorers survived, so the
   * whole result is reported transient and never cached. Defaults to treating
   * every failure as the candidate's, which is the safe assumption.
   */
  isTransient?: (err: unknown) => boolean;
}

/**
 * Turns braintrust / autoevals scorers into a GEPA metric.
 *
 * The important part is not the number — it is that scorer metadata (an
 * LLM judge's rationale, a diff, a validation error) is carried through as
 * feedback, which is what the reflection model actually reads.
 */
export function createBraintrustScorer<Output = string>(
  options: BraintrustScorerOptions<Output>,
): (args: BraintrustScorerArgs<Output>) => Promise<ScoreResult> {
  const {
    scorers,
    weights,
    buildFeedback = defaultFeedback,
    isTransient = () => false,
  } = options;

  assertUsableWeights(weights);

  return async (args) => {
    const outcomes = await Promise.all(
      scorers.map(async (scorer, index) => {
        try {
          return {
            score: normalizeScore({ value: await scorer(args), index }),
            transient: false,
          };
        } catch (err) {
          return {
            score: {
              name: `scorer_${index}`,
              score: null,
              metadata: {
                error: err instanceof Error ? err.message : String(err),
              },
            } satisfies BraintrustScoreLike,
            transient: isTransient(err),
          };
        }
      }),
    );

    const settled = outcomes.map((outcome) => outcome.score);
    // One infrastructure failure degrades the whole composite, not just its
    // own objective: what survives is a blend over a different set of scorers
    // than the one the candidate is supposed to be measured on.
    const degraded = outcomes.some((outcome) => outcome.transient);
    const usable = settled.filter((score) => score.score !== null);
    assertDistinctNames(usable);

    const objectiveScores: Record<string, number> = {};
    let weightedTotal = 0;
    let weightSum = 0;

    for (const score of usable) {
      const weight = weights?.[score.name] ?? 1;
      objectiveScores[score.name] = score.score as number;
      weightedTotal += (score.score as number) * weight;
      weightSum += weight;
    }

    // Every scorer failing is a legitimate zero. Every scorer being weighted
    // to nothing is a config error, and returning a plausible-looking zero
    // would hide it.
    if (usable.length === 0) {
      return {
        score: 0,
        feedback: buildFeedback(settled),
        objectiveScores,
        ...(degraded ? { transient: true } : {}),
      };
    }
    if (weightSum === 0) {
      throw new Error(
        `Every scorer that produced a score has weight 0 (${usable
          .map((score) => score.name)
          .join(", ")}); the composite score would be undefined`,
      );
    }

    return {
      score: weightedTotal / weightSum,
      feedback: buildFeedback(settled),
      objectiveScores,
      ...(degraded ? { transient: true } : {}),
    };
  };
}

/**
 * A negative weight can push the blended score outside [0, 1], and a
 * non-finite one propagates NaN into the Pareto matrix where it silently
 * disqualifies the candidate rather than raising.
 */
function assertUsableWeights(
  weights: Record<string, number> | undefined,
): void {
  for (const [name, weight] of Object.entries(weights ?? {})) {
    if (!Number.isFinite(weight) || weight < 0) {
      throw new Error(
        `Scorer weight for "${name}" must be a finite number >= 0, received ${weight}`,
      );
    }
  }
}

/**
 * Two scorers reporting one name would make the blended score count both while
 * `objectiveScores` kept only the last, and would leave `weights[name]`
 * ambiguous between them.
 */
function assertDistinctNames(scores: readonly BraintrustScoreLike[]): void {
  const seen = new Set<string>();

  for (const score of scores) {
    if (seen.has(score.name)) {
      throw new Error(
        `Two scorers reported the name "${score.name}"; scorer names must be unique`,
      );
    }
    seen.add(score.name);
  }
}

function normalizeScore(args: {
  value: BraintrustScoreLike | number;
  index: number;
}): BraintrustScoreLike {
  const { value, index } = args;

  if (typeof value === "number") {
    return { name: `scorer_${index}`, score: value };
  }
  return value;
}

function defaultFeedback(scores: readonly BraintrustScoreLike[]): string {
  return scores
    .map((score) => {
      const rationale =
        score.metadata?.rationale ??
        score.metadata?.reason ??
        score.metadata?.error;
      const value = score.score === null ? "skipped" : score.score.toFixed(3);

      return rationale === undefined
        ? `${score.name}: ${value}`
        : `${score.name}: ${value} — ${String(rationale)}`;
    })
    .join("\n");
}
