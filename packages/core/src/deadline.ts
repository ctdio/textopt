export interface Deadline {
  exceeded(): boolean;
  /** Milliseconds left, `Infinity` when no limit was set. Never negative. */
  remainingMs(): number;
}

/**
 * A wall-clock limit on a run.
 *
 * Rollout and cost ceilings both bound what a run *spends*, and neither bounds
 * how long it takes: a run waiting on a rate-limited provider can sit for an
 * hour without spending a dollar. A deadline is what makes an optimizer safe to
 * put behind a request timeout or a nightly job.
 *
 * Checked between evaluations, so a run overruns by at most the length of one.
 * The clock is injectable because a deadline that can only be tested by waiting
 * is a deadline nobody tests.
 */
export function createDeadline(args: {
  maxWallClockMs?: number;
  now?: () => number;
}): Deadline {
  const { maxWallClockMs, now = Date.now } = args;
  const startedAt = now();

  return {
    exceeded: () =>
      maxWallClockMs !== undefined && now() - startedAt >= maxWallClockMs,
    remainingMs: () =>
      maxWallClockMs === undefined
        ? Number.POSITIVE_INFINITY
        : Math.max(0, maxWallClockMs - (now() - startedAt)),
  };
}
