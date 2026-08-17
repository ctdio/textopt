import type { Budget } from "./types.js";

/**
 * Rollouts are GEPA's currency: the paper's efficiency claim is measured in
 * metric calls, not iterations. The engine debits this budget before every
 * evaluation and stops when it can no longer afford the next one.
 *
 * Debiting happens up front, as an atomic reserve-then-refund rather than a
 * check followed by a charge: proposals evaluated concurrently would otherwise
 * each see the same remaining allowance and all spend it.
 */
export function createBudget(args: {
  maxMetricCalls: number;
  /** Rollouts a resumed run already paid for before the checkpoint. */
  spent?: number;
}): Budget {
  const { maxMetricCalls, spent = 0 } = args;

  if (!Number.isFinite(maxMetricCalls) || maxMetricCalls <= 0) {
    throw new Error(
      `maxMetricCalls must be a positive number, received ${maxMetricCalls}`,
    );
  }

  let used = spent;

  return {
    maxMetricCalls,
    spent: () => used,
    remaining: () => maxMetricCalls - used,
    canAfford: (calls: number) => used + calls <= maxMetricCalls,
    reserve: (calls: number) => {
      if (used + calls > maxMetricCalls) {
        return false;
      }
      used += calls;
      return true;
    },
    refund: (calls: number) => {
      used = Math.max(0, used - calls);
    },
  };
}
