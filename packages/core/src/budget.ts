import type { Budget } from "./types.js";

/**
 * Rollouts are GEPA's currency: the paper's efficiency claim is measured in
 * metric calls, not iterations. The engine debits this budget before every
 * evaluation and stops when it can no longer afford the next one.
 */
export function createBudget(args: { maxMetricCalls: number }): Budget {
  const { maxMetricCalls } = args;

  if (!Number.isFinite(maxMetricCalls) || maxMetricCalls <= 0) {
    throw new Error(
      `maxMetricCalls must be a positive number, received ${maxMetricCalls}`,
    );
  }

  let used = 0;

  return {
    maxMetricCalls,
    spent: () => used,
    remaining: () => maxMetricCalls - used,
    canAfford: (calls: number) => used + calls <= maxMetricCalls,
    charge: (calls: number) => {
      if (used + calls > maxMetricCalls) {
        throw new Error(
          `Metric call budget exceeded: tried to charge ${calls} with ${maxMetricCalls - used} remaining`,
        );
      }
      used += calls;
    },
  };
}
