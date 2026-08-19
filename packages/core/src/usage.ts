import type { RolloutUsage } from "./types.js";

/** What a model charges, in the per-million-tokens units every vendor quotes. */
export interface TokenPricing {
  inputPerMillionUsd: number;
  outputPerMillionUsd: number;
}

const TOKENS_PER_PRICED_UNIT = 1_000_000;

/**
 * Costs a token reading, leaving it untouched when no price list is given.
 *
 * Prices belong to the caller rather than to this library: they change without
 * warning, differ per account, and a stale table baked in here would report
 * confident numbers that are quietly wrong.
 */
export function priceUsage(args: {
  usage: RolloutUsage;
  pricing?: TokenPricing;
}): RolloutUsage {
  const { usage, pricing } = args;
  if (pricing === undefined) {
    return usage;
  }

  return {
    ...usage,
    costUsd:
      ((usage.inputTokens ?? 0) * pricing.inputPerMillionUsd +
        (usage.outputTokens ?? 0) * pricing.outputPerMillionUsd) /
      TOKENS_PER_PRICED_UNIT,
  };
}
