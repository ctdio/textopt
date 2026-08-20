export function argmax(values: readonly number[]): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index] as number;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

/** Mean over the values that exist; unscored instances are not zeros. */
export function mean(values: readonly (number | undefined)[]): number {
  let total = 0;
  let count = 0;

  for (const value of values) {
    if (value !== undefined) {
      total += value;
      count += 1;
    }
  }
  return count === 0 ? 0 : total / count;
}

export function sum(values: readonly number[]): number {
  let total = 0;
  for (const value of values) {
    total += value;
  }
  return total;
}

/**
 * One-sided p-value from a paired sign-flip (permutation) test: how often a
 * total this large arises when each difference is equally likely to have come
 * out the other way. Exact by enumeration for small samples, normal-approximated
 * past `maxExact` where 2^n stops being cheap.
 *
 * Paired and distribution-free, which is what a comparison of two runs over the
 * same instances needs — the scores are not independent draws and are not
 * normal, so a t-test on them is answering a different question.
 */
export function signFlipPValue(args: {
  differences: readonly number[];
  observed: number;
  maxExact: number;
}): number {
  const { differences, observed, maxExact } = args;

  const moved = differences.filter((difference) => difference !== 0);
  if (moved.length === 0) {
    return 1;
  }
  if (moved.length > maxExact) {
    return normalTailProbability({ differences: moved, observed });
  }

  const sums = achievableSums(moved);
  const atLeastAsExtreme = sums.filter((total) => total >= observed).length;
  return atLeastAsExtreme / sums.length;
}

/**
 * Holm-Bonferroni step-down adjustment: the p-value each comparison would need
 * to survive on its own if the whole family were held to one error rate,
 * rather than letting the smallest of several tests read as significant by
 * volume alone. Sorted ascending, each rank is scaled by how many comparisons
 * are still in contention at that rank, and the running maximum keeps a later,
 * less-scaled rank from reporting looser than an earlier one already has.
 *
 * `familySize` may exceed `pValues.length`: a comparison `signFlipPValue`
 * could not test at all (every paired difference identical, so no p reflects
 * a real margin) is still a member of the family being controlled for, and
 * excluding it from the denominator would understate the correction owed to
 * the comparisons that could be tested.
 */
export function holmAdjust(args: {
  pValues: readonly number[];
  familySize: number;
}): number[] {
  const { pValues, familySize } = args;

  const ranked = pValues
    .map((pValue, index) => ({ pValue, index }))
    .sort((a, b) => a.pValue - b.pValue);

  const adjusted = new Array<number>(pValues.length);
  let runningMax = 0;
  ranked.forEach(({ pValue, index }, rank) => {
    runningMax = Math.max(
      runningMax,
      Math.min(1, pValue * (familySize - rank)),
    );
    adjusted[index] = runningMax;
  });

  return adjusted;
}

/**
 * Every total reachable by flipping some subset of `differences`' signs,
 * built by doubling rather than by scoring each of the 2^n sign masks
 * independently: after `k` differences there are 2^k sums, and folding in
 * difference `k+1` only ever adds or subtracts it from each of them, so the
 * whole enumeration costs O(2^n) instead of the O(2^n * n) a per-mask loop
 * pays for re-summing n terms every time. Measured at n=20, twenty seeds
 * being the bench's ceiling: about 63ms scoring masks one at a time against
 * about 16ms building sums this way — the difference between affording exact
 * enumeration through a full twenty-seed run and falling back to the normal
 * approximation, which is what raising `EXACT_LIMIT` in compare.ts to 20
 * relies on.
 */
function achievableSums(differences: readonly number[]): number[] {
  let sums = [0];
  for (const difference of differences) {
    const next = new Array<number>(sums.length * 2);
    for (let index = 0; index < sums.length; index += 1) {
      const total = sums[index] as number;
      next[index] = total + difference;
      next[index + sums.length] = total - difference;
    }
    sums = next;
  }
  return sums;
}

/**
 * The same tail probability from a normal approximation, for batches too large
 * to enumerate. Under the sign-flip null each difference has mean zero and
 * variance equal to its square, so the total's variance is their sum.
 */
function normalTailProbability(args: {
  differences: readonly number[];
  observed: number;
}): number {
  const { differences, observed } = args;

  const variance = differences.reduce(
    (total, difference) => total + difference * difference,
    0,
  );
  if (variance === 0) {
    return observed > 0 ? 0 : 1;
  }
  return 1 - standardNormalCdf(observed / Math.sqrt(variance));
}

/** Abramowitz and Stegun 7.1.26, which is accurate to about 1e-7. */
function standardNormalCdf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const scaled = Math.abs(value) / Math.SQRT2;

  const t = 1 / (1 + 0.3275911 * scaled);
  const series =
    t *
    (0.254829592 +
      t *
        (-0.284496736 +
          t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const error = 1 - series * Math.exp(-scaled * scaled);

  return 0.5 * (1 + sign * error);
}
