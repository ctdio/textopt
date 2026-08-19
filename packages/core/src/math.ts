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

  const assignments = 2 ** moved.length;
  let atLeastAsExtreme = 0;

  for (let mask = 0; mask < assignments; mask += 1) {
    let total = 0;
    for (let index = 0; index < moved.length; index += 1) {
      const sign = (mask >> index) & 1 ? -1 : 1;
      total += sign * (moved[index] as number);
    }
    if (total >= observed) {
      atLeastAsExtreme += 1;
    }
  }
  return atLeastAsExtreme / assignments;
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
