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
