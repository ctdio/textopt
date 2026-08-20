import { describe, expect, test } from "vitest";
import { argmax, holmAdjust, mean, signFlipPValue, sum } from "./math.js";

describe("argmax", () => {
  test("returns the index of the largest value", () => {
    expect(argmax([0.1, 0.9, 0.4])).toBe(1);
  });

  test("returns the first index of a tie", () => {
    expect(argmax([0.5, 0.5, 0.5])).toBe(0);
  });

  test("returns 0 for an empty array", () => {
    expect(argmax([])).toBe(0);
  });

  test("handles negative values", () => {
    expect(argmax([-3, -1, -2])).toBe(1);
  });
});

describe("mean", () => {
  test("averages over the values that exist, not over the gaps", () => {
    expect(mean([1, undefined, 0])).toBe(0.5);
  });

  test("returns 0 when every value is unknown", () => {
    expect(mean([undefined, undefined])).toBe(0);
  });

  test("returns 0 for an empty array", () => {
    expect(mean([])).toBe(0);
  });

  test("averages a fully scored array", () => {
    expect(mean([1, 2, 3])).toBe(2);
  });
});

describe("sum", () => {
  test("totals the values", () => {
    expect(sum([1, 2, 3.5])).toBe(6.5);
  });

  test("returns 0 for an empty array", () => {
    expect(sum([])).toBe(0);
  });
});

describe("signFlipPValue", () => {
  test("returns 1 when every difference is zero", () => {
    expect(
      signFlipPValue({ differences: [0, 0, 0], observed: 0, maxExact: 16 }),
    ).toBe(1);
  });

  test("finds the unique sign assignment that reaches the observed total", () => {
    // Three same-signed, unequal differences: only flipping none of them
    // reaches the observed sum, since every other assignment loses at least
    // one term's full magnitude. One of eight sign assignments, exactly.
    const differences = [0.5, 0.45, 0.6];
    expect(
      signFlipPValue({
        differences,
        observed: sum(differences),
        maxExact: 16,
      }),
    ).toBeCloseTo(1 / 8);
  });

  test("falls back to the normal approximation past maxExact", () => {
    // Ten unit differences: enumerating exactly gives 1/1024, exact to machine
    // precision; the normal approximation is what runs when maxExact forces it,
    // and the two should already differ measurably at this size.
    const differences = Array.from({ length: 10 }, () => 1);
    const exact = signFlipPValue({
      differences,
      observed: sum(differences),
      maxExact: 16,
    });
    const approximated = signFlipPValue({
      differences,
      observed: sum(differences),
      maxExact: 5,
    });
    expect(exact).toBeCloseTo(1 / 1024, 6);
    expect(approximated).not.toBeCloseTo(exact, 6);
  });

  test("enumerates twenty differences exactly rather than approximating", () => {
    // Same-signed, unequal magnitudes: the true answer is 2^-20, the unique
    // all-positive assignment out of 2^20. The normal approximation this
    // falls back to below `maxExact: 20` gives roughly 4.1e-6 for this same
    // input — measurably different from the exact 9.5367e-7, which is the gap
    // raising `EXACT_LIMIT` in compare.ts to 20 closes.
    const differences = Array.from({ length: 19 }, () => 0.5).concat(0.6);
    const result = signFlipPValue({
      differences,
      observed: sum(differences),
      maxExact: 20,
    });
    expect(result).toBeCloseTo(2 ** -20, 9);
  });
});

describe("holmAdjust", () => {
  test("scales each p-value by its rank among the family, ascending", () => {
    // Sorted ascending: 0.01 (rank 0, x3 = 0.03), 0.02 (rank 1, x2 = 0.04,
    // but 0.04 > 0.03 so it stands), 0.2 (rank 2, x1 = 0.2).
    expect(holmAdjust({ pValues: [0.01, 0.02, 0.2], familySize: 3 })).toEqual([
      0.03, 0.04, 0.2,
    ]);
  });

  test("never lets a later rank undercut an earlier one's adjustment", () => {
    // Sorted ascending: 0.25 (rank 0, x2 = 0.5) then 0.3 (rank 1, x1 = 0.3) —
    // 0.3 alone would read as a loosening after 0.5, so it carries the
    // running maximum instead.
    const [first, second] = holmAdjust({
      pValues: [0.25, 0.3],
      familySize: 2,
    });
    expect(first).toBeCloseTo(0.5);
    expect(second).toBeCloseTo(0.5);
  });

  test("clamps an adjustment past 1 to 1", () => {
    expect(holmAdjust({ pValues: [0.6], familySize: 3 })).toEqual([1]);
  });

  test("counts a family member that could not be tested in the correction", () => {
    // Only one p-value is in hand, but the family it belongs to has three
    // members — a comparison that could not be tested (see signFlipPValue's
    // degenerate case) still competes for the same error budget, so the
    // multiplier is the family size, not the count of testable members.
    const [adjusted] = holmAdjust({ pValues: [0.1], familySize: 3 });
    expect(adjusted).toBeCloseTo(0.3);
  });
});
