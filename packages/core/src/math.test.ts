import { describe, expect, test } from "vitest";
import { argmax, mean, sum } from "./math.js";

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
