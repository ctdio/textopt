import { describe, expect, test } from "vitest";
import { createBudget } from "./budget.js";

describe("createBudget", () => {
  test("starts with the full allowance remaining", () => {
    const budget = createBudget({ maxMetricCalls: 100 });

    expect(budget.remaining()).toBe(100);
    expect(budget.spent()).toBe(0);
  });

  test("charging reduces the remaining allowance", () => {
    const budget = createBudget({ maxMetricCalls: 100 });

    budget.charge(30);

    expect(budget.spent()).toBe(30);
    expect(budget.remaining()).toBe(70);
  });

  test("canAfford is false once the request exceeds what is left", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    budget.charge(8);

    expect(budget.canAfford(2)).toBe(true);
    expect(budget.canAfford(3)).toBe(false);
  });

  test("throws when charged beyond the allowance", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    expect(() => budget.charge(11)).toThrow(/budget/i);
  });

  test("rejects a non-positive allowance", () => {
    expect(() => createBudget({ maxMetricCalls: 0 })).toThrow(/maxMetricCalls/);
  });
});
