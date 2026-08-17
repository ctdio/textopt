import { describe, expect, test } from "vitest";
import { createBudget } from "./budget.js";

describe("createBudget", () => {
  test("starts with the full allowance remaining", () => {
    const budget = createBudget({ maxMetricCalls: 100 });

    expect(budget.remaining()).toBe(100);
    expect(budget.spent()).toBe(0);
  });

  test("reserving reduces the remaining allowance", () => {
    const budget = createBudget({ maxMetricCalls: 100 });

    budget.reserve(30);

    expect(budget.spent()).toBe(30);
    expect(budget.remaining()).toBe(70);
  });

  test("canAfford is false once the request exceeds what is left", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    budget.reserve(8);

    expect(budget.canAfford(2)).toBe(true);
    expect(budget.canAfford(3)).toBe(false);
  });

  test("refuses a reservation beyond the allowance", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    expect(budget.reserve(11)).toBe(false);
    expect(budget.spent()).toBe(0);
  });

  test("grants a reservation it can afford", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    expect(budget.reserve(10)).toBe(true);
  });

  test("refunding returns unspent calls to the allowance", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    budget.reserve(6);
    budget.refund(4);

    expect(budget.spent()).toBe(2);
    expect(budget.remaining()).toBe(8);
  });

  test("never refunds below zero", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    budget.reserve(2);
    budget.refund(5);

    expect(budget.spent()).toBe(0);
  });

  test("only one of two concurrent reservations for the same calls succeeds", () => {
    const budget = createBudget({ maxMetricCalls: 10 });

    const granted = [budget.reserve(7), budget.reserve(7)];

    expect(granted).toEqual([true, false]);
  });

  test("rejects a non-positive allowance", () => {
    expect(() => createBudget({ maxMetricCalls: 0 })).toThrow(/maxMetricCalls/);
  });
});
