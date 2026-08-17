import { describe, expect, test } from "vitest";
import { createSeededRng } from "./rng.js";

describe("createSeededRng", () => {
  test("produces the same sequence for the same seed", () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);

    const first = [a.next(), a.next(), a.next()];
    const second = [b.next(), b.next(), b.next()];

    expect(first).toEqual(second);
  });

  test("produces a different sequence for a different seed", () => {
    const a = createSeededRng(1);
    const b = createSeededRng(2);

    expect(a.next()).not.toEqual(b.next());
  });

  test("returns values in the unit interval", () => {
    const rng = createSeededRng(7);

    for (let i = 0; i < 1000; i += 1) {
      const value = rng.next();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  test("nextInt stays below the exclusive maximum", () => {
    const rng = createSeededRng(9);

    for (let i = 0; i < 500; i += 1) {
      const value = rng.nextInt(5);
      expect(Number.isInteger(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(5);
    }
  });

  test("pick returns an element of the input array", () => {
    const rng = createSeededRng(3);
    const items = ["a", "b", "c"];

    for (let i = 0; i < 50; i += 1) {
      expect(items).toContain(rng.pick(items));
    }
  });

  test("shuffle returns a permutation without mutating the input", () => {
    const rng = createSeededRng(11);
    const items = [1, 2, 3, 4, 5];

    const shuffled = rng.shuffle(items);

    expect(items).toEqual([1, 2, 3, 4, 5]);
    expect([...shuffled].sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
  });

  test("shuffle is deterministic for a given seed", () => {
    const items = [1, 2, 3, 4, 5, 6, 7, 8];

    expect(createSeededRng(5).shuffle(items)).toEqual(
      createSeededRng(5).shuffle(items),
    );
  });

  test("sample returns k distinct items", () => {
    const selected = createSeededRng(3).sample([1, 2, 3, 4, 5], 3);

    expect(selected).toHaveLength(3);
    expect(new Set(selected).size).toBe(3);
  });

  test("sample returns every item when k exceeds the input length", () => {
    expect(new Set(createSeededRng(3).sample([1, 2, 3], 10))).toEqual(
      new Set([1, 2, 3]),
    );
  });

  test("sample returns nothing for a non-positive k", () => {
    expect(createSeededRng(3).sample([1, 2, 3], 0)).toEqual([]);
  });

  test("weighted never returns a zero-weighted item", () => {
    const rng = createSeededRng(9);

    for (let i = 0; i < 100; i += 1) {
      expect(rng.weighted(["a", "b", "c"], [0, 1, 0])).toBe("b");
    }
  });

  test("weighted favours the heavier item", () => {
    const rng = createSeededRng(9);

    const picks = Array.from({ length: 200 }, () =>
      rng.weighted(["a", "b"], [9, 1]),
    );

    expect(picks.filter((pick) => pick === "a").length).toBeGreaterThan(150);
  });

  test("weighted falls back to a uniform pick when every weight is zero", () => {
    const rng = createSeededRng(9);
    const picks = new Set<string>();

    for (let i = 0; i < 100; i += 1) {
      picks.add(rng.weighted(["a", "b", "c"], [0, 0, 0]));
    }

    expect(picks).toEqual(new Set(["a", "b", "c"]));
  });

  test("weighted throws on an empty array", () => {
    expect(() => createSeededRng(1).weighted([], [])).toThrow(/empty/i);
  });
});
