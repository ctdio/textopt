import { describe, expect, test } from "vitest";
import { createMemoryCache, evaluationCacheKey } from "./cache.js";

describe("evaluationCacheKey", () => {
  test("is stable across component key ordering", () => {
    const first = evaluationCacheKey({
      candidate: { a: "one", b: "two" },
      instanceId: "0",
      split: "val",
    });
    const second = evaluationCacheKey({
      candidate: { b: "two", a: "one" },
      instanceId: "0",
      split: "val",
    });

    expect(first).toBe(second);
  });

  test("separates train and val instances that share an id", () => {
    const train = evaluationCacheKey({
      candidate: { a: "one" },
      instanceId: "0",
      split: "train",
    });
    const val = evaluationCacheKey({
      candidate: { a: "one" },
      instanceId: "0",
      split: "val",
    });

    expect(train).not.toBe(val);
  });

  test("changes when component text changes", () => {
    const first = evaluationCacheKey({
      candidate: { a: "one" },
      instanceId: "0",
      split: "val",
    });
    const second = evaluationCacheKey({
      candidate: { a: "onE" },
      instanceId: "0",
      split: "val",
    });

    expect(first).not.toBe(second);
  });

  test("changes when the instance changes", () => {
    const first = evaluationCacheKey({
      candidate: { a: "one" },
      instanceId: "0",
      split: "val",
    });
    const second = evaluationCacheKey({
      candidate: { a: "one" },
      instanceId: "1",
      split: "val",
    });

    expect(first).not.toBe(second);
  });
});

describe("createMemoryCache", () => {
  test("returns undefined for an unknown key", () => {
    expect(createMemoryCache().get("missing")).toBeUndefined();
  });

  test("returns the stored score", () => {
    const cache = createMemoryCache();

    cache.set("key", { score: 0.75 });

    expect(cache.get("key")?.score).toBe(0.75);
  });

  test("returns the stored objective scores", () => {
    const cache = createMemoryCache();

    cache.set("key", {
      score: 0.5,
      objectiveScores: { accuracy: 1, brevity: 0 },
    });

    expect(cache.get("key")?.objectiveScores).toEqual({
      accuracy: 1,
      brevity: 0,
    });
  });

  test("evicts the oldest entry past maxEntries", () => {
    const cache = createMemoryCache({ maxEntries: 2 });

    cache.set("a", { score: 1 });
    cache.set("b", { score: 2 });
    cache.set("c", { score: 3 });

    expect(cache.get("a")).toBeUndefined();
    expect(cache.get("b")?.score).toBe(2);
    expect(cache.get("c")?.score).toBe(3);
  });

  test("exposes its entries for checkpointing", () => {
    const cache = createMemoryCache();

    cache.set("a", { score: 1 });
    cache.set("b", { score: 2, objectiveScores: { accuracy: 1 } });

    expect(cache.entries?.()).toEqual([
      ["a", { score: 1 }],
      ["b", { score: 2, objectiveScores: { accuracy: 1 } }],
    ]);
  });

  test("restores entries it was seeded with", () => {
    const cache = createMemoryCache({ entries: [["a", { score: 0.25 }]] });

    expect(cache.get("a")?.score).toBe(0.25);
  });
});
