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

    cache.set("key", 0.75);

    expect(cache.get("key")).toBe(0.75);
  });

  test("evicts the oldest entry past maxEntries", () => {
    const cache = createMemoryCache({ maxEntries: 2 });

    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);

    expect(cache.get("a")).toBeUndefined();
    expect(cache.get("b")).toBe(2);
    expect(cache.get("c")).toBe(3);
  });
});
