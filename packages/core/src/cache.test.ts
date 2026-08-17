import { describe, expect, test } from "vitest";
import {
  candidateHash,
  createMemoryCache,
  evaluationCacheKey,
} from "./cache.js";

describe("candidateHash", () => {
  test("is stable across component key ordering", () => {
    expect(candidateHash({ a: "one", b: "two" })).toBe(
      candidateHash({ b: "two", a: "one" }),
    );
  });

  test("changes when component text changes", () => {
    expect(candidateHash({ a: "one" })).not.toBe(candidateHash({ a: "onE" }));
  });

  test("separates text moved between components", () => {
    expect(candidateHash({ a: "one", b: "" })).not.toBe(
      candidateHash({ a: "", b: "one" }),
    );
  });
});

describe("evaluationCacheKey", () => {
  test("separates train and val instances that share an id", () => {
    const hash = candidateHash({ a: "one" });

    expect(
      evaluationCacheKey({ hash, instanceId: "0", split: "train" }),
    ).not.toBe(evaluationCacheKey({ hash, instanceId: "0", split: "val" }));
  });

  test("changes when the candidate changes", () => {
    const first = evaluationCacheKey({
      hash: candidateHash({ a: "one" }),
      instanceId: "0",
      split: "val",
    });
    const second = evaluationCacheKey({
      hash: candidateHash({ a: "onE" }),
      instanceId: "0",
      split: "val",
    });

    expect(first).not.toBe(second);
  });

  test("changes when the instance changes", () => {
    const hash = candidateHash({ a: "one" });

    expect(
      evaluationCacheKey({ hash, instanceId: "0", split: "val" }),
    ).not.toBe(evaluationCacheKey({ hash, instanceId: "1", split: "val" }));
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
