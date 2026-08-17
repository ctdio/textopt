import { describe, expect, test } from "vitest";
import { proposeMerge, selectMergeSubsample } from "./merge.js";
import { createSeededRng } from "./rng.js";
import type { Candidate, CandidateRecord } from "./types.js";

function record(args: {
  id: number;
  candidate: Candidate;
  parentIds?: number[];
  aggregateScore?: number;
  instanceScores?: number[];
}): CandidateRecord {
  const aggregateScore = args.aggregateScore ?? 0;

  return {
    id: args.id,
    candidate: args.candidate,
    parentIds: args.parentIds ?? [],
    instanceScores: args.instanceScores ?? [aggregateScore],
    aggregateScore,
    source: args.parentIds === undefined ? "seed" : "mutation",
    updatedComponents: [],
    iteration: args.id,
    componentCursor: 0,
  };
}

const SEED = record({
  id: 0,
  candidate: { retriever: "base retriever", writer: "base writer" },
});

function merge(args: {
  records: CandidateRecord[];
  pool: number[];
  attempted?: Set<string>;
  attemptedDescriptions?: Set<string>;
}) {
  return proposeMerge({
    records: args.records,
    pool: args.pool,
    rng: createSeededRng(1),
    attempted: args.attempted ?? new Set(),
    attemptedDescriptions: args.attemptedDescriptions ?? new Set(),
  });
}

describe("proposeMerge", () => {
  test("returns null when the pool has fewer than two candidates", () => {
    expect(merge({ records: [SEED], pool: [0] })).toBeNull();
  });

  test("combines complementary components from two lineages", () => {
    const left = record({
      id: 1,
      candidate: { retriever: "better retriever", writer: "base writer" },
      parentIds: [0],
      aggregateScore: 0.6,
    });
    const right = record({
      id: 2,
      candidate: { retriever: "base retriever", writer: "better writer" },
      parentIds: [0],
      aggregateScore: 0.7,
    });

    const merged = merge({ records: [SEED, left, right], pool: [1, 2] });

    expect(merged?.candidate).toEqual({
      retriever: "better retriever",
      writer: "better writer",
    });
    expect(merged?.parentIds).toEqual([1, 2]);
    expect(merged?.ancestorId).toBe(0);
  });

  test("returns null when both lineages changed the same component", () => {
    const left = record({
      id: 1,
      candidate: { retriever: "left retriever", writer: "base writer" },
      parentIds: [0],
    });
    const right = record({
      id: 2,
      candidate: { retriever: "right retriever", writer: "base writer" },
      parentIds: [0],
    });

    expect(merge({ records: [SEED, left, right], pool: [1, 2] })).toBeNull();
  });

  test("takes a doubly-changed component from the stronger lineage", () => {
    const left = record({
      id: 1,
      candidate: { retriever: "left retriever", writer: "left writer" },
      parentIds: [0],
      aggregateScore: 0.4,
    });
    const right = record({
      id: 2,
      candidate: { retriever: "right retriever", writer: "base writer" },
      parentIds: [0],
      aggregateScore: 0.8,
    });

    const merged = merge({ records: [SEED, left, right], pool: [1, 2] });

    expect(merged?.candidate).toEqual({
      retriever: "right retriever",
      writer: "left writer",
    });
  });

  test("refuses to merge a candidate with its own ancestor", () => {
    const parent = record({
      id: 1,
      candidate: { retriever: "new retriever", writer: "base writer" },
      parentIds: [0],
      aggregateScore: 0.5,
    });
    const child = record({
      id: 2,
      candidate: { retriever: "new retriever", writer: "new writer" },
      parentIds: [1],
      aggregateScore: 0.6,
    });

    expect(merge({ records: [SEED, parent, child], pool: [1, 2] })).toBeNull();
  });

  test("rejects an ancestor that outscores one of its descendants", () => {
    const mid = record({
      id: 1,
      candidate: { retriever: "mid retriever", writer: "mid writer" },
      parentIds: [0],
      aggregateScore: 0.9,
    });
    const left = record({
      id: 2,
      candidate: { retriever: "left retriever", writer: "mid writer" },
      parentIds: [1],
      aggregateScore: 0.5,
    });
    const right = record({
      id: 3,
      candidate: { retriever: "mid retriever", writer: "right writer" },
      parentIds: [1],
      aggregateScore: 0.5,
    });

    expect(
      merge({ records: [SEED, mid, left, right], pool: [2, 3] }),
    ).toBeNull();
  });

  test("uses the ancestor both descendants improved on", () => {
    const mid = record({
      id: 1,
      candidate: { retriever: "mid retriever", writer: "mid writer" },
      parentIds: [0],
      aggregateScore: 0.4,
    });
    const left = record({
      id: 2,
      candidate: { retriever: "left retriever", writer: "mid writer" },
      parentIds: [1],
      aggregateScore: 0.5,
    });
    const right = record({
      id: 3,
      candidate: { retriever: "mid retriever", writer: "right writer" },
      parentIds: [1],
      aggregateScore: 0.5,
    });

    const merged = merge({ records: [SEED, mid, left, right], pool: [2, 3] });

    expect(merged?.ancestorId).toBe(1);
    expect(merged?.candidate).toEqual({
      retriever: "left retriever",
      writer: "right writer",
    });
  });

  test("only merges candidates in the pool", () => {
    const inPoolLeft = record({
      id: 1,
      candidate: { retriever: "pool retriever", writer: "base writer" },
      parentIds: [0],
      aggregateScore: 0.5,
    });
    const inPoolRight = record({
      id: 2,
      candidate: { retriever: "base retriever", writer: "pool writer" },
      parentIds: [0],
      aggregateScore: 0.5,
    });
    const outOfPool = record({
      id: 3,
      candidate: { retriever: "base retriever", writer: "excluded writer" },
      parentIds: [0],
      aggregateScore: 0.9,
    });

    const merged = merge({
      records: [SEED, inPoolLeft, inPoolRight, outOfPool],
      pool: [1, 2],
    });

    expect(merged?.candidate).toEqual({
      retriever: "pool retriever",
      writer: "pool writer",
    });
  });

  test("returns null when the merge would duplicate an existing candidate", () => {
    const left = record({
      id: 1,
      candidate: { retriever: "better retriever", writer: "base writer" },
      parentIds: [0],
    });
    const right = record({
      id: 2,
      candidate: { retriever: "base retriever", writer: "better writer" },
      parentIds: [0],
    });
    const alreadyMerged = record({
      id: 3,
      candidate: { retriever: "better retriever", writer: "better writer" },
      parentIds: [1, 2],
    });

    expect(
      merge({ records: [SEED, left, right, alreadyMerged], pool: [1, 2] }),
    ).toBeNull();
  });

  test("skips triplets that were already attempted", () => {
    const left = record({
      id: 1,
      candidate: { retriever: "better retriever", writer: "base writer" },
      parentIds: [0],
    });
    const right = record({
      id: 2,
      candidate: { retriever: "base retriever", writer: "better writer" },
      parentIds: [0],
    });

    expect(
      merge({
        records: [SEED, left, right],
        pool: [1, 2],
        attempted: new Set(["1:2:0"]),
      }),
    ).toBeNull();
  });

  test("skips a merge that would repeat an earlier component selection", () => {
    const left = record({
      id: 1,
      candidate: { retriever: "better retriever", writer: "base writer" },
      parentIds: [0],
    });
    const right = record({
      id: 2,
      candidate: { retriever: "base retriever", writer: "better writer" },
      parentIds: [0],
    });
    const records = [SEED, left, right];

    const first = merge({ records, pool: [1, 2] });

    expect(
      merge({
        records,
        pool: [1, 2],
        attemptedDescriptions: new Set([first?.descriptionKey as string]),
      }),
    ).toBeNull();
  });

  test("returns a stable attempt key for the chosen triplet", () => {
    const left = record({
      id: 1,
      candidate: { retriever: "better retriever", writer: "base writer" },
      parentIds: [0],
    });
    const right = record({
      id: 2,
      candidate: { retriever: "base retriever", writer: "better writer" },
      parentIds: [0],
    });

    expect(
      merge({ records: [SEED, left, right], pool: [1, 2] })?.attemptKey,
    ).toBe("1:2:0");
  });
});

describe("selectMergeSubsample", () => {
  test("returns no more instances than the requested size", () => {
    const selected = selectMergeSubsample({
      scores1: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
      scores2: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
      rng: createSeededRng(1),
      size: 5,
    });

    expect(selected).toHaveLength(5);
  });

  test("covers instances each parent uniquely wins", () => {
    const selected = selectMergeSubsample({
      scores1: [1, 1, 1, 0, 0, 0, 0.5, 0.5, 0.5],
      scores2: [0, 0, 0, 1, 1, 1, 0.5, 0.5, 0.5],
      rng: createSeededRng(3),
      size: 5,
    });

    const firstWins = selected.filter((index) => index < 3);
    const secondWins = selected.filter((index) => index >= 3 && index < 6);

    expect(firstWins.length).toBeGreaterThan(0);
    expect(secondWins.length).toBeGreaterThan(0);
  });

  test("returns every instance when there are fewer than the requested size", () => {
    const selected = selectMergeSubsample({
      scores1: [1, 0],
      scores2: [0, 1],
      rng: createSeededRng(1),
      size: 5,
    });

    expect(new Set(selected)).toEqual(new Set([0, 1]));
  });

  test("returns nothing when the parents share no instances", () => {
    expect(
      selectMergeSubsample({
        scores1: [],
        scores2: [],
        rng: createSeededRng(1),
      }),
    ).toEqual([]);
  });
});
