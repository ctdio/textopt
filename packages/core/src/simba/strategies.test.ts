import { describe, expect, test } from "vitest";
import { createSeededRng } from "../rng.js";
import {
  buildBuckets,
  evenlySpacedIndices,
  percentile,
  samplePoisson,
  softmaxWeights,
  topKPlusBaseline,
} from "./strategies.js";

describe("percentile", () => {
  test("interpolates between the two nearest ranks", () => {
    expect(percentile([0, 1, 2, 3, 4], 10)).toBeCloseTo(0.4);
    expect(percentile([0, 1, 2, 3, 4], 90)).toBeCloseTo(3.6);
  });

  test("returns the single value when there is only one", () => {
    expect(percentile([0.5], 10)).toBe(0.5);
  });

  test("returns the same value at every percentile when all scores tie", () => {
    expect(percentile([0.25, 0.25, 0.25], 90)).toBe(0.25);
  });
});

describe("buildBuckets", () => {
  test("groups every sample of one instance into that instance's bucket", () => {
    const buckets = buildBuckets({
      batch: ["a", "b"],
      samples: [
        { programIndex: 0, scores: [0.1, 0.9] },
        { programIndex: 1, scores: [0.5, 0.9] },
      ],
    });

    const forA = buckets.find((bucket) => bucket.datum === "a");
    expect(forA?.rollouts.map((rollout) => rollout.score)).toEqual([0.5, 0.1]);
  });

  test("sorts rollouts within a bucket from best to worst", () => {
    const buckets = buildBuckets({
      batch: ["a"],
      samples: [
        { programIndex: 0, scores: [0.2] },
        { programIndex: 1, scores: [0.8] },
        { programIndex: 2, scores: [0.5] },
      ],
    });

    expect(buckets[0]?.rollouts.map((rollout) => rollout.programIndex)).toEqual(
      [1, 2, 0],
    );
  });

  test("ranks the instance with the widest disagreement first", () => {
    const buckets = buildBuckets({
      batch: ["narrow", "wide"],
      samples: [
        { programIndex: 0, scores: [0.5, 0.0] },
        { programIndex: 1, scores: [0.6, 1.0] },
      ],
    });

    expect(buckets[0]?.datum).toBe("wide");
  });

  test("breaks a gap tie on the higher best score", () => {
    const buckets = buildBuckets({
      batch: ["low", "high"],
      samples: [
        { programIndex: 0, scores: [0.0, 0.5] },
        { programIndex: 1, scores: [0.4, 0.9] },
      ],
    });

    expect(buckets[0]?.datum).toBe("high");
  });

  test("carries the output and feedback of each rollout", () => {
    const buckets = buildBuckets({
      batch: ["a"],
      samples: [
        {
          programIndex: 0,
          scores: [0.3],
          outputs: ["worse answer"],
          feedback: ["missing a term"],
        },
      ],
    });

    expect(buckets[0]?.rollouts[0]?.output).toBe("worse answer");
    expect(buckets[0]?.rollouts[0]?.feedback).toBe("missing a term");
  });
});

describe("softmaxWeights", () => {
  test("weights a higher score above a lower one", () => {
    const [low, high] = softmaxWeights([0.2, 0.8], 0.2);
    expect(high as number).toBeGreaterThan(low as number);
  });

  test("weights tied scores equally", () => {
    expect(softmaxWeights([0.5, 0.5], 0.2)).toEqual([1, 1]);
  });

  test("sharpens as the temperature falls", () => {
    const cold = softmaxWeights([0, 1], 0.1);
    const warm = softmaxWeights([0, 1], 1);
    expect((cold[1] as number) / (cold[0] as number)).toBeGreaterThan(
      (warm[1] as number) / (warm[0] as number),
    );
  });

  test("stays finite for scores far outside the unit interval", () => {
    for (const weight of softmaxWeights([0, 5000], 0.2)) {
      expect(Number.isFinite(weight)).toBe(true);
    }
  });
});

describe("topKPlusBaseline", () => {
  test("returns the highest scoring programs", () => {
    expect(topKPlusBaseline({ scores: [0.1, 0.9, 0.5, 0.7], k: 2 })).toEqual([
      1, 0,
    ]);
  });

  test("keeps the baseline in the pool even when it scores worst", () => {
    expect(topKPlusBaseline({ scores: [0.0, 0.9, 0.8], k: 2 })).toContain(0);
  });

  test("never returns a program twice", () => {
    const pool = topKPlusBaseline({ scores: [0.9, 0.8, 0.7], k: 3 });
    expect(new Set(pool).size).toBe(pool.length);
  });
});

describe("samplePoisson", () => {
  test("averages close to lambda over many draws", () => {
    const rng = createSeededRng(1);
    const draws = Array.from({ length: 4000 }, () => samplePoisson(rng, 0.75));
    const observed = draws.reduce((total, draw) => total + draw, 0) / 4000;

    expect(observed).toBeGreaterThan(0.65);
    expect(observed).toBeLessThan(0.85);
  });

  test("always draws zero at lambda zero", () => {
    const rng = createSeededRng(1);
    expect(
      Array.from({ length: 20 }, () => samplePoisson(rng, 0)).every(
        (draw) => draw === 0,
      ),
    ).toBe(true);
  });
});

describe("evenlySpacedIndices", () => {
  test("spans the first and last position", () => {
    expect(evenlySpacedIndices({ length: 5, count: 3 })).toEqual([0, 2, 4]);
  });

  test("collapses to the only position when there is one", () => {
    expect(evenlySpacedIndices({ length: 1, count: 4 })).toEqual([0]);
  });

  test("never returns a position twice", () => {
    const indices = evenlySpacedIndices({ length: 3, count: 7 });
    expect(new Set(indices).size).toBe(indices.length);
  });
});
