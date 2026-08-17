import { describe, expect, test } from "vitest";
import {
  buildInstanceFronts,
  buildObjectiveFronts,
  computeInstanceBests,
  objectiveBests,
  pruneDominatedFronts,
  selectParetoCandidate,
} from "./pareto.js";
import { createSeededRng } from "./rng.js";

describe("computeInstanceBests", () => {
  test("returns the best score achieved on each instance", () => {
    const scoreMatrix = [
      [1, 0, 0.5],
      [0, 1, 0.25],
    ];

    expect(computeInstanceBests(scoreMatrix)).toEqual([1, 1, 0.5]);
  });

  test("returns an empty array when there are no candidates", () => {
    expect(computeInstanceBests([])).toEqual([]);
  });
});

describe("buildInstanceFronts", () => {
  test("assigns the sole best candidate to each instance front", () => {
    const fronts = buildInstanceFronts({
      scoreMatrix: [
        [1, 0],
        [0, 1],
      ],
    });

    expect(fronts).toEqual([new Set([0]), new Set([1])]);
  });

  test("includes every candidate tied at the instance best", () => {
    const fronts = buildInstanceFronts({
      scoreMatrix: [
        [1, 0],
        [1, 0],
        [0, 1],
      ],
    });

    expect(fronts[0]).toEqual(new Set([0, 1]));
    expect(fronts[1]).toEqual(new Set([2]));
  });

  test("treats scores within epsilon of the best as tied", () => {
    const fronts = buildInstanceFronts({
      scoreMatrix: [[1], [0.99]],
      epsilon: 0.05,
    });

    expect(fronts[0]).toEqual(new Set([0, 1]));
  });
});

describe("buildObjectiveFronts", () => {
  test("assigns the best candidate to each objective", () => {
    const fronts = buildObjectiveFronts({
      objectiveScores: [
        { accuracy: 1, brevity: 0 },
        { accuracy: 0, brevity: 1 },
      ],
    });

    expect(fronts).toEqual([new Set([0]), new Set([1])]);
  });

  test("includes every candidate tied on an objective", () => {
    const fronts = buildObjectiveFronts({
      objectiveScores: [{ accuracy: 1 }, { accuracy: 1 }, { accuracy: 0.5 }],
    });

    expect(fronts).toEqual([new Set([0, 1])]);
  });

  test("ignores candidates that were never scored on an objective", () => {
    const fronts = buildObjectiveFronts({
      objectiveScores: [{ accuracy: 0.5 }, undefined, { brevity: 1 }],
    });

    expect(fronts).toEqual([new Set([0]), new Set([2])]);
  });

  test("treats scores within epsilon of the best as tied", () => {
    const fronts = buildObjectiveFronts({
      objectiveScores: [{ accuracy: 1 }, { accuracy: 0.99 }],
      epsilon: 0.05,
    });

    expect(fronts).toEqual([new Set([0, 1])]);
  });

  test("returns no fronts when no candidate has objective scores", () => {
    expect(buildObjectiveFronts({ objectiveScores: [undefined] })).toEqual([]);
  });
});

describe("objectiveBests", () => {
  test("returns the best score reached on each objective", () => {
    const bests = objectiveBests([
      { accuracy: 0.5, brevity: 1 },
      { accuracy: 0.75 },
    ]);

    expect(bests).toEqual({ accuracy: 0.75, brevity: 1 });
  });
});

describe("pruneDominatedFronts", () => {
  test("removes a candidate whose winning instances are covered by another", () => {
    // Candidate 0 is best only on instance 0. Candidate 1 is best on instances
    // 0 and 1, so candidate 0 contributes nothing unique.
    const fronts = [new Set([0, 1]), new Set([1])];

    const pruned = pruneDominatedFronts({
      fronts,
      aggregateScores: [0.5, 0.9],
    });

    expect(pruned).toEqual([new Set([1]), new Set([1])]);
  });

  test("keeps candidates that uniquely win at least one instance", () => {
    const fronts = [new Set([0]), new Set([1])];

    const pruned = pruneDominatedFronts({
      fronts,
      aggregateScores: [0.5, 0.9],
    });

    expect(pruned).toEqual([new Set([0]), new Set([1])]);
  });

  test("never empties an instance front", () => {
    const fronts = [new Set([0, 1, 2]), new Set([0, 1]), new Set([2])];

    const pruned = pruneDominatedFronts({
      fronts,
      aggregateScores: [0.1, 0.2, 0.3],
    });

    for (const front of pruned) {
      expect(front.size).toBeGreaterThan(0);
    }
  });

  test("breaks ties in favour of the higher aggregate score", () => {
    // Both candidates win exactly instance 0; only one should survive.
    const fronts = [new Set([0, 1])];

    const pruned = pruneDominatedFronts({
      fronts,
      aggregateScores: [0.2, 0.8],
    });

    expect(pruned[0]).toEqual(new Set([1]));
  });

  test("removes a chain of candidates each covered by the next", () => {
    // 0 wins only where 1 wins, 1 only where 2 wins, and 2 wins everywhere.
    const fronts = [new Set([0, 1, 2]), new Set([1, 2]), new Set([2])];

    const pruned = pruneDominatedFronts({
      fronts,
      aggregateScores: [0.1, 0.2, 0.3],
    });

    expect(pruned).toEqual([new Set([2]), new Set([2]), new Set([2])]);
  });

  test("prunes a large pool without rescanning it once per removal", () => {
    // The adversarial shape: survivors sort first, so every later removal
    // re-checks the whole surviving prefix. 400 candidates that each uniquely
    // win one instance, then 400 tied on shared instances of which only the
    // strongest survives.
    const unique = 400;
    const tied = 400;
    const tiedIds = Array.from({ length: tied }, (_, i) => unique + i);
    const fronts = [
      ...Array.from({ length: unique }, (_, id) => new Set([id])),
      ...Array.from({ length: tied }, () => new Set(tiedIds)),
    ];
    const aggregateScores = Array.from({ length: unique + tied }, (_, id) =>
      id < unique ? 0.1 : 0.5 + id / 10_000,
    );

    const pruned = pruneDominatedFronts({ fronts, aggregateScores });

    expect(pruned.slice(0, unique)).toEqual(
      Array.from({ length: unique }, (_, id) => new Set([id])),
    );
    for (const front of pruned.slice(unique)) {
      expect(front).toEqual(new Set([unique + tied - 1]));
    }
  });
});

describe("selectParetoCandidate", () => {
  test("samples proportionally to the number of instances a candidate wins", () => {
    // Candidate 0 wins three instances, candidate 1 wins one.
    const fronts = [new Set([0]), new Set([0]), new Set([0]), new Set([1])];
    const rng = createSeededRng(17);

    const counts = new Map<number, number>();
    for (let i = 0; i < 4000; i += 1) {
      const index = selectParetoCandidate({
        fronts,
        aggregateScores: [0.5, 0.5],
        rng,
      });
      counts.set(index, (counts.get(index) ?? 0) + 1);
    }

    const ratio = (counts.get(0) ?? 0) / (counts.get(1) ?? 1);
    expect(ratio).toBeGreaterThan(2.4);
    expect(ratio).toBeLessThan(3.6);
  });

  test("never selects a dominated candidate", () => {
    const fronts = [new Set([0, 1]), new Set([1])];
    const rng = createSeededRng(23);

    for (let i = 0; i < 200; i += 1) {
      expect(
        selectParetoCandidate({ fronts, aggregateScores: [0.5, 0.9], rng }),
      ).toBe(1);
    }
  });

  test("is deterministic for a given seed", () => {
    const fronts = [new Set([0]), new Set([1]), new Set([0, 2])];
    const aggregateScores = [0.4, 0.6, 0.5];

    const first = Array.from({ length: 20 }, () =>
      selectParetoCandidate({
        fronts,
        aggregateScores,
        rng: createSeededRng(99),
      }),
    );
    const second = Array.from({ length: 20 }, () =>
      selectParetoCandidate({
        fronts,
        aggregateScores,
        rng: createSeededRng(99),
      }),
    );

    expect(first).toEqual(second);
  });
});
