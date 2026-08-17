import { describe, expect, test } from "vitest";
import { createSeededRng } from "./rng.js";
import {
  createEpochShuffledSampler,
  currentBestSelector,
  epsilonGreedySelector,
  improvementAcceptance,
  paretoSelector,
  roundRobinComponentSelector,
  topKParetoSelector,
} from "./strategies.js";

const STATE = {
  scoreMatrix: [
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 0],
  ],
  aggregateScores: [1 / 3, 2 / 3, 0],
};

describe("paretoSelector", () => {
  test("only returns candidates on the instance frontier", () => {
    const rng = createSeededRng(4);

    for (let i = 0; i < 100; i += 1) {
      expect([0, 1]).toContain(paretoSelector()({ state: STATE, rng }));
    }
  });

  test("can return a low aggregate scorer that uniquely wins an instance", () => {
    const rng = createSeededRng(4);
    const picks = new Set<number>();

    for (let i = 0; i < 100; i += 1) {
      picks.add(paretoSelector()({ state: STATE, rng }));
    }

    expect(picks.has(0)).toBe(true);
  });
});

describe("currentBestSelector", () => {
  test("returns the highest aggregate scorer", () => {
    const rng = createSeededRng(1);

    expect(currentBestSelector()({ state: STATE, rng })).toBe(1);
  });
});

describe("epsilonGreedySelector", () => {
  test("always exploits when epsilon is zero", () => {
    const rng = createSeededRng(1);

    for (let i = 0; i < 50; i += 1) {
      expect(epsilonGreedySelector({ epsilon: 0 })({ state: STATE, rng })).toBe(
        1,
      );
    }
  });

  test("explores every candidate when epsilon is one", () => {
    const rng = createSeededRng(8);
    const picks = new Set<number>();

    for (let i = 0; i < 200; i += 1) {
      picks.add(epsilonGreedySelector({ epsilon: 1 })({ state: STATE, rng }));
    }

    expect(picks).toEqual(new Set([0, 1, 2]));
  });
});

describe("topKParetoSelector", () => {
  test("restricts selection to the top k aggregate scorers", () => {
    const rng = createSeededRng(2);

    for (let i = 0; i < 100; i += 1) {
      expect(topKParetoSelector({ k: 1 })({ state: STATE, rng })).toBe(1);
    }
  });
});

describe("roundRobinComponentSelector", () => {
  test("cycles through components as the cursor advances", () => {
    const select = roundRobinComponentSelector();
    const candidate = { alpha: "a", beta: "b" };
    const rng = createSeededRng(1);

    const picks = [0, 1, 2, 3].map((cursor) =>
      select({ candidate, cursor, iteration: 0, rng }),
    );

    expect(picks).toEqual([["alpha"], ["beta"], ["alpha"], ["beta"]]);
  });

  test("ignores the global iteration counter", () => {
    const select = roundRobinComponentSelector();
    const candidate = { alpha: "a", beta: "b" };
    const rng = createSeededRng(1);

    // A candidate reselected only on even iterations must still advance
    // through its own components rather than being pinned to one of them.
    const picks = [0, 2, 4, 6].map((iteration, cursor) =>
      select({ candidate, cursor, iteration, rng }),
    );

    expect(picks).toEqual([["alpha"], ["beta"], ["alpha"], ["beta"]]);
  });
});

describe("createEpochShuffledSampler", () => {
  test("returns exactly minibatchSize indices", () => {
    const sample = createEpochShuffledSampler({ minibatchSize: 3 });
    const trainset = [1, 2, 3, 4, 5, 6];

    expect(
      sample({ trainset, iteration: 0, rng: createSeededRng(1) }),
    ).toHaveLength(3);
  });

  test("covers the whole trainset over one epoch", () => {
    const sample = createEpochShuffledSampler({ minibatchSize: 3 });
    const trainset = [1, 2, 3, 4, 5, 6];
    const rng = createSeededRng(1);

    const seen = new Set<number>([
      ...sample({ trainset, iteration: 0, rng }),
      ...sample({ trainset, iteration: 1, rng }),
    ]);

    expect(seen).toEqual(new Set([0, 1, 2, 3, 4, 5]));
  });

  test("pads a trainset smaller than the minibatch", () => {
    const sample = createEpochShuffledSampler({ minibatchSize: 4 });
    const trainset = [1, 2, 3];

    const indices = sample({
      trainset,
      iteration: 0,
      rng: createSeededRng(1),
    });

    expect(indices).toHaveLength(4);
    expect(new Set(indices)).toEqual(new Set([0, 1, 2]));
  });

  test("is deterministic for a given seed", () => {
    const first = createEpochShuffledSampler({ minibatchSize: 2 })({
      trainset: [1, 2, 3, 4],
      iteration: 0,
      rng: createSeededRng(12),
    });
    const second = createEpochShuffledSampler({ minibatchSize: 2 })({
      trainset: [1, 2, 3, 4],
      iteration: 0,
      rng: createSeededRng(12),
    });

    expect(first).toEqual(second);
  });

  test("throws on an empty trainset", () => {
    const sample = createEpochShuffledSampler({ minibatchSize: 2 });

    expect(() =>
      sample({ trainset: [], iteration: 0, rng: createSeededRng(1) }),
    ).toThrow(/empty/i);
  });
});

describe("improvementAcceptance", () => {
  test("accepts a child that scores higher on the minibatch", () => {
    expect(
      improvementAcceptance()({
        parentScores: [0.5, 0.5],
        childScores: [0.6, 0.5],
      }),
    ).toBe(true);
  });

  test("rejects a child that ties the parent", () => {
    expect(
      improvementAcceptance()({
        parentScores: [0.5, 0.5],
        childScores: [0.5, 0.5],
      }),
    ).toBe(false);
  });

  test("rejects an improvement below the minimum threshold", () => {
    expect(
      improvementAcceptance({ minImprovement: 0.5 })({
        parentScores: [0.5],
        childScores: [0.6],
      }),
    ).toBe(false);
  });
});
