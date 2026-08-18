import { describe, expect, test } from "vitest";
import { createSeededRng } from "./rng.js";
import { createEpochShuffledSampler } from "./sampling.js";

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
