import { describe, expect, test } from "vitest";
import { createSeededRng } from "../rng.js";
import {
  currentBestSelector,
  lowerBoundEvaluationPolicy,
  pairedPermutationAcceptance,
  epsilonGreedySelector,
  fullEvaluationPolicy,
  improvementAcceptance,
  paretoSelector,
  roundRobinComponentSelector,
  subsampledEvaluationPolicy,
  topKParetoSelector,
} from "./strategies.js";
import type { CandidateRecord } from "./types.js";

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

  test("selects on objectives rather than instances when asked", () => {
    const rng = createSeededRng(4);
    const state = {
      ...STATE,
      // Candidate 2 wins no instance, so instance selection can never reach it.
      objectiveScores: [
        { accuracy: 0.2, brevity: 0.1 },
        { accuracy: 0.3, brevity: 0.2 },
        { accuracy: 1, brevity: 1 },
      ],
    };
    const picks = new Set<number>();

    for (let i = 0; i < 100; i += 1) {
      picks.add(paretoSelector({ frontier: "objective" })({ state, rng }));
    }

    expect(picks).toEqual(new Set([2]));
  });

  test("keeps instance winners alongside objective leaders under the hybrid frontier", () => {
    const rng = createSeededRng(4);
    const state = {
      ...STATE,
      objectiveScores: [{ accuracy: 0.2 }, { accuracy: 0.3 }, { accuracy: 1 }],
    };
    const picks = new Set<number>();

    for (let i = 0; i < 100; i += 1) {
      picks.add(paretoSelector({ frontier: "hybrid" })({ state, rng }));
    }

    expect(picks).toEqual(new Set([0, 1, 2]));
  });

  test("rejects an objective frontier when no candidate has objective scores", () => {
    const rng = createSeededRng(4);

    expect(() =>
      paretoSelector({ frontier: "objective" })({ state: STATE, rng }),
    ).toThrow(/objective/i);
  });
});

describe("fullEvaluationPolicy", () => {
  const policy = fullEvaluationPolicy();

  test("selects every validation instance", () => {
    expect(
      policy.selectInstances({
        validationSet: ["a", "b", "c"],
        candidate: { instruction: "x" },
        records: [],
        iteration: 0,
        rng: createSeededRng(1),
      }),
    ).toEqual([0, 1, 2]);
  });

  test("returns the highest mean over the instances actually scored", () => {
    expect(
      policy.bestCandidate([
        buildRecord({ id: 0, instanceScores: [1, 0] }),
        buildRecord({ id: 1, instanceScores: [0.75, 0.75] }),
      ]),
    ).toBe(1);
  });

  test("breaks a tie in favour of the wider coverage", () => {
    expect(
      policy.bestCandidate([
        buildRecord({ id: 0, instanceScores: [1, undefined] }),
        buildRecord({ id: 1, instanceScores: [1, 1] }),
      ]),
    ).toBe(1);
  });
});

describe("subsampledEvaluationPolicy", () => {
  test("selects no more instances than the requested size", () => {
    const selected = subsampledEvaluationPolicy({ size: 2 }).selectInstances({
      validationSet: ["a", "b", "c", "d"],
      candidate: { instruction: "x" },
      records: [],
      iteration: 0,
      rng: createSeededRng(3),
    });

    expect(selected).toHaveLength(2);
    expect(new Set(selected).size).toBe(2);
  });

  test("selects every instance when the validationSet is smaller than the size", () => {
    const selected = subsampledEvaluationPolicy({ size: 5 }).selectInstances({
      validationSet: ["a", "b"],
      candidate: { instruction: "x" },
      records: [],
      iteration: 0,
      rng: createSeededRng(3),
    });

    expect(selected.sort()).toEqual([0, 1]);
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

function buildRecord(args: {
  id: number;
  instanceScores: (number | undefined)[];
}): CandidateRecord {
  const scored = args.instanceScores.filter(
    (score): score is number => score !== undefined,
  );

  return {
    id: args.id,
    candidate: { instruction: `candidate ${args.id}` },
    parentIds: [],
    instanceScores: args.instanceScores,
    aggregateScore:
      scored.length === 0
        ? 0
        : scored.reduce((total, score) => total + score, 0) / scored.length,
    source: "mutation",
    updatedComponents: [],
    iteration: 0,
    componentCursor: 0,
  };
}

describe("pairedPermutationAcceptance", () => {
  test("accepts a child that improved on every instance of the batch", () => {
    const accepted = pairedPermutationAcceptance()({
      parentScores: [0, 0, 0],
      childScores: [1, 1, 1],
    });

    expect(accepted).toBe(true);
  });

  test("rejects a child carried by one instance out of four", () => {
    // Sum improvement accepts this: +1 on one instance beats three ties. One
    // rollout is not evidence, and promoting on it is how a run fills its
    // pool with candidates the validation sweep then throws away.
    const scores = { parentScores: [1, 1, 1, 0], childScores: [1, 1, 1, 1] };

    expect(improvementAcceptance()(scores)).toBe(true);
    expect(pairedPermutationAcceptance()(scores)).toBe(false);
  });

  test("rejects a child that lost ground overall", () => {
    const accepted = pairedPermutationAcceptance()({
      parentScores: [1, 1, 1],
      childScores: [1, 0, 1],
    });

    expect(accepted).toBe(false);
  });

  test("rejects a batch in which nothing moved", () => {
    const accepted = pairedPermutationAcceptance()({
      parentScores: [1, 1, 1],
      childScores: [1, 1, 1],
    });

    expect(accepted).toBe(false);
  });

  test("accepts a consistent gain over a batch too large to enumerate", () => {
    // Past `maxExact` the p-value comes from a normal approximation rather
    // than every sign flip, and has to reach the same verdict.
    const parentScores = Array.from({ length: 24 }, () => 0.5);
    const childScores = Array.from({ length: 24 }, () => 0.6);

    expect(
      pairedPermutationAcceptance({ maxExact: 12 })({
        parentScores,
        childScores,
      }),
    ).toBe(true);
  });
});

describe("lowerBoundEvaluationPolicy", () => {
  test("prefers the steadier of two candidates swept over the same instances", () => {
    // The spiky candidate has the higher mean, carried by four instances out
    // of six. That is the shape of a candidate fitted to the validation set
    // rather than to the task, and it is what the run then reports.
    const records = [
      buildRecord({
        id: 0,
        instanceScores: [0.55, 0.55, 0.55, 0.55, 0.55, 0.55],
      }),
      buildRecord({ id: 1, instanceScores: [1, 1, 1, 1, 0, 0.2] }),
    ];

    expect(fullEvaluationPolicy().bestCandidate(records)).toBe(1);
    expect(lowerBoundEvaluationPolicy({ z: 2 }).bestCandidate(records)).toBe(0);
  });

  test("returns the higher mean when both candidates are equally steady", () => {
    const records = [
      buildRecord({ id: 0, instanceScores: [0.5, 0.5, 0.5, 0.5] }),
      buildRecord({ id: 1, instanceScores: [0.6, 0.6, 0.6, 0.6] }),
    ];

    expect(lowerBoundEvaluationPolicy().bestCandidate(records)).toBe(1);
  });

  test("selects every validation instance, as a full sweep does", () => {
    const selected = lowerBoundEvaluationPolicy().selectInstances({
      validationSet: [1, 2, 3],
      candidate: { instruction: "text" },
      records: [],
      iteration: 0,
      rng: createSeededRng(1),
    });

    expect(selected).toEqual([0, 1, 2]);
  });
});
