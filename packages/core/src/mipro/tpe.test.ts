import { describe, expect, test } from "vitest";
import { createSeededRng } from "../rng.js";
import { proposeConfiguration } from "./tpe.js";

const RNG = () => createSeededRng(7);

describe("proposeConfiguration", () => {
  test("samples uniformly while there is nothing to learn from", () => {
    const drawn = new Set<string>();
    const rng = RNG();

    for (let draw = 0; draw < 50; draw += 1) {
      drawn.add(
        proposeConfiguration({
          observations: [],
          menuSizes: [3, 3],
          rng,
        }).join(","),
      );
    }

    // Uniform sampling over 9 configurations reaches most of them in 50 draws.
    expect(drawn.size).toBeGreaterThan(5);
  });

  test("concentrates on the choice every good observation shares", () => {
    const observations = [
      { choices: [2, 0], score: 0.9 },
      { choices: [2, 1], score: 0.85 },
      { choices: [2, 2], score: 0.8 },
      { choices: [0, 0], score: 0.1 },
      { choices: [1, 1], score: 0.05 },
      { choices: [0, 2], score: 0.0 },
    ];
    const rng = RNG();

    const picks = Array.from(
      { length: 20 },
      () =>
        proposeConfiguration({
          observations,
          menuSizes: [3, 3],
          gamma: 0.5,
          // Enough draws that the argmax reflects the density rather than
          // which corners of it a short sample happened to reach.
          samples: 100,
          startupTrials: 1,
          rng,
        })[0],
    );

    expect(picks.every((pick) => pick === 2)).toBe(true);
  });

  test("narrows the good set as observations accumulate", () => {
    // Thirty observations. The three strongest all take option 0; the seven
    // behind them all take option 1. A tenth of thirty is three, so the good
    // set is exactly the leading group and nothing else — a flat quarter would
    // pull in the seven runners-up and let them outvote the leaders.
    const observations = [
      { choices: [0, 0], score: 0.99 },
      { choices: [0, 1], score: 0.98 },
      { choices: [0, 2], score: 0.97 },
      ...Array.from({ length: 7 }, (_, index) => ({
        choices: [1, index % 3],
        score: 0.96 - index / 100,
      })),
      ...Array.from({ length: 20 }, (_, index) => ({
        choices: [2, index % 3],
        score: 0.5 - index / 100,
      })),
    ];

    const rng = RNG();
    const picks = Array.from(
      { length: 20 },
      () =>
        proposeConfiguration({
          observations,
          menuSizes: [3, 3],
          samples: 100,
          rng,
        })[0],
    );

    expect(picks.every((pick) => pick === 0)).toBe(true);
  });

  test("commits on a decided component while another stays open", () => {
    const observations = [
      { choices: [0, 1], score: 0.9 },
      { choices: [1, 1], score: 0.9 },
      { choices: [2, 1], score: 0.85 },
      { choices: [0, 0], score: 0.1 },
      { choices: [1, 2], score: 0.1 },
      { choices: [2, 0], score: 0.0 },
    ];

    const choices = proposeConfiguration({
      observations,
      menuSizes: [3, 3],
      gamma: 0.5,
      samples: 100,
      startupTrials: 1,
      rng: RNG(),
    });

    // Every good observation shares the second component and disagrees on the
    // first, so only the second should come out settled.
    expect(choices[1]).toBe(1);
  });

  test("learns which choices work together, not just which are common", () => {
    // The marginals here are deliberately uninformative: every option appears
    // in exactly one good and one bad observation, so per-component histograms
    // are flat and identical. Only the pairing carries signal.
    const observations = [
      { choices: [0, 0], score: 0.9 },
      { choices: [1, 1], score: 0.9 },
      { choices: [0, 0], score: 0.8 },
      { choices: [1, 1], score: 0.8 },
      { choices: [0, 1], score: 0.1 },
      { choices: [1, 0], score: 0.1 },
      { choices: [0, 1], score: 0.0 },
      { choices: [1, 0], score: 0.0 },
    ];

    const rng = RNG();
    const drawn = Array.from({ length: 20 }, () =>
      proposeConfiguration({
        observations,
        menuSizes: [2, 2],
        gamma: 0.5,
        startupTrials: 1,
        rng,
      }).join(","),
    );

    for (const configuration of drawn) {
      expect(["0,0", "1,1"]).toContain(configuration);
    }
  });

  test("allows a configuration back once it has been tried", () => {
    const observations = [
      { choices: [0, 0], score: 0.9 },
      { choices: [0, 1], score: 0.1 },
      { choices: [0, 0], score: 0.8 },
      { choices: [0, 1], score: 0.2 },
      { choices: [0, 0], score: 0.85 },
    ];

    // Every configuration of a 1x2 space is taken, and the strong one stays
    // the right answer: a repeat is a second reading, not a wasted trial.
    const choices = proposeConfiguration({
      observations,
      menuSizes: [1, 2],
      gamma: 0.5,
      rng: RNG(),
    });

    expect(choices).toEqual([0, 0]);
  });

  test("keeps an option nobody has tried reachable", () => {
    const observations = Array.from({ length: 8 }, (_, index) => ({
      choices: [index % 2],
      score: index / 8,
    }));

    const drawn = new Set<number>();
    const rng = RNG();
    for (let draw = 0; draw < 200; draw += 1) {
      drawn.add(
        proposeConfiguration({
          observations,
          menuSizes: [3],
          samples: 1,
          startupTrials: 1,
          rng,
        })[0] as number,
      );
    }

    // Smoothing is what keeps index 2 in play despite never being observed.
    expect(drawn.has(2)).toBe(true);
  });

  test("is reproducible for a given seed", () => {
    const observations = [
      { choices: [2, 0], score: 0.9 },
      { choices: [1, 1], score: 0.5 },
      { choices: [0, 2], score: 0.1 },
      { choices: [0, 1], score: 0.2 },
      { choices: [1, 2], score: 0.3 },
    ];
    const draw = () =>
      proposeConfiguration({ observations, menuSizes: [3, 3], rng: RNG() });

    expect(draw()).toEqual(draw());
  });

  test("keeps exploring while every observation ties", () => {
    // Eight of the nine configurations, all scoring the same. Nothing here
    // ranks one above another, so a split into good and bad is arbitrary —
    // and acting on an arbitrary split burns the budget re-proposing what has
    // already been measured instead of reaching what has not.
    const observations = [];
    for (let alpha = 0; alpha < 3; alpha += 1) {
      for (let beta = 0; beta < 3; beta += 1) {
        if (alpha !== 1 || beta !== 1) {
          observations.push({ choices: [alpha, beta], score: 0 });
        }
      }
    }

    const rng = RNG();
    const drawn = new Set<string>();
    for (let draw = 0; draw < 50; draw += 1) {
      drawn.add(
        proposeConfiguration({
          observations,
          menuSizes: [3, 3],
          startupTrials: 1,
          rng,
        }).join(","),
      );
    }

    expect(drawn.has("1,1")).toBe(true);
  });

  test("refuses a component with an empty menu", () => {
    expect(() =>
      proposeConfiguration({
        observations: [],
        menuSizes: [3, 0],
        rng: RNG(),
      }),
    ).toThrow(/menu/);
  });
});
