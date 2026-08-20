import { afterEach, describe, expect, test, vi } from "vitest";
import type { Optimizer, OptimizerResult } from "../optimizer.js";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "../testing.js";
import type { Adapter, TextModel } from "../types.js";
import { MiproOptimizer } from "./optimize.js";
import type { MiproSnapshot, MiproStopReason } from "./optimize.js";

/** An instance that names the split it belongs to, so an adapter can tell them apart. */
interface Split {
  id: number;
  kind: "train" | "validate";
}

/** The terms each instance needs, split across two components. */
const JOINT_OPTIONS = {
  alpha: ["hold ten seconds", "sprocket", "widget"],
  beta: ["ticket portal thirty days billing prorated", "lorem", "ipsum"],
};

function baseAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const keyword = createKeywordAdapter();
  return { evaluate: (args) => keyword.evaluate(args) };
}

/** The base adapter, reporting ten input tokens for every rollout it runs. */
function pricedAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const keyword = createKeywordAdapter();
  return {
    evaluate: (args) => ({
      ...keyword.evaluate(args),
      usage: args.batch.map(() => ({ inputTokens: 10, outputTokens: 5 })),
    }),
  };
}

/**
 * "spiky" scores 0.5 over the whole validation set but reads a perfect 1.0 whenever a
 * minibatch lands on instances 0-1. "steady" is worth 0.75 on every instance.
 * Screening on a single lucky reading therefore rates the worse candidate
 * higher, and a bar that only ever rises locks the better one out for good.
 */
function noisyAdapter(): Adapter<{ id: number }, unknown, string> {
  return {
    evaluate: ({ batch, candidate }) => ({
      outputs: batch.map(() => candidate.alpha),
      scores: batch.map((datum) => {
        if (candidate.alpha === "spiky") {
          return datum.id < 2 ? 1 : 0;
        }
        return candidate.alpha === "steady" ? 0.75 : 0;
      }),
    }),
  };
}

/** Half the trainingSet scores perfectly, so bootstrapping has something to keep. */
function demoAdapter(): Adapter<
  { id: number; good: boolean },
  unknown,
  string
> {
  return {
    evaluate: ({ batch, candidate }) => ({
      outputs: batch.map((datum) => `answer ${datum.id}`),
      scores: batch.map((datum) =>
        datum.good && candidate.instruction !== "" ? 1 : 0,
      ),
    }),
  };
}

const DEMO_TRAINSET = Array.from({ length: 8 }, (_, id) => ({
  id,
  good: id % 2 === 0,
}));

const UNUSED_REFLECT: TextModel = async () => "```\nunused\n```";

/**
 * Scores only when both components name the same format and it is the one the
 * instance needs. Non-separable by construction: the best choice for either
 * component depends entirely on the other.
 */
function pairingAdapter(): Adapter<{ format: string }, unknown, string> {
  return {
    evaluate: ({ batch, candidate }) => {
      const agreed =
        candidate.alpha === candidate.beta ? candidate.alpha : "mismatch";

      return {
        outputs: batch.map(() => agreed),
        scores: batch.map((datum) => (agreed === datum.format ? 1 : 0)),
      };
    },
  };
}

function jointTask() {
  return {
    seedCandidate: { alpha: "", beta: "" },
    trainingSet: KEYWORD_EXAMPLES,
    adapter: baseAdapter(),
    reflect: UNUSED_REFLECT,
    componentOptions: JOINT_OPTIONS,
    maxMetricCalls: 2000,
  };
}

describe("MiproOptimizer", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("stops once the wall clock deadline passes", async () => {
    // Rollout and cost ceilings bound what a run spends, not how long it
    // takes: a run stuck behind a rate limit costs nothing and runs forever.
    vi.useFakeTimers();

    const result = await new MiproOptimizer({
      maxTrials: 20,
      minibatchSize: 2,
    }).optimize({
      ...jointTask(),
      adapter: {
        evaluate: ({ batch }) => {
          vi.advanceTimersByTime(400);
          return {
            outputs: batch.map(() => ""),
            scores: batch.map(() => 0.5),
          };
        },
      },
      maxWallClockMs: 1000,
    });

    expect(result.stopReason).toBe("deadlineReached");
  });
  test("stops once the reported cost reaches the ceiling", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 8,
      minibatchSize: 2,
    }).optimize({
      ...jointTask(),
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 0.5),
          usage: batch.map(() => ({ costUsd: 1 })),
        }),
      },
      maxCostUsd: 4,
    });

    expect(result.stopReason).toBe("costExhausted");
    expect(result.usage.costUsd).toBe(4);
  });

  test("leaves a transiently failed rollout out of the score it reports", async () => {
    // A rate limit measured the provider, not the candidate. Averaging its
    // zero in reports a score no rollout ever produced, and the surrogate then
    // fits a configuration to it.
    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      ...jointTask(),
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map((_, index) => (index === 2 ? 0 : 1)),
          transient: batch.map((_, index) => index === 2),
        }),
      },
      retry: { attempts: 0 },
    });

    expect(result.seedScore).toBe(1);
  });

  test("satisfies the Optimizer contract", async () => {
    const mipro = new MiproOptimizer({ maxTrials: 4, minibatchSize: 2 });
    const contract: Optimizer<MiproStopReason> = mipro;

    const result = await mipro.optimize(jointTask());
    const outcome: OptimizerResult<"alpha" | "beta", MiproStopReason> = result;

    expect(contract).toBe(mipro);
    expect(outcome.bestScore).toBeGreaterThanOrEqual(0);
  });

  test("uses caller-supplied options without a reflection call", async () => {
    let reflectionCalls = 0;

    const result = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
    }).optimize({
      ...jointTask(),
      reflect: async () => {
        reflectionCalls += 1;
        return "```\ngenerated\n```";
      },
    });

    expect(reflectionCalls).toBe(0);
    expect(result.reflectionCalls).toBe(0);
    // Seed text plus the three supplied options, per component.
    expect(result.menu.alpha).toHaveLength(4);
    expect(result.menu.beta).toHaveLength(4);
  });

  test("puts the seed text at the head of every menu", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
    }).optimize({
      ...jointTask(),
      seedCandidate: { alpha: "seed alpha", beta: "seed beta" },
    });

    expect(result.menu.alpha[0]).toBe("seed alpha");
    expect(result.menu.beta[0]).toBe("seed beta");
  });

  test("finds the best pairing over the joint space", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 30,
      minibatchSize: 4,
      seed: 3,
    }).optimize(jointTask());

    expect(result.bestCandidate.alpha).toBe(JOINT_OPTIONS.alpha[0]);
    expect(result.bestCandidate.beta).toBe(JOINT_OPTIONS.beta[0]);
    expect(result.bestScore).toBe(1);
  });

  test("solves a task no single-component move can improve on", async () => {
    // Both components must name the same format, and it must be the one the
    // data needs. From the seed, every single-component move scores 0 — only
    // the joint move does anything at all, which is precisely what a search
    // that updates one component at a time and screens it in isolation
    // cannot see.
    const result = await new MiproOptimizer({
      maxTrials: 40,
      minibatchSize: 2,
      seed: 3,
    }).optimize({
      seedCandidate: { alpha: "", beta: "" },
      trainingSet: Array.from({ length: 4 }, () => ({ format: "json" })),
      adapter: pairingAdapter(),
      reflect: UNUSED_REFLECT,
      componentOptions: {
        alpha: ["json", "xml"],
        beta: ["json", "xml"],
      },
      maxMetricCalls: 2000,
    });

    expect(result.seedScore).toBe(0);
    expect(result.bestCandidate).toEqual({ alpha: "json", beta: "json" });
    expect(result.bestScore).toBe(1);
  });

  test("full-evaluates the best configuration despite an early lucky minibatch", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 20,
      minibatchSize: 2,
      seed: 5,
    }).optimize({
      seedCandidate: { alpha: "" },
      trainingSet: [{ id: 0 }, { id: 1 }, { id: 2 }, { id: 3 }],
      adapter: noisyAdapter(),
      reflect: UNUSED_REFLECT,
      componentOptions: { alpha: ["spiky", "steady"] },
      maxMetricCalls: 4000,
    });

    expect(result.bestCandidate.alpha).toBe("steady");
    expect(result.bestScore).toBe(0.75);
  });

  test("bootstraps a demo menu from rollouts the metric rewarded", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 3,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: demoAdapter(),
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 400,
    });

    const blocks = result.menu.demos.filter((entry) =>
      entry.includes("<demo>"),
    );
    expect(blocks.length).toBeGreaterThan(0);
    // Harvested, never authored: the reflection model writes the instruction
    // and the dataset summary, and never a demo.
    expect(result.reflectionCalls).toBe(2);
  });

  test("checkpoints a trial whose rollouts all failed transiently", async () => {
    // The trial is spent either way: the rollouts were bought and the counter
    // moved. A snapshot skipped because the provider was down leaves a resumed
    // run repeating trials it already paid for.
    const TRAIN: Split[] = Array.from({ length: 8 }, (_, id) => ({
      id,
      kind: "train",
    }));
    const VALIDATE: Split[] = Array.from({ length: 4 }, (_, id) => ({
      id,
      kind: "validate",
    }));
    const trials: number[] = [];

    await new MiproOptimizer({
      maxTrials: 3,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      seedCandidate: { alpha: "", beta: "" },
      trainingSet: TRAIN,
      validationSet: VALIDATE,
      componentOptions: JOINT_OPTIONS,
      reflect: UNUSED_REFLECT,
      maxMetricCalls: 2000,
      retry: { attempts: 0 },
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 0.5),
          transient: batch.map((datum) => datum.kind === "train"),
        }),
      },
      onCheckpoint: (snapshot) => {
        trials.push(snapshot.trial);
      },
    });

    expect(trials).toEqual([0, 1, 2, 3]);
  });

  test("checkpoints a trial only once its cadence sweep has run", async () => {
    // A snapshot names a trial, and a resumed run schedules the next sweep an
    // interval past the trial the snapshot names. Taking it before that trial's
    // sweep describes half a trial, and the resume skips the sweep entirely.
    const swept: number[] = [];

    await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      fullEvalInterval: 1,
      seed: 1,
    }).optimize({
      ...jointTask(),
      onCheckpoint: (snapshot) => {
        swept.push(snapshot.fullEvaluations);
      },
    });

    // The seed sweep, then one per trial, each already recorded by the
    // snapshot that names its trial.
    expect(swept).toEqual([1, 2, 3]);
  });

  test("counts the tokens harvesting spent in the run's usage", async () => {
    // Harvesting runs the seed over the training set on its own evaluator,
    // which for a demo-heavy run is most of what the run spends. Usage that
    // omits it makes `maxCostUsd` a ceiling on part of the run.
    const result = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 2,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map((datum) => `answer ${datum.id}`),
          scores: batch.map((datum) =>
            datum.good && candidate.instruction !== "" ? 1 : 0,
          ),
          usage: batch.map(() => ({ inputTokens: 10, outputTokens: 5 })),
        }),
      },
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 400,
    });

    // One rollout is ten input tokens, so usage must cover every rollout the
    // run paid for, harvesting included.
    expect(result.usage.inputTokens).toBe(result.metricCalls * 10);
  });

  test("stops bootstrapping demo sets once the cost ceiling is reached", async () => {
    // Menu construction is many evaluations on its own evaluator. A ceiling it
    // never consults bounds only the trial loop that follows, and a run that
    // spends its whole allowance choosing demos never scores a candidate.
    const result = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 3,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      validationSet: [DEMO_TRAINSET[0] as (typeof DEMO_TRAINSET)[number]],
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map((datum) => `answer ${datum.id}`),
          scores: batch.map((datum) =>
            datum.good && candidate.instruction !== "" ? 1 : 0,
          ),
          usage: batch.map(() => ({ costUsd: 1 })),
        }),
      },
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 400,
      maxCostUsd: 2,
    });

    expect(result.stopReason).toBe("costExhausted");
    // A dollar on the first demo set, three on the batch of the second that
    // crossed the ceiling, one on the seed sweep. The third set is never
    // built; unbounded, the three sets alone sweep the trainingSet three times.
    expect(result.usage.costUsd).toBe(5);
  });

  test("re-runs the trainingSet for each demo set", async () => {
    // A stochastic system does not give the same verdict twice. Here nothing
    // succeeds on first sight and everything succeeds on second, so a single
    // harvesting pass comes back empty and only an independent pass per set
    // finds anything to show. Under deterministic scoring the passes agree and
    // this costs rollouts for nothing, which is the trade MIPROv2 makes.
    const seen = new Set<number>();

    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 2,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map((datum) => `answer ${datum.id}`),
          scores: batch.map((datum) => {
            const repeat = seen.has(datum.id);
            seen.add(datum.id);
            return repeat ? 1 : 0;
          }),
        }),
      },
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 400,
    });

    const blocks = result.menu.demos.filter((entry) =>
      entry.includes("<demo>"),
    );
    expect(blocks.length).toBeGreaterThan(0);
  });

  test("builds a demo set from labels when no rollout earns one", async () => {
    // MIPROv2 always keeps a labels-only set on the menu. Nothing the system
    // produces here is good enough to harvest, so without gold outputs the
    // component would have no examples to search over at all — and a labelled
    // set costs no rollouts, because the output is already known.
    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 2,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => "wrong"),
          scores: batch.map(() => 0),
        }),
      },
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      goldOutput: (datum) => `answer ${datum.id}`,
      maxMetricCalls: 400,
    });

    const blocks = result.menu.demos.filter((entry) =>
      entry.includes("<demo>"),
    );

    expect(blocks.length).toBeGreaterThan(0);
    expect(blocks[0]).toContain("answer 0");
  });

  test("keeps a zero-shot option in every demo menu", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 2,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "seeded block" },
      trainingSet: DEMO_TRAINSET,
      adapter: demoAdapter(),
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 400,
    });

    expect(result.menu.demos).toContain("");
  });

  test("charges bootstrapping to the metric budget", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 1,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 2,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: demoAdapter(),
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 400,
    });

    expect(result.bootstrapMetricCalls).toBeGreaterThan(0);
    expect(result.metricCalls).toBeGreaterThanOrEqual(
      result.bootstrapMetricCalls,
    );
  });

  test("refuses a component that is both supplied and bootstrapped", async () => {
    await expect(
      new MiproOptimizer({ maxTrials: 1, minibatchSize: 2 }).optimize({
        seedCandidate: { demos: "" },
        trainingSet: DEMO_TRAINSET,
        adapter: demoAdapter(),
        reflect: UNUSED_REFLECT,
        demoComponents: ["demos"],
        componentOptions: { demos: ["a", "b"] },
        maxMetricCalls: 400,
      }),
    ).rejects.toThrow(/both/);
  });

  test("searches bootstrapped demos alongside instructions", async () => {
    // Carrying examples is worth more than the instruction alone, so the run
    // can only reach 1 if a harvested block actually entered the search space.
    const result = await new MiproOptimizer({
      maxTrials: 12,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 3,
      demoMinScore: 0.5,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map((datum) => `answer ${datum.id}`),
          scores: batch.map(() =>
            candidate.demos.includes("<demo>") ? 1 : 0.5,
          ),
        }),
      },
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 600,
    });

    expect(result.bestCandidate.demos).toContain("<demo>");
    expect(result.bestScore).toBe(1);
    expect(result.seedScore).toBe(0.5);
  });

  test("generates a demo set of the full requested size", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      demoSets: 3,
      maxDemos: 4,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer.", demos: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: demoAdapter(),
      reflect: async () => "```\nproposal\n```",
      demoComponents: ["demos"],
      maxMetricCalls: 600,
    });

    const largest = Math.max(
      ...result.menu.demos.map((entry) => entry.split("<demo>").length - 1),
    );
    expect(largest).toBe(4);
  });

  test("charges every harvested rollout even when the budget runs out mid-menu", async () => {
    // The first demo component can exhaust the allowance, leaving the second
    // with nothing to spend. Whatever it still manages to run has to land on
    // the bill.
    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      demoSets: 2,
      seed: 5,
    }).optimize({
      seedCandidate: { first: "", second: "" },
      trainingSet: DEMO_TRAINSET,
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map((datum) => `answer ${datum.id}`),
          scores: batch.map(() => 1),
        }),
      },
      reflect: UNUSED_REFLECT,
      demoComponents: ["first", "second"],
      maxMetricCalls: 12,
    });

    // Some harvesting fits, the rest does not, and the seed sweep still runs.
    expect(result.bootstrapMetricCalls).toBeGreaterThan(0);
    expect(result.metricCalls).toBeGreaterThanOrEqual(
      result.bootstrapMetricCalls,
    );
    expect(result.metricCalls).toBeLessThanOrEqual(12);
    expect(result.seedScore).toBe(1);
  });

  test("solves a space too large to cover by modelling components jointly", async () => {
    // 3125 configurations against 60 trials. Score counts adjacent components
    // that agree and never which option they use, so every per-component
    // marginal is flat: only a surrogate that keeps components together has
    // anything to learn from.
    const options = ["v", "w", "x", "y", "z"];
    const names = ["a", "b", "c", "d", "e"] as const;

    const result = await new MiproOptimizer({
      maxTrials: 60,
      minibatchSize: 2,
      seed: 2,
    }).optimize({
      seedCandidate: { a: "", b: "", c: "", d: "", e: "" },
      trainingSet: Array.from({ length: 4 }, (_, n) => ({ n })),
      adapter: {
        evaluate: ({ batch, candidate }) => {
          const parts = names.map((name) => candidate[name] as string);
          let agree = 0;
          for (let index = 0; index + 1 < parts.length; index += 1) {
            if (parts[index] === parts[index + 1] && parts[index] !== "") {
              agree += 1;
            }
          }
          return {
            outputs: batch.map(() => ""),
            scores: batch.map(() => agree / 4),
          };
        },
      },
      reflect: UNUSED_REFLECT,
      componentOptions: {
        a: options,
        b: options,
        c: options,
        d: options,
        e: options,
      },
      maxMetricCalls: 20000,
    });

    expect(result.bestScore).toBeGreaterThanOrEqual(0.75);
  });

  test("steers from the seed's score without spending a trial to learn it", async () => {
    // 256 configurations, and only the seed scores. Reaching it by chance in
    // 20 draws is under one run in ten; reaching it because the seed's full
    // sweep was registered as a trial is immediate.
    const options = ["p", "q", "r"];
    const names = ["a", "b", "c", "d"] as const;

    const result = await new MiproOptimizer({
      maxTrials: 20,
      minibatchSize: 2,
      startupTrials: 1,
      seed: 3,
    }).optimize({
      seedCandidate: { a: "", b: "", c: "", d: "" },
      trainingSet: Array.from({ length: 4 }, (_, n) => ({ n })),
      adapter: {
        evaluate: ({ batch, candidate }) => {
          const seeded = names.every((name) => candidate[name] === "");
          return {
            outputs: batch.map(() => ""),
            scores: batch.map(() => (seeded ? 1 : 0)),
          };
        },
      },
      reflect: UNUSED_REFLECT,
      componentOptions: { a: options, b: options, c: options, d: options },
      maxMetricCalls: 20000,
    });

    expect(
      result.observations.some((observation) =>
        observation.choices.every((choice) => choice === 0),
      ),
    ).toBe(true);
  });

  test("stops chasing a configuration its full sweep disproved", async () => {
    // "lucky" scores on one training instance and nothing else, so a minibatch
    // can read it high while a full sweep reads it at zero. Feeding the sweep's
    // score back is what keeps that lucky reading from pulling proposals for
    // the rest of the run. The effect is a shift in how often a disproved
    // configuration comes back, so it is counted across seeds: a single run
    // turns on which batch the sampler happened to draw.
    let chased = 0;

    for (let seed = 1; seed <= 10; seed += 1) {
      const result = await new MiproOptimizer({
        maxTrials: 24,
        minibatchSize: 1,
        fullEvalInterval: 4,
        startupTrials: 1,
        seed,
      }).optimize({
        seedCandidate: { a: "" },
        trainingSet: Array.from({ length: 6 }, (_, n) => ({ n })),
        validationSet: Array.from({ length: 6 }, (_, n) => ({ n: n + 100 })),
        adapter: {
          evaluate: ({ batch, candidate }) => ({
            outputs: batch.map(() => ""),
            scores: batch.map((datum) => {
              if (candidate.a === "lucky") {
                return datum.n === 0 ? 1 : 0;
              }
              return candidate.a === "real" ? 0.6 : 0;
            }),
          }),
        },
        reflect: UNUSED_REFLECT,
        componentOptions: { a: ["lucky", "real", "dud"] },
        maxMetricCalls: 20000,
      });

      const sweptAt = result.observations.find(
        (observation) => observation.promoted && observation.choices[0] === 1,
      );
      chased += result.observations.filter(
        (observation) =>
          observation.choices[0] === 1 &&
          observation.trial > (sweptAt?.trial ?? Number.MAX_SAFE_INTEGER),
      ).length;
    }

    // 32 across these seeds with the sweep's score fed back, 65 without.
    expect(chased).toBeLessThanOrEqual(45);
  });

  test("shows the proposer the rest of the system it is writing into", async () => {
    // A component's instruction is read alongside its neighbours, not alone.
    // MIPROv2's proposer is given the whole program for the same reason: an
    // instruction written blind to what the other components already say will
    // duplicate them or contradict them.
    const prompts: string[] = [];

    await new MiproOptimizer({
      maxTrials: 1,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      seed: 5,
    }).optimize({
      seedCandidate: {
        retriever: "Find the relevant passage.",
        writer: "Answer in one sentence.",
      },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: baseAdapter(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nproposal\n```";
      },
      maxMetricCalls: 400,
    });

    const forRetriever = prompts.find((prompt) =>
      prompt.includes("Find the relevant passage."),
    ) as string;

    expect(forRetriever).toContain("Answer in one sentence.");
  });

  test("grounds proposals in a summary of the data", async () => {
    // A handful of raw exemplars shows what one input looks like, not what the
    // dataset is. MIPROv2's proposer reads a summary written over more data
    // than fits in the prompt, so an instruction can be aimed at the task as a
    // whole rather than at three examples of it.
    const prompts: string[] = [];

    await new MiproOptimizer({
      maxTrials: 1,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      seed: 5,
    }).optimize({
      seedCandidate: { instruction: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: baseAdapter(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nshort support questions about devices\n```";
      },
      maxMetricCalls: 400,
    });

    const proposal = prompts.find((prompt) =>
      prompt.includes("<current_instruction>"),
    ) as string;

    expect(proposal).toContain("short support questions about devices");
  });

  test("generates a menu with the reflection model when none is supplied", async () => {
    let call = 0;

    const result = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
      instructionsPerComponent: 3,
    }).optimize({
      seedCandidate: { instruction: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: baseAdapter(),
      // The first call writes the dataset summary, so proposals start at one.
      reflect: async ({ prompt }) => {
        if (prompt.includes("<examples>")) {
          return "```\na dataset\n```";
        }
        call += 1;
        return `\`\`\`\nproposal ${call}\n\`\`\``;
      },
      maxMetricCalls: 400,
    });

    // Three instructions, plus the one call that summarizes the dataset.
    expect(result.reflectionCalls).toBe(4);
    expect(result.menu.instruction).toEqual([
      "Answer.",
      "proposal 1",
      "proposal 2",
      "proposal 3",
    ]);
  });

  test("grounds generated instructions in example task inputs", async () => {
    const prompts: string[] = [];

    await new MiproOptimizer({
      maxTrials: 1,
      minibatchSize: 2,
      instructionsPerComponent: 1,
      exemplars: 2,
    }).optimize({
      seedCandidate: { instruction: "Answer." },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: baseAdapter(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nproposal\n```";
      },
      maxMetricCalls: 400,
    });

    const proposal = prompts.find((prompt) =>
      prompt.includes("<current_instruction>"),
    ) as string;
    const questions = KEYWORD_EXAMPLES.map((example) => example.question);
    const shown = questions.filter((question) => proposal.includes(question));

    expect(shown).toHaveLength(2);
  });

  test("promotes a promising configuration to a full evaluation", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 12,
      minibatchSize: 2,
      seed: 3,
    }).optimize(jointTask());

    expect(result.fullEvaluations).toBeGreaterThan(0);
    // The reported best score is a full sweep, never a minibatch reading.
    expect(result.observations.some((entry) => entry.promoted)).toBe(true);
  });

  test("never spends more than the metric call budget", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 100,
      minibatchSize: 3,
    }).optimize({ ...jointTask(), maxMetricCalls: 41 });

    expect(result.metricCalls).toBeLessThanOrEqual(41);
    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("stops once the trial limit is reached", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 5,
      minibatchSize: 2,
    }).optimize(jointTask());

    expect(result.trials).toBe(5);
    expect(result.stopReason).toBe("maxTrials");
  });

  test("stops once no full sweep is affordable", async () => {
    // A twenty-instance validation set against single-instance minibatches: after the
    // seed sweep the allowance still covers plenty of readings but never
    // another sweep, so nothing found from here could be promoted and the
    // incumbent is already decided. Spending the rest on readings nobody can
    // act on is waste, not caution.
    const result = await new MiproOptimizer({
      maxTrials: 100,
      minibatchSize: 1,
      seed: 5,
    }).optimize({
      ...jointTask(),
      trainingSet: Array.from({ length: 20 }, () => KEYWORD_EXAMPLES[0]!),
      maxMetricCalls: 25,
    });

    expect(result.stopReason).toBe("budgetExhausted");
    expect(result.trials).toBeLessThanOrEqual(2);
  });

  test("records every configuration it scored", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 6,
      minibatchSize: 2,
    }).optimize(jointTask());

    expect(result.observations).toHaveLength(6);
    for (const observation of result.observations) {
      expect(observation.choices).toHaveLength(2);
    }
  });

  test("scores the winner on a held-out testSet", async () => {
    const result = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
    }).optimize({
      ...jointTask(),
      testSet: [
        { question: "held out, satisfied", required: ["hold"] },
        { question: "held out, unsatisfiable", required: ["zzz-never"] },
      ],
    });

    expect(result.testMetricCalls).toBe(2);
    expect(result.testScore).toBeGreaterThanOrEqual(0);
  });

  test("keeps the held-out sweep out of the run's usage", async () => {
    // No ceiling bounds the held-out sweep: it runs once the search has already
    // stopped. Counting it in `usage` would describe a run that honoured
    // `maxCostUsd` as having overrun it.
    const result = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
    }).optimize({
      ...jointTask(),
      adapter: pricedAdapter(),
      testSet: [
        { question: "held out, satisfied", required: ["hold"] },
        { question: "held out, unsatisfiable", required: ["zzz-never"] },
      ],
    });

    expect(result.usage.rollouts).toBe(result.metricCalls);
    expect(result.testUsage).toEqual({
      inputTokens: 20,
      outputTokens: 10,
      totalTokens: 30,
      costUsd: 0,
      rollouts: 2,
    });
  });
  test("stops when the signal is aborted", async () => {
    const controller = new AbortController();
    const keyword = createKeywordAdapter();
    let evaluations = 0;

    const result = await new MiproOptimizer({
      maxTrials: 100,
      minibatchSize: 2,
    }).optimize({
      ...jointTask(),
      adapter: {
        evaluate: (args) => {
          evaluations += 1;
          if (evaluations === 3) {
            controller.abort();
          }
          return keyword.evaluate(args);
        },
      },
      signal: controller.signal,
    });

    expect(result.stopReason).toBe("aborted");
  });

  test("is reproducible for a given seed", async () => {
    const run = async () =>
      (
        await new MiproOptimizer({
          maxTrials: 8,
          minibatchSize: 2,
          seed: 11,
        }).optimize(jointTask())
      ).observations.map((entry) => entry.choices.join(","));

    expect(await run()).toEqual(await run());
  });
});

describe("MiproOptimizer checkpoints", () => {
  test("survives a round trip through JSON", async () => {
    let snapshot: MiproSnapshot | undefined;

    await new MiproOptimizer({ maxTrials: 3, minibatchSize: 2 }).optimize({
      ...jointTask(),
      onCheckpoint: (taken) => {
        snapshot = taken;
      },
    });

    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  test("checkpoints after the menus are built and after every trial", async () => {
    const trials: number[] = [];

    await new MiproOptimizer({ maxTrials: 3, minibatchSize: 2 }).optimize({
      ...jointTask(),
      onCheckpoint: (taken) => {
        trials.push(taken.trial);
      },
    });

    expect(trials).toEqual([0, 1, 2, 3]);
  });

  test("does not re-score the seed candidate", async () => {
    const interrupted = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
    }).optimize({ ...jointTask(), cache: false });

    const phases: string[] = [];
    await new MiproOptimizer({ maxTrials: 4, minibatchSize: 2 }).optimize({
      ...jointTask(),
      cache: false,
      resumeFrom: interrupted.snapshot,
      reporters: [
        {
          onEvent: (event) => {
            phases.push(event.type === "evaluation" ? event.phase : event.type);
          },
        },
      ],
    });

    expect(phases).not.toContain("seed");
  });

  test("does not restart candidate ids after a resume", async () => {
    // Reporters key rows by candidateId. Restarting the counter at zero makes a
    // resumed run's candidates collide with the interrupted run's in whatever
    // store the reporter is writing to.
    let before = 0;
    let after = 0;

    const interrupted = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      ...jointTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              before = event.bestCandidateId;
            }
          },
        },
      ],
    });

    await new MiproOptimizer({
      maxTrials: 8,
      minibatchSize: 2,
      seed: 1,
    }).optimize({
      ...jointTask(),
      resumeFrom: interrupted.snapshot,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "finish") {
              after = event.bestCandidateId;
            }
          },
        },
      ],
    });

    expect(before).toBeGreaterThan(0);
    expect(after).toBeGreaterThanOrEqual(before);
  });

  test("carries usage already spent into a resumed run", async () => {
    // `maxCostUsd` is a ceiling on the run, not on the segment. A resumed run
    // that restarts its token accounting at zero lets an interrupted-and-
    // resumed loop spend the ceiling over and over.
    const priced = {
      ...jointTask(),
      cache: false as const,
      adapter: pricedAdapter(),
    };

    const interrupted = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
    }).optimize(priced);

    const resumed = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
    }).optimize({ ...priced, resumeFrom: interrupted.snapshot });

    expect(interrupted.usage.inputTokens).toBeGreaterThan(0);
    // Ten input tokens a rollout, uncached, over both segments.
    expect(resumed.usage.inputTokens).toBe(resumed.metricCalls * 10);
  });

  test("reuses the menus the first run paid for", async () => {
    // Rebuilding them would buy the same options again and reindex every
    // choice vector the surrogate was already fitted on.
    const interrupted = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
    }).optimize(jointTask());

    const resumed = await new MiproOptimizer({
      maxTrials: 4,
      minibatchSize: 2,
    }).optimize({ ...jointTask(), resumeFrom: interrupted.snapshot });

    expect(resumed.menu).toEqual(interrupted.menu);
  });

  test("keeps the observations the surrogate was fitted on", async () => {
    const interrupted = await new MiproOptimizer({
      maxTrials: 3,
      minibatchSize: 2,
    }).optimize(jointTask());

    const resumed = await new MiproOptimizer({
      maxTrials: 5,
      minibatchSize: 2,
    }).optimize({ ...jointTask(), resumeFrom: interrupted.snapshot });

    expect(resumed.observations.length).toBeGreaterThan(
      interrupted.observations.length,
    );
    expect(
      resumed.observations.slice(0, interrupted.observations.length),
    ).toEqual(interrupted.observations);
  });

  test("refuses a checkpoint taken against a different seed candidate", async () => {
    const interrupted = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
    }).optimize(jointTask());

    await expect(
      new MiproOptimizer({ maxTrials: 2, minibatchSize: 2 }).optimize({
        ...jointTask(),
        seedCandidate: { alpha: "different", beta: "" },
        resumeFrom: interrupted.snapshot,
      }),
    ).rejects.toThrow("does not belong to this run");
  });

  test("carries the evaluation cache in the checkpoint", async () => {
    const interrupted = await new MiproOptimizer({
      maxTrials: 2,
      minibatchSize: 2,
    }).optimize(jointTask());

    expect(interrupted.snapshot.cache?.length).toBeGreaterThan(0);
  });
});

describe("MiproOptimizer reporting", () => {
  test("reports the seed as candidate 0, before any improvement", async () => {
    // The seed is what every later candidate is read against. A report that
    // starts at the first improvement has nothing to compare it to.
    const accepted: { id: number; candidate: Record<string, string> }[] = [];

    await new MiproOptimizer({ maxTrials: 6, minibatchSize: 2 }).optimize({
      ...jointTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              accepted.push({
                id: event.candidateId,
                candidate: { ...event.candidate },
              });
            }
          },
        },
      ],
    });

    expect(accepted[0]?.id).toBe(0);
    expect(accepted[0]?.candidate).toEqual({ alpha: "", beta: "" });
  });

  test("reports an acceptance with the text that scored", async () => {
    const accepted: Record<string, string>[] = [];

    await new MiproOptimizer({ maxTrials: 6, minibatchSize: 2 }).optimize({
      ...jointTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              accepted.push({ ...event.candidate });
            }
          },
        },
      ],
    });

    expect(accepted.length).toBeGreaterThan(0);
  });

  test("reports a per-instance row aligned with the validation set", async () => {
    const rows: (number | undefined)[][] = [];

    await new MiproOptimizer({ maxTrials: 6, minibatchSize: 2 }).optimize({
      ...jointTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              rows.push([...event.instanceScores]);
            }
          },
        },
      ],
    });

    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row).toHaveLength(KEYWORD_EXAMPLES.length);
    }
  });

  test("names the winner in finish with an id an acceptance carried", async () => {
    const acceptedIds: number[] = [];
    let bestCandidateId: number | undefined;

    await new MiproOptimizer({ maxTrials: 6, minibatchSize: 2 }).optimize({
      ...jointTask(),
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted") {
              acceptedIds.push(event.candidateId);
            }
            if (event.type === "finish") {
              bestCandidateId = event.bestCandidateId;
            }
          },
        },
      ],
    });

    expect(acceptedIds).toContain(bestCandidateId);
  });

  test("flushes every reporter once the run ends", async () => {
    const flushed: string[] = [];

    await new MiproOptimizer({ maxTrials: 6, minibatchSize: 2 }).optimize({
      ...jointTask(),
      reporters: [
        {
          flush: async () => {
            flushed.push("first");
          },
        },
        {
          flush: async () => {
            flushed.push("second");
          },
        },
      ],
    });

    expect(flushed.toSorted()).toEqual(["first", "second"]);
  });
});
