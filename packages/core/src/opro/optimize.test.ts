import { afterEach, describe, expect, test, vi } from "vitest";
import type { Optimizer, OptimizerResult } from "../optimizer.js";
import {
  KEYWORD_EXAMPLES,
  createHillClimbingReflector,
  createKeywordAdapter,
} from "../testing.js";
import type { Adapter } from "../types.js";
import { OproOptimizer } from "./optimize.js";
import type { OproSnapshot, OproStopReason } from "./optimize.js";

const SEED = { instruction: "Answer the user question." };

/** The adapter contract without any reflective surface on it at all. */
function baseAdapter(): Adapter<
  (typeof KEYWORD_EXAMPLES)[number],
  unknown,
  string
> {
  const keyword = createKeywordAdapter();
  return { evaluate: (args) => keyword.evaluate(args) };
}

function task() {
  return {
    seedCandidate: SEED,
    trainingSet: KEYWORD_EXAMPLES,
    adapter: baseAdapter(),
    reflect: createHillClimbingReflector(),
    maxMetricCalls: 400,
  };
}

/** The scores an OPRO prompt shows, in the order it shows them. */
function scoresIn(prompt: string): number[] {
  return [...prompt.matchAll(/score:\s*([\d.]+)/g)].map((match) =>
    Number(match[1]),
  );
}

describe("OproOptimizer", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  test("stops once the wall clock deadline passes", async () => {
    // Rollout and cost ceilings bound what a run spends, not how long it
    // takes: a run stuck behind a rate limit costs nothing and runs forever.
    vi.useFakeTimers();

    const result = await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 20,
    }).optimize({
      ...task(),
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
    const result = await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 5,
    }).optimize({
      ...task(),
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
    // zero in reports a score no rollout ever produced, and the search then
    // compares every later attempt against it.
    const result = await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 1,
      seed: 1,
    }).optimize({
      ...task(),
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
    const opro = new OproOptimizer({ proposalsPerRound: 2, maxRounds: 2 });
    const contract: Optimizer<OproStopReason> = opro;

    const result = await opro.optimize(task());
    const outcome: OptimizerResult<"instruction", OproStopReason> = result;

    expect(contract).toBe(opro);
    expect(outcome.bestScore).toBeGreaterThanOrEqual(0);
  });

  test("runs against an adapter that reports no feedback at all", async () => {
    const keyword = createKeywordAdapter();

    const result = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 4,
    }).optimize({
      ...task(),
      adapter: {
        evaluate: async (args) => {
          const evaluation = await keyword.evaluate(args);
          return { outputs: evaluation.outputs, scores: evaluation.scores };
        },
      },
    });

    expect(result.bestScore).toBeGreaterThan(result.seedScore);
  });

  test("improves on the seed by reading the score history", async () => {
    const result = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 8,
    }).optimize(task());

    expect(result.seedScore).toBe(0);
    expect(result.bestScore).toBeGreaterThan(0);
  });

  test("shows the proposer where its instruction lands", async () => {
    // OPRO's meta-prompt marks the instruction's slot inside each example so
    // the model writes something that fits where it will actually be read,
    // rather than prose about the task in the abstract.
    const prompts: string[] = [];

    await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 1,
      exemplars: 2,
      seed: 1,
    }).optimize({
      ...task(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nnew instruction\n```";
      },
    });

    const inputs = prompts[0]?.slice(
      prompts[0].indexOf("<inputs>"),
      prompts[0].indexOf("</inputs>"),
    ) as string;

    expect([...inputs.matchAll(/<INS>/g)]).toHaveLength(2);
  });

  test("draws fresh exemplars each round", async () => {
    // The reference resamples its few-shot questions every step. Holding one
    // slice fixed for the whole run lets the search tune its instruction to
    // three particular inputs, which is the overfitting the resampling exists
    // to prevent.
    const prompts: string[] = [];

    await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 6,
      exemplars: 1,
      seed: 1,
    }).optimize({
      ...task(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nnew instruction\n```";
      },
    });

    const shown = prompts.map((prompt) =>
      prompt.slice(prompt.indexOf("<inputs>"), prompt.indexOf("</inputs>")),
    );

    expect(shown.length).toBeGreaterThan(1);
    expect(new Set(shown).size).toBeGreaterThan(1);
  });

  test("shows the proposer the attempts in ascending score order", async () => {
    const prompts: string[] = [];

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 5 }).optimize({
      ...task(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nAnswer the user question. hold\n```";
      },
    });

    const withHistory = prompts.filter((prompt) => scoresIn(prompt).length > 1);
    expect(withHistory.length).toBeGreaterThan(0);
    for (const prompt of withHistory) {
      const scores = scoresIn(prompt);
      expect([...scores].sort((a, b) => a - b)).toEqual(scores);
    }
  });

  test("caps the history it shows, keeping the strongest attempts", async () => {
    const prompts: string[] = [];
    const reflect = createHillClimbingReflector();

    await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 8,
      historySize: 3,
    }).optimize({
      ...task(),
      reflect: async (args) => {
        prompts.push(args.prompt);
        return reflect(args);
      },
    });

    const last = prompts[prompts.length - 1] as string;
    expect(scoresIn(last).length).toBeLessThanOrEqual(3);
  });

  test("grounds the prompt in example task inputs", async () => {
    const prompts: string[] = [];

    await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 1,
      exemplars: 2,
    }).optimize({
      ...task(),
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nrewritten\n```";
      },
    });

    const questions = KEYWORD_EXAMPLES.map((example) => example.question);
    const shown = questions.filter((question) =>
      (prompts[0] as string).includes(question),
    );
    expect(shown).toHaveLength(2);
  });

  test("never spends more than the metric call budget", async () => {
    const result = await new OproOptimizer({ proposalsPerRound: 3 }).optimize({
      ...task(),
      maxMetricCalls: 29,
    });

    expect(result.metricCalls).toBeLessThanOrEqual(29);
    expect(result.stopReason).toBe("budgetExhausted");
  });

  test("stops when the proposal model stops producing anything new", async () => {
    // A round whose every proposal was already tried spends no rollouts, so
    // the budget guard never fires: with the default round limit the run would
    // never end at all.
    const result = await new OproOptimizer({ proposalsPerRound: 2 }).optimize({
      ...task(),
      reflect: async () => "```\nthe only thing it ever says\n```",
      maxMetricCalls: 10_000,
    });

    expect(result.stopReason).toBe("proposalsExhausted");
  });

  test("stops once the round limit is reached", async () => {
    const result = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 3,
    }).optimize({ ...task(), maxMetricCalls: 10_000 });

    expect(result.rounds).toBe(3);
    expect(result.stopReason).toBe("maxRounds");
  });

  test("stops once the reflection budget is reached", async () => {
    const result = await new OproOptimizer({
      proposalsPerRound: 2,
      maxReflectionCalls: 5,
    }).optimize({ ...task(), maxMetricCalls: 10_000 });

    expect(result.reflectionCalls).toBeLessThanOrEqual(5);
    expect(result.stopReason).toBe("reflectionBudgetExhausted");
  });

  test("records every attempt it scored", async () => {
    const result = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 3,
    }).optimize({ ...task(), maxMetricCalls: 10_000 });

    // The seed, plus whatever survived deduplication in each round.
    expect(result.trajectory[0]?.round).toBe(0);
    expect(result.trajectory.length).toBeGreaterThan(1);
    for (const attempt of result.trajectory) {
      expect(attempt.score).toBeGreaterThanOrEqual(0);
    }
  });

  test("scores the winner on a held-out testSet", async () => {
    const result = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 2,
    }).optimize({
      ...task(),
      testSet: [
        { question: "held out, satisfied", required: ["answer"] },
        { question: "held out, unsatisfiable", required: ["zzz-never"] },
      ],
    });

    expect(result.testScore).toBe(0.5);
    expect(result.testMetricCalls).toBe(2);
  });

  test("stops when the signal is aborted", async () => {
    const controller = new AbortController();
    const keyword = createKeywordAdapter();
    let evaluations = 0;

    const result = await new OproOptimizer({
      proposalsPerRound: 2,
    }).optimize({
      ...task(),
      maxMetricCalls: 10_000,
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

  test("keeps a separate history per component", async () => {
    const components: string[] = [];

    await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 4,
    }).optimize({
      ...task(),
      seedCandidate: { intro: "Answer.", outro: "Be brief." },
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "roundStart") {
              components.push(event.component);
            }
          },
        },
      ],
    });

    expect(components).toEqual(["intro", "outro", "intro", "outro"]);
  });
});

describe("OproOptimizer scoring subset", () => {
  const TRAIN = Array.from({ length: 20 }, (_, n) => ({ n }));
  const VAL = Array.from({ length: 20 }, (_, n) => ({ n: n + 100 }));

  test("refuses a scoring set that is not a positive integer", () => {
    expect(() => new OproOptimizer({ scoringSetSize: 0 })).toThrow(
      /scoringSetSize/,
    );
  });

  test("scores proposals on the subset instead of the whole validationSet", async () => {
    let drawn = 0;
    // Four proposals against a 20-instance validation set costs 80 rollouts to screen.
    // OPRO screens on a small fixed slice of the training set instead, which is
    // what makes a large validation set affordable at all.
    const result = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 2,
      scoringSetSize: 4,
      seed: 1,
    }).optimize({
      seedCandidate: { instruction: "seed" },
      trainingSet: TRAIN,
      validationSet: VAL,
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => 0.5),
        }),
      },
      reflect: async () => {
        drawn += 1;
        return `\`\`\`\nrewrite ${drawn}\n\`\`\``;
      },
      maxMetricCalls: 5000,
    });

    // Screening all four on the validation set would be 80 rollouts on its own.
    expect(result.metricCalls).toBeLessThan(80);
  });

  test("keeps the scoring subset fixed for the whole run", async () => {
    // The meta-prompt ranks attempts against each other, so a score is only
    // meaningful if every attempt was measured on the same instances. A
    // resampled subset would turn that ranking into noise.
    const scored = new Set<number>();
    let call = 0;

    await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 3,
      scoringSetSize: 4,
      seed: 1,
    }).optimize({
      seedCandidate: { instruction: "seed" },
      trainingSet: TRAIN,
      validationSet: VAL,
      adapter: {
        evaluate: ({ batch }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map((datum) => {
            if (datum.n < 100) {
              scored.add(datum.n);
            }
            return 0.5;
          }),
        }),
      },
      reflect: async () => {
        call += 1;
        return `\`\`\`\nrewrite ${call}\n\`\`\``;
      },
      maxMetricCalls: 5000,
    });

    expect(scored.size).toBe(4);
  });

  test("reports a winner measured on the validationSet, not on the subset", async () => {
    // "over" is perfect on the training set and worthless on the validation set. The
    // search will chase it, because the search only ever sees the subset — but
    // what gets reported has to be a number the full validation set actually produced.
    const result = await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 3,
      scoringSetSize: 4,
      fullEvalInterval: 1,
      seed: 1,
    }).optimize({
      seedCandidate: { instruction: "seed" },
      trainingSet: TRAIN,
      validationSet: VAL,
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map((datum) => {
            if (candidate.instruction !== "over") {
              return 0.5;
            }
            return datum.n < 100 ? 1 : 0;
          }),
        }),
      },
      reflect: async () => "```\nover\n```",
      maxMetricCalls: 5000,
    });

    expect(result.bestScore).toBe(0.5);
    expect(result.bestCandidate.instruction).toBe("seed");
  });
});

describe("OproOptimizer score history", () => {
  test("never shows a score measured before another component moved", async () => {
    // Round 0 scores alpha="A1" at 0.5 while beta is still the seed; round 1
    // moves beta and re-measures the pair at 1.0. By round 2 the earlier 50 is
    // a reading of a system that no longer exists, and listing it beside the
    // current one invites the model to read a gradient across two
    // incomparable measurements. A1 itself may reappear — with its new score.
    const prompts: string[] = [];
    const proposals = ["A1", "B1", "A2"];
    let call = 0;

    await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 3,
      seed: 3,
    }).optimize({
      seedCandidate: { alpha: "", beta: "" },
      trainingSet: [{ id: 0 }],
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(
            () =>
              (candidate.alpha === "A1" ? 0.5 : 0) +
              (candidate.beta === "B1" ? 0.5 : 0),
          ),
        }),
      },
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        const text = proposals[call] as string;
        call += 1;
        return `\`\`\`\n${text}\n\`\`\``;
      },
      maxMetricCalls: 400,
    });

    expect(prompts).toHaveLength(3);
    expect(prompts[2]).not.toContain("score: 50");
    expect(prompts[2]).toContain("score: 100");
  });

  test("retries a rejected text once another component has moved", async () => {
    // "A1" is worthless on its own and perfect beside "B1". Round 0 measures it
    // alone and rejects it; round 1 accepts "B1". A candidate is the whole
    // assignment, so proposing "A1" again in round 2 is a different candidate
    // than the one that failed — treating it as already tried puts the pair
    // permanently out of reach.
    const proposals = ["A1", "B1", "A1"];
    let call = 0;

    const result = await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 3,
      seed: 3,
    }).optimize({
      seedCandidate: { alpha: "", beta: "" },
      trainingSet: [{ id: 0 }],
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => {
            if (candidate.alpha === "A1" && candidate.beta === "B1") {
              return 1;
            }
            return candidate.beta === "B1" ? 0.5 : 0;
          }),
        }),
      },
      reflect: async () => {
        const text = proposals[call] as string;
        call += 1;
        return `\`\`\`\n${text}\n\`\`\``;
      },
      maxMetricCalls: 400,
    });

    expect(result.bestScore).toBe(1);
  });

  test("keeps an anchor in every component's history when another one moves", async () => {
    // Filtering stale attempts is only sound if something survives the filter.
    // An accepted candidate is a real measurement of every component's current
    // text in the new context, so it re-anchors the histories it invalidated.
    const blocks: string[] = [];
    const proposals = ["A1", "B1", "A2", "B2"];
    let call = 0;

    await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 4,
      seed: 3,
    }).optimize({
      seedCandidate: { alpha: "", beta: "" },
      trainingSet: [{ id: 0 }],
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(
            () =>
              (candidate.alpha === "A1" ? 0.5 : 0) +
              (candidate.beta === "B1" ? 0.5 : 0),
          ),
        }),
      },
      reflect: async ({ prompt }) => {
        blocks.push(
          prompt.slice(
            prompt.indexOf("<attempts>"),
            prompt.indexOf("</attempts>"),
          ),
        );
        const text = proposals[call] as string;
        call += 1;
        return `\`\`\`\n${text}\n\`\`\``;
      },
      maxMetricCalls: 400,
    });

    expect(blocks).toHaveLength(4);
    for (const block of blocks) {
      expect(block).toContain("score:");
    }
  });

  test("keeps the whole history when the candidate has one component", async () => {
    const prompts: string[] = [];
    const proposals = ["first", "second", "third"];
    let call = 0;

    await new OproOptimizer({
      proposalsPerRound: 1,
      maxRounds: 3,
      seed: 3,
    }).optimize({
      seedCandidate: { alpha: "" },
      trainingSet: [{ id: 0 }],
      adapter: {
        evaluate: ({ batch, candidate }) => ({
          outputs: batch.map(() => ""),
          scores: batch.map(() => (candidate.alpha === "first" ? 0.5 : 0.1)),
        }),
      },
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        const text = proposals[call] as string;
        call += 1;
        return `\`\`\`\n${text}\n\`\`\``;
      },
      maxMetricCalls: 400,
    });

    // Nothing else can move, so every attempt stays comparable — this is the
    // single-instruction case the paper describes.
    expect(prompts[2]).toContain("first");
    expect(prompts[2]).toContain("second");
  });
});

describe("OproOptimizer checkpoints", () => {
  test("survives a round trip through JSON", async () => {
    let snapshot: OproSnapshot | undefined;

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 2 }).optimize({
      ...task(),
      onCheckpoint: (taken) => {
        snapshot = taken;
      },
    });

    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  test("checkpoints after the seed sweep and after every round", async () => {
    const rounds: number[] = [];

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 3 }).optimize({
      ...task(),
      onCheckpoint: (taken) => {
        rounds.push(taken.round);
      },
    });

    expect(rounds).toEqual([0, 1, 2, 3]);
  });

  test("resumes without re-scoring the seed candidate", async () => {
    const interrupted = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 1,
    }).optimize({ ...task(), cache: false });

    const resumed = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 3,
    }).optimize({
      ...task(),
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.rounds).toBe(3);
    expect(resumed.seedScore).toBe(interrupted.seedScore);
    expect(resumed.bestScore).toBeGreaterThanOrEqual(interrupted.bestScore);
  });

  test("re-sweeps nothing when it resumes at the round it stopped on", async () => {
    const interrupted = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 1,
    }).optimize({ ...task(), cache: false });

    const resumed = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 1,
    }).optimize({
      ...task(),
      cache: false,
      resumeFrom: interrupted.snapshot,
    });

    expect(resumed.metricCalls).toBe(interrupted.metricCalls);
  });

  test("carries the score history the meta-prompt is written from", async () => {
    const interrupted = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 2,
    }).optimize(task());

    const resumed = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 3,
    }).optimize({ ...task(), resumeFrom: interrupted.snapshot });

    expect(
      resumed.snapshot.histories.instruction?.length,
    ).toBeGreaterThanOrEqual(
      interrupted.snapshot.histories.instruction?.length ?? 0,
    );
  });

  test("keeps the screening slice a resumed run was drawn against", async () => {
    // Attempts screened on different instances are not the gradient the
    // meta-prompt asks the model to read.
    const interrupted = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 1,
      scoringSetSize: 3,
    }).optimize(task());

    const resumed = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 2,
      scoringSetSize: 3,
    }).optimize({ ...task(), resumeFrom: interrupted.snapshot });

    expect(resumed.snapshot.scoringIndices).toEqual(
      interrupted.snapshot.scoringIndices,
    );
  });

  test("refuses a checkpoint taken against a different seed candidate", async () => {
    const interrupted = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 1,
    }).optimize(task());

    await expect(
      new OproOptimizer({ proposalsPerRound: 2, maxRounds: 1 }).optimize({
        ...task(),
        seedCandidate: { instruction: "Something else entirely." },
        resumeFrom: interrupted.snapshot,
      }),
    ).rejects.toThrow("does not belong to this run");
  });

  test("carries the evaluation cache in the checkpoint", async () => {
    const interrupted = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 1,
    }).optimize(task());

    expect(interrupted.snapshot.cache?.length).toBeGreaterThan(0);
  });
});

describe("OproOptimizer concurrency", () => {
  test("screens a round's proposals at the same time", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await new OproOptimizer({
      proposalsPerRound: 4,
      concurrency: 4,
      maxRounds: 2,
    }).optimize({ ...task(), adapter: tracked.adapter });

    expect(tracked.maxInFlight()).toBeGreaterThan(1);
  });

  test("screens them one at a time by default", async () => {
    const tracked = withOverlapTracking(baseAdapter());

    await new OproOptimizer({ proposalsPerRound: 4, maxRounds: 2 }).optimize({
      ...task(),
      adapter: tracked.adapter,
    });

    expect(tracked.maxInFlight()).toBe(1);
  });

  test("reaches the same trajectory whether or not the screens overlap", async () => {
    const serial = await new OproOptimizer({
      proposalsPerRound: 4,
      maxRounds: 3,
    }).optimize(task());
    const concurrent = await new OproOptimizer({
      proposalsPerRound: 4,
      concurrency: 4,
      maxRounds: 3,
    }).optimize(task());

    expect(concurrent.trajectory).toEqual(serial.trajectory);
    expect(concurrent.bestCandidate).toEqual(serial.bestCandidate);
    expect(concurrent.bestScore).toBe(serial.bestScore);
    expect(concurrent.metricCalls).toBe(serial.metricCalls);
    expect(concurrent.stopReason).toBe(serial.stopReason);
  });

  test("records the same history whichever order the screens finish in", async () => {
    // The history is what the next round's meta-prompt reads, and an attempt
    // is marked accepted only if it beat every attempt before it. Both are
    // decided in draw order, never by whichever screen returned first.
    const run = async (pace: (candidate: string) => number) => {
      const accepted: string[] = [];

      const result = await new OproOptimizer({
        proposalsPerRound: 4,
        concurrency: 4,
        maxRounds: 3,
      }).optimize({
        ...task(),
        adapter: withPacing(baseAdapter(), pace),
        reporters: [
          {
            onEvent: (event) => {
              if (event.type === "attempt") {
                accepted.push(
                  `${event.round}:${event.score}:${event.accepted}`,
                );
              }
            },
          },
        ],
      });

      return { accepted, history: result.snapshot.histories };
    };

    const shortestFirst = await run((candidate) => candidate.length);
    const longestFirst = await run((candidate) => 100 - candidate.length);

    expect(shortestFirst.accepted).toEqual(longestFirst.accepted);
    expect(shortestFirst.history).toEqual(longestFirst.history);
    expect(shortestFirst.accepted.length).toBeGreaterThan(0);
  });

  test("never spends past the budget when the screens overlap", async () => {
    const result = await new OproOptimizer({
      proposalsPerRound: 4,
      concurrency: 4,
      maxRounds: 20,
    }).optimize({ ...task(), maxMetricCalls: 30 });

    expect(result.metricCalls).toBeLessThanOrEqual(30);
  });
});

/** Wraps an adapter to observe how many evaluations are ever in flight at once. */
function withOverlapTracking<Datum, Trajectory, Output>(
  adapter: Adapter<Datum, Trajectory, Output>,
): {
  adapter: Adapter<Datum, Trajectory, Output>;
  maxInFlight: () => number;
} {
  let inFlight = 0;
  let peak = 0;

  return {
    maxInFlight: () => peak,
    adapter: {
      evaluate: async (args) => {
        inFlight += 1;
        peak = Math.max(peak, inFlight);
        try {
          await Promise.resolve();
          return await adapter.evaluate(args);
        } finally {
          inFlight -= 1;
        }
      },
    },
  };
}

/** Settles each evaluation after a delay the candidate's own text decides. */
function withPacing<Datum, Trajectory, Output>(
  adapter: Adapter<Datum, Trajectory, Output>,
  pace: (candidate: string) => number,
): Adapter<Datum, Trajectory, Output> {
  return {
    evaluate: async (args) => {
      const ticks = pace(Object.values(args.candidate).join(" "));
      for (let tick = 0; tick < ticks; tick += 1) {
        await Promise.resolve();
      }
      return adapter.evaluate(args);
    },
  };
}

describe("OproOptimizer reporting", () => {
  test("does not report the seed again on a resumed run", async () => {
    // The seed is candidate 0 of one run, not of every process that continues
    // it. Re-emitting would give a reporter a second baseline for the same
    // search, and a checkpoint is exactly the case where nothing re-measured
    // it to report.
    const interrupted = await new OproOptimizer({
      proposalsPerRound: 2,
      maxRounds: 1,
    }).optimize({ ...task(), cache: false });

    const seedReports: number[] = [];

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 3 }).optimize({
      ...{ ...task(), cache: false },
      resumeFrom: interrupted.snapshot,
      reporters: [
        {
          onEvent: (event) => {
            if (event.type === "candidateAccepted" && event.candidateId === 0) {
              seedReports.push(event.aggregateScore);
            }
          },
        },
      ],
    });

    expect(seedReports).toEqual([]);
  });

  test("reports the seed as candidate 0, before any improvement", async () => {
    // The seed is what every later candidate is read against. A report that
    // starts at the first improvement has nothing to compare it to.
    const accepted: { id: number; candidate: Record<string, string> }[] = [];

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 4 }).optimize({
      ...task(),
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
    expect(accepted[0]?.candidate).toEqual(SEED);
  });

  test("reports an acceptance with the text that scored", async () => {
    const accepted: Record<string, string>[] = [];

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 4 }).optimize({
      ...task(),
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

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 4 }).optimize({
      ...task(),
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

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 4 }).optimize({
      ...task(),
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

    await new OproOptimizer({ proposalsPerRound: 2, maxRounds: 4 }).optimize({
      ...task(),
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
