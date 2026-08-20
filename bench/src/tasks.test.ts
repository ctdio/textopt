import { describe, expect, test } from "vitest";
import { buildAdvicePrompt } from "textopt/simba";
import type { Candidate, EvaluationContext } from "textopt";
import {
  ACTIONS,
  benchTasks,
  MAX_MINIBATCH,
  createBenchAdviser,
  createBenchReflector,
  policyCandidate,
  redactObservations,
  renderBenchDemo,
  requiredActions,
  bestUnconditionalCandidate,
  shotgunCandidate,
  type BenchDatum,
  type BenchTask,
} from "./tasks.js";

const RUN: EvaluationContext = {
  iteration: 0,
  phase: "validation",
  split: "val",
  candidateId: null,
};

describe("splits", () => {
  test("partitions every feature combination across the three splits", () => {
    const task = taskNamed("clean");
    const combos = [
      ...task.trainingSet,
      ...task.validationSet,
      ...task.testSet,
    ].map(comboOf);

    expect(new Set(combos).size).toBe(combos.length);
  });

  test("holds every test combination out of training and validation", () => {
    const task = taskNamed("clean");
    const seen = new Set(
      [...task.trainingSet, ...task.validationSet].map(comboOf),
    );

    for (const datum of task.testSet) {
      expect(seen.has(comboOf(datum))).toBe(false);
    }
  });

  test("selects candidates against instances the training set does not contain", () => {
    const task = taskNamed("clean");
    const training = new Set(task.trainingSet.map(comboOf));

    for (const datum of task.validationSet) {
      expect(training.has(comboOf(datum))).toBe(false);
    }
  });

  test("gives every split the same mix of feature values, so selection and scoring agree", () => {
    const task = taskNamed("clean");

    for (const feature of ["tier", "channel", "issue"] as const) {
      const training = tallyOf({ data: task.trainingSet, feature });
      expect(tallyOf({ data: task.validationSet, feature })).toEqual(training);
      expect(tallyOf({ data: task.testSet, feature })).toEqual(training);
    }
  });

  test("varies each feature against the others, so no rule can hide behind another", () => {
    const task = taskNamed("clean");

    for (const split of [task.trainingSet, task.validationSet, task.testSet]) {
      for (const [left, right] of [
        ["tier", "channel"],
        ["tier", "issue"],
        ["channel", "issue"],
      ] as const) {
        const partners = new Map<string, Set<string>>();
        for (const datum of split) {
          const seen = partners.get(datum[left]) ?? new Set<string>();
          seen.add(datum[right]);
          partners.set(datum[left], seen);
        }

        for (const seen of partners.values()) {
          expect(seen.size).toBeGreaterThan(1);
        }
      }
    }
  });

  test("shows every feature value in all three splits, so every rule is learnable", () => {
    const task = taskNamed("clean");

    for (const feature of ["tier", "channel", "issue"] as const) {
      const training = valuesOf({ data: task.trainingSet, feature });
      expect(valuesOf({ data: task.validationSet, feature })).toEqual(training);
      expect(valuesOf({ data: task.testSet, feature })).toEqual(training);
    }
  });
});

describe("decoys", () => {
  test("carries a feature the policy never rewards", () => {
    const task = taskNamed("clean");
    const policy = policyCandidate(task).instruction as string;

    expect(task.trainingSet[0]).toHaveProperty("region");
    expect(policy).not.toContain("when region is");
  });

  test("scores a coincidence below the rule it imitates", () => {
    const task = taskNamed("clean");
    const decoy = fill({ task, text: "when region is us: sla" });
    const real = fill({ task, text: "when tier is enterprise: sla" });

    expect(meanScore({ task, candidate: decoy })).toBeLessThan(
      meanScore({ task, candidate: real }),
    );
  });

  test("fools the proposer when a coincidence is as well evidenced as the rule", async () => {
    const reflect = createBenchReflector();
    const proposed = await reflect({
      prompt: reflectionPrompt([
        "tier=enterprise channel=phone issue=outage region=us | missing: sla",
        "tier=enterprise channel=email issue=billing region=us | missing: sla",
      ]),
    });

    expect(proposed).toMatch(/when region is us: sla/);
  });

  test("leaves the proposer able to be wrong at the widest batch the bench sweeps", async () => {
    const task = taskNamed("clean");
    const correct = new Set(
      (policyCandidate(task).instruction as string).split("\n"),
    );
    const reflect = createBenchReflector();

    let drawn = 0;
    let right = 0;
    for (
      let start = 0;
      start + MAX_MINIBATCH <= task.trainingSet.length;
      start += 1
    ) {
      const window = task.trainingSet.slice(start, start + MAX_MINIBATCH);
      const proposed = await reflect({
        prompt: reflectionPrompt(
          window.map((datum) => {
            const scored = evaluate({
              task,
              candidate: task.seedCandidate,
              batch: [datum],
            });
            return scored.feedback?.[0] ?? "";
          }),
        ),
      });
      const rules = (proposed.match(/when \w+ is [a-z ]+: [a-z ]+/g) ?? []).map(
        (rule) => rule.trim(),
      );
      drawn += rules.length;
      right += rules.filter((rule) => correct.has(rule)).length;
    }

    expect(drawn).toBeGreaterThan(0);
    expect(right).toBeLessThan(drawn);
  });
});

describe("redaction", () => {
  test("removes the diagnosis and leaves the component's remit", () => {
    const task = taskNamed("clean");
    const scored = evaluate({
      task,
      candidate: task.seedCandidate,
      batch: task.testSet,
    });
    const prompt = `${scored.feedback?.[0] ?? ""} | governs: tier, channel`;

    const redacted = redactObservations(prompt);

    expect(redacted).not.toMatch(/missing:/);
    expect(redacted).toContain("governs: tier, channel");
  });

  test("leaves a redacted prompt with nothing to induce from", async () => {
    const observations = [
      "tier=enterprise channel=phone issue=outage region=us | missing: sla",
      "tier=enterprise channel=email issue=billing region=eu | missing: sla",
      "tier=enterprise channel=chat issue=howto region=us | missing: sla",
    ];

    const informed = await createBenchReflector()({
      prompt: reflectionPrompt(observations),
    });
    const blind = await createBenchReflector()({
      prompt: redactObservations(reflectionPrompt(observations)),
    });

    expect(informed).toMatch(/when tier is enterprise: sla/);
    expect(blind).not.toMatch(/when tier is enterprise: sla/);
  });
});

describe("scoring", () => {
  test("scores the ground-truth policy at the ceiling on held-out data", () => {
    const task = taskNamed("clean");
    const score = meanScore({ task, candidate: policyCandidate(task) });

    expect(score).toBe(1);
  });

  test("scores an empty candidate at the floor", () => {
    const task = taskNamed("clean");
    const score = meanScore({ task, candidate: task.seedCandidate });

    expect(score).toBe(0);
  });

  test("penalises a candidate that emits every action unconditionally", () => {
    const task = taskNamed("clean");
    const shotgun = fill({
      task,
      text: `when tier is any: ${ACTIONS.join(", ")}`,
    });

    expect(meanScore({ task, candidate: shotgun })).toBeLessThan(0.8);
  });

  test("sprays every action on every ticket in each component that can fire it", () => {
    const task = taskNamed("interacting");
    const score = meanScore({ task, candidate: shotgunCandidate(task) });

    expect(score).toBeGreaterThan(0);
  });

  test("scores the same spray on a pipeline task as on a single-component one", () => {
    const single = taskNamed("clean");
    const pipeline = taskNamed("interacting");

    expect(
      meanScore({ task: pipeline, candidate: shotgunCandidate(pipeline) }),
    ).toBe(meanScore({ task: single, candidate: shotgunCandidate(single) }));
  });

  test("picks the best unconditional answer it can find without the held-out set", () => {
    const task = taskNamed("clean");
    const best = bestUnconditionalCandidate(task);

    expect(
      meanScoreOn({ task, candidate: best, batch: task.validationSet }),
    ).toBeGreaterThanOrEqual(
      meanScoreOn({
        task,
        candidate: shotgunCandidate(task),
        batch: task.validationSet,
      }),
    );
  });

  test("does not let a candidate that memorises training answers transfer", () => {
    const task = taskNamed("clean");
    const memorised = fill({
      task,
      text: [...new Set(task.trainingSet.flatMap(requiredActions))].join(" "),
    });

    expect(meanScore({ task, candidate: memorised })).toBe(0);
  });

  test("pays a rule only on the instances its condition matches", () => {
    const task = taskNamed("clean");
    const narrow = fill({
      task,
      text: "when issue is outage: status page, escalate",
    });
    const scored = evaluate({ task, candidate: narrow, batch: task.testSet });

    const scoredTickets = task.testSet.map((datum, index) => ({
      datum,
      score: scored.scores[index] as number,
    }));
    const paid = scoredTickets
      .filter((entry) => entry.datum.issue === "outage")
      .map((entry) => entry.score);
    const unpaid = scoredTickets
      .filter((entry) => entry.datum.issue !== "outage")
      .map((entry) => entry.score);

    expect(Math.min(...paid)).toBeGreaterThan(0);
    expect(Math.max(...unpaid)).toBe(0);
  });
});

describe("feedback", () => {
  test("names the instance's features and what it missed, not the rule that explains them", () => {
    const task = taskNamed("clean");
    const scored = evaluate({
      task,
      candidate: task.seedCandidate,
      batch: task.testSet,
    });
    const feedback = scored.feedback?.[0] ?? "";

    expect(feedback).toMatch(/tier=\w+ channel=\w+ issue=\w+ region=\w+/);
    expect(feedback).toMatch(/missing:/);
    expect(feedback).not.toMatch(/when \w+ is/);
  });
});

describe("noise", () => {
  test("repeats a reading of the same candidate on the same instance", () => {
    const task = taskNamed("noisy");
    const candidate = policyCandidate(task);

    const first = evaluate({ task, candidate, batch: task.testSet });
    const second = evaluate({ task, candidate, batch: task.testSet });

    expect(second.scores).toEqual(first.scores);
  });

  test("moves a reading away from the noiseless one", () => {
    const clean = taskNamed("clean");
    const noisy = taskNamed("noisy");

    const noiseless = evaluate({
      task: clean,
      candidate: policyCandidate(clean),
      batch: clean.testSet,
    });
    const jittered = evaluate({
      task: noisy,
      candidate: policyCandidate(noisy),
      batch: noisy.testSet,
    });

    expect(jittered.scores).not.toEqual(noiseless.scores);
  });
});

describe("proposal model", () => {
  test("induces the condition a missing action co-occurs with", async () => {
    const reflect = createBenchReflector();
    const proposed = await reflect({
      prompt: reflectionPrompt([
        "tier=enterprise channel=phone issue=outage region=us | missing: sla",
        "tier=enterprise channel=email issue=billing region=eu | missing: sla",
        "tier=enterprise channel=chat issue=howto region=us | missing: sla",
      ]),
    });

    expect(proposed).toMatch(/when tier is enterprise: sla/);
  });

  test("cannot tell which feature explains a single instance", async () => {
    const reflect = createBenchReflector();
    const proposed = await reflect({
      prompt: reflectionPrompt([
        "tier=enterprise channel=phone issue=outage region=us | missing: sla",
      ]),
    });

    expect(proposed).not.toMatch(/when tier is enterprise: sla/);
  });

  test("draws blind when the prompt carries no feedback to induce from", async () => {
    const reflect = createBenchReflector();
    const proposed = await reflect({
      prompt: "<current_instruction>\n\n</current_instruction>",
    });

    expect(proposed).toMatch(/when \w+ is \w+:/);
  });

  test("sends two runs down different parts of the pool", async () => {
    const drawnAt = async (seed: number) => {
      const reflect = createBenchReflector({ seed });
      const drawn: string[] = [];
      for (let call = 0; call < 5; call += 1) {
        drawn.push(
          await reflect({
            prompt: "<current_instruction>\n\n</current_instruction>",
          }),
        );
      }
      return drawn.join("|");
    };

    expect(await drawnAt(1)).not.toBe(await drawnAt(2));
  });

  test("repeats a run's draws, so a rollout can be cached", async () => {
    const drawnAt = async (seed: number) =>
      createBenchReflector({ seed })({
        prompt: "<current_instruction>\n\n</current_instruction>",
      });

    expect(await drawnAt(3)).toBe(await drawnAt(3));
  });

  test("draws mostly rules the policy does not want, so speculation is a gamble", async () => {
    const reflect = createBenchReflector();
    const task = taskNamed("clean");
    const correct = new Set(
      (policyCandidate(task).instruction as string).split("\n"),
    );

    const drawn: string[] = [];
    for (let call = 0; call < 20; call += 1) {
      const proposed = await reflect({
        prompt: "<current_instruction>\n\n</current_instruction>",
      });
      drawn.push(...(proposed.match(/when \w+ is [a-z ]+: [a-z ]+/g) ?? []));
    }
    const hits = drawn.filter((rule) => correct.has(rule));

    expect(drawn.length).toBeGreaterThan(0);
    expect(hits.length / drawn.length).toBeLessThan(0.25);
  });

  test("addresses the components the advice prompt actually names", async () => {
    const advise = createBenchAdviser();
    const proposed = await advise({
      prompt: buildAdvicePrompt({
        components: ["triage", "response"],
        current: { triage: "", response: "" },
        input: { tier: "enterprise", channel: "phone", issue: "outage" },
        worse: { output: "", score: 0 },
      }),
    });

    expect(proposed).toMatch(/<advice component="triage">/);
    expect(proposed).toMatch(/<advice component="response">/);
  });

  test("answers in the advice shape SIMBA asks for", async () => {
    const advise = createBenchAdviser();
    const proposed = await advise({
      prompt: [
        '<component name="instruction"></component>',
        "tier=enterprise channel=phone issue=outage region=us | missing: sla",
        "tier=enterprise channel=email issue=billing region=eu | missing: sla",
      ].join("\n"),
    });

    expect(proposed).toMatch(/<advice component="instruction">/);
    expect(proposed).toMatch(/when tier is enterprise: sla/);
  });
});

describe("demonstrations", () => {
  test("renders an instance's features and its actions, never the ground truth of another", () => {
    const datum: BenchDatum = {
      id: 7,
      tier: "pro",
      channel: "phone",
      issue: "outage",
      region: "us",
    };
    const rendered = renderBenchDemo({
      demo: { input: datum, output: "status page escalate callback" },
      index: 0,
    });

    expect(rendered).toContain("tier=pro");
    expect(rendered).toContain("status page");
    expect(rendered).not.toContain("id");
  });

  test("carries a demo's actions to an instance it resembles", () => {
    const task = taskNamed("demonstrated");
    const near = task.testSet[0] as BenchDatum;
    const demo = renderBenchDemo({
      demo: {
        input: { ...near, id: -1 },
        output: requiredActions(near).join(" "),
      },
      index: 0,
    });

    const withDemo = fill({ task, text: demo });
    const scored = evaluate({ task, candidate: withDemo, batch: [near] });

    expect(scored.scores[0]).toBeGreaterThan(0);
  });

  test("leaves the three rule-shaped tasks nothing to harvest", () => {
    for (const name of ["clean", "interacting"]) {
      const task = taskNamed(name);
      const scored = evaluate({
        task,
        candidate: task.seedCandidate,
        batch: task.trainingSet,
      });

      expect(Math.max(...scored.scores)).toBe(0);
    }
  });

  test("answers some instances unaided on the task demonstrations exist for", () => {
    const task = taskNamed("demonstrated");
    const scored = evaluate({
      task,
      candidate: task.seedCandidate,
      batch: task.trainingSet,
    });

    expect(Math.max(...scored.scores)).toBeGreaterThan(0);
    expect(Math.min(...scored.scores)).toBe(0);
  });
});

describe("interaction", () => {
  test("pays nothing for a downstream rule while the upstream component is empty", () => {
    const task = taskNamed("interacting");
    const ticket = gatedTicket(task);
    const scored = evaluate({
      task,
      candidate: { triage: "", response: "when tier is enterprise: sla" },
      batch: [ticket],
    });

    expect(scored.scores[0]).toBe(0);
  });

  test("pays the same downstream rule once the upstream component is right", () => {
    const task = taskNamed("interacting");
    const ticket = gatedTicket(task);
    const scored = evaluate({
      task,
      candidate: {
        triage: (policyCandidate(task).triage as string) ?? "",
        response: "when tier is enterprise: sla",
      },
      batch: [ticket],
    });

    expect(scored.scores[0]).toBe(1);
  });

  test("pays the upstream component on its own, so the task keeps a gradient", () => {
    const task = taskNamed("interacting");
    const ticket = gatedTicket(task);
    const scored = evaluate({
      task,
      candidate: {
        triage: (policyCandidate(task).triage as string) ?? "",
        response: "",
      },
      batch: [ticket],
    });

    expect(scored.scores[0]).toBeGreaterThan(0);
    expect(scored.scores[0]).toBeLessThan(1);
  });
});

function taskNamed(name: string): BenchTask {
  return benchTasks().find((task) => task.name === name) as BenchTask;
}

function comboOf(datum: BenchDatum): string {
  return `${datum.tier}:${datum.channel}:${datum.issue}:${datum.region}`;
}

function tallyOf(args: {
  data: readonly BenchDatum[];
  feature: "tier" | "channel" | "issue";
}): Record<string, number> {
  const { data, feature } = args;
  const tally: Record<string, number> = {};
  for (const datum of data) {
    tally[datum[feature]] = (tally[datum[feature]] ?? 0) + 1;
  }
  return tally;
}

function valuesOf(args: {
  data: readonly BenchDatum[];
  feature: "tier" | "channel" | "issue";
}): string[] {
  const { data, feature } = args;
  return [...new Set(data.map((datum) => datum[feature]))].sort();
}

function fill(args: { task: BenchTask; text: string }): Candidate {
  const { task, text } = args;
  return Object.fromEntries(
    Object.keys(task.seedCandidate).map((component) => [component, text]),
  );
}

function evaluate(args: {
  task: BenchTask;
  candidate: Candidate;
  batch: readonly BenchDatum[];
}) {
  const { task, candidate, batch } = args;
  return task.adapter.evaluate({
    batch,
    candidate,
    captureTraces: false,
    run: RUN,
  }) as { scores: number[]; feedback?: string[] };
}

function meanScore(args: { task: BenchTask; candidate: Candidate }): number {
  const { task, candidate } = args;
  return meanScoreOn({ task, candidate, batch: task.testSet });
}

function meanScoreOn(args: {
  task: BenchTask;
  candidate: Candidate;
  batch: readonly BenchDatum[];
}): number {
  const { task, candidate, batch } = args;
  const scored = evaluate({ task, candidate, batch });
  return (
    scored.scores.reduce((total, score) => total + score, 0) /
    scored.scores.length
  );
}

function reflectionPrompt(feedback: readonly string[]): string {
  return [
    "<current_instruction>",
    "",
    "</current_instruction>",
    ...feedback.map((line) => `<example>${line}</example>`),
  ].join("\n");
}

/** A ticket needing an action from each component: one gated, one upstream. */
function gatedTicket(task: BenchTask): BenchDatum {
  return task.testSet.find(
    (datum) => datum.tier === "enterprise" && datum.issue !== "outage",
  ) as BenchDatum;
}
