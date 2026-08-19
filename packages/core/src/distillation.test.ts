import { describe, expect, test } from "vitest";
import { toTrainingJsonl } from "./distillation.js";
import type { Rollout } from "./harvest.js";

const ROLLOUTS: Rollout<{ question: string }, string>[] = [
  {
    input: { question: "How do I reset a device?" },
    output: "Hold for ten seconds.",
    score: 1,
  },
  {
    input: { question: "What is the refund window?" },
    output: "Thirty days.",
    score: 1,
  },
];

const render = ({ rollout }: { rollout: (typeof ROLLOUTS)[number] }) => ({
  messages: [
    { role: "user" as const, content: rollout.input.question },
    { role: "assistant" as const, content: rollout.output },
  ],
});

describe("toTrainingJsonl", () => {
  test("writes one example per line", () => {
    const jsonl = toTrainingJsonl({ rollouts: ROLLOUTS, render });

    expect(jsonl.split("\n")).toHaveLength(2);
  });

  test("writes each line as the example the renderer returned", () => {
    const jsonl = toTrainingJsonl({ rollouts: ROLLOUTS, render });

    expect(JSON.parse(jsonl.split("\n")[0] as string)).toEqual({
      messages: [
        { role: "user", content: "How do I reset a device?" },
        { role: "assistant", content: "Hold for ten seconds." },
      ],
    });
  });

  test("returns an empty string for no rollouts", () => {
    expect(toTrainingJsonl({ rollouts: [], render })).toBe("");
  });

  test("skips rollouts the renderer declines", () => {
    const jsonl = toTrainingJsonl({
      rollouts: ROLLOUTS,
      render: (args) => (args.index === 0 ? null : render(args)),
    });

    expect(jsonl.split("\n")).toHaveLength(1);
  });

  test("rejects an example carrying no messages", () => {
    expect(() =>
      toTrainingJsonl({ rollouts: ROLLOUTS, render: () => ({ messages: [] }) }),
    ).toThrow(/messages/);
  });

  test("names the rollout an unusable example came from", () => {
    expect(() =>
      toTrainingJsonl({
        rollouts: ROLLOUTS,
        render: (args) => (args.index === 1 ? { messages: [] } : render(args)),
      }),
    ).toThrow(/1/);
  });
});
