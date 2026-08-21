import { describe, expect, test } from "vitest";
import { createJudge } from "./judge.js";
import type { TextModel } from "./types.js";

const CRITERIA = [
  { name: "accuracy", description: "Every claim is supported by the input." },
  { name: "tone", description: "Reads as a support agent, not a robot." },
];

describe("createJudge", () => {
  test("normalizes the judge's scale into a 0..1 score", async () => {
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">5</score>\n<feedback>Fine.</feedback>',
      ),
      criteria: [CRITERIA[0] as (typeof CRITERIA)[number]],
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.score).toBe(1);
  });

  test("averages the criteria and keeps each one as an objective", async () => {
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">4</score>\n<score name="tone">2</score>\n<feedback>Warmer.</feedback>',
      ),
      criteria: CRITERIA,
      scale: 4,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.score).toBe(0.75);
    expect(verdict.objectiveScores).toEqual({ accuracy: 1, tone: 0.5 });
  });

  test("returns the judge's written feedback for reflection to read", async () => {
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">1</score>\n<feedback>State the refund window.</feedback>',
      ),
      criteria: [CRITERIA[0] as (typeof CRITERIA)[number]],
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.feedback).toBe("State the refund window.");
  });

  test("marks a response missing a criterion as transient rather than zero", async () => {
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">3</score>\n<feedback>ok</feedback>',
      ),
      criteria: CRITERIA,
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.transient).toBe(true);
    expect(verdict.score).toBe(0);
  });

  test("marks an unparseable response as transient", async () => {
    const judge = createJudge({
      model: replying("I would rather not grade this."),
      criteria: CRITERIA,
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.transient).toBe(true);
  });

  test("clamps a score the judge pushed past its own scale", async () => {
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">9</score>\n<feedback>ok</feedback>',
      ),
      criteria: [CRITERIA[0] as (typeof CRITERIA)[number]],
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.score).toBe(1);
  });

  test("refuses a scale that cannot normalize a grade", () => {
    // Every grade is divided by it, so a zero or negative scale turns each
    // verdict into a non-finite number or a negative one, and the search then
    // ranks candidates by it.
    expect(() =>
      createJudge({ model: replying(""), criteria: CRITERIA, scale: 0 }),
    ).toThrow(/scale/);
  });

  test("shows the judge the input, the output and the expected answer", async () => {
    let seen = "";
    const judge = createJudge({
      model: async ({ prompt }) => {
        seen = prompt;
        return '<score name="accuracy">5</score>\n<feedback>ok</feedback>';
      },
      criteria: [CRITERIA[0] as (typeof CRITERIA)[number]],
    });

    await judge({
      input: "How long is the hold?",
      output: "About a minute.",
      expected: "Ten seconds.",
    });

    expect(seen).toContain("How long is the hold?");
    expect(seen).toContain("About a minute.");
    expect(seen).toContain("Ten seconds.");
    expect(seen).toContain("Every claim is supported by the input.");
  });

  test("weights a criterion the caller cares more about", async () => {
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">4</score>\n<score name="tone">0</score>\n<feedback>Warmer.</feedback>',
      ),
      criteria: [
        { ...(CRITERIA[0] as (typeof CRITERIA)[number]), weight: 3 },
        { ...(CRITERIA[1] as (typeof CRITERIA)[number]), weight: 1 },
      ],
      scale: 4,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.score).toBe(0.75);
  });

  test("zeroes the instance when a gated criterion falls below its bar", async () => {
    // An unweighted mean lets a search trade a hard requirement away: three
    // style criteria at full marks outrank an incumbent that kept the one
    // rule that was not negotiable. A gate is not a term in the average.
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">1</score>\n<score name="tone">5</score>\n<feedback>Cite the policy.</feedback>',
      ),
      criteria: [
        { ...(CRITERIA[0] as (typeof CRITERIA)[number]), gate: 3 },
        CRITERIA[1] as (typeof CRITERIA)[number],
      ],
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.score).toBe(0);
  });

  test("keeps every criterion as an objective when a gate zeroes the score", async () => {
    // The aggregate is what selection reads; the objectives are what says
    // which criterion did the zeroing, and reflection needs both.
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">1</score>\n<score name="tone">5</score>\n<feedback>Cite the policy.</feedback>',
      ),
      criteria: [
        { ...(CRITERIA[0] as (typeof CRITERIA)[number]), gate: 3 },
        CRITERIA[1] as (typeof CRITERIA)[number],
      ],
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.objectiveScores).toEqual({ accuracy: 0.2, tone: 1 });
  });

  test("scores normally when a gated criterion clears its bar", async () => {
    const judge = createJudge({
      model: replying(
        '<score name="accuracy">3</score>\n<score name="tone">5</score>\n<feedback>ok</feedback>',
      ),
      criteria: [
        { ...(CRITERIA[0] as (typeof CRITERIA)[number]), gate: 3 },
        CRITERIA[1] as (typeof CRITERIA)[number],
      ],
      scale: 5,
    });

    const verdict = await judge({ input: "q", output: "a" });

    expect(verdict.score).toBe(0.8);
  });

  test("refuses a gate no grade could ever fail", () => {
    expect(() =>
      createJudge({
        model: replying(""),
        criteria: [{ ...(CRITERIA[0] as (typeof CRITERIA)[number]), gate: 0 }],
        scale: 5,
      }),
    ).toThrow(/gate/);
  });

  test("refuses a gate no grade could ever clear", () => {
    expect(() =>
      createJudge({
        model: replying(""),
        criteria: [{ ...(CRITERIA[0] as (typeof CRITERIA)[number]), gate: 6 }],
        scale: 5,
      }),
    ).toThrow(/gate/);
  });

  test("refuses criteria whose weights sum to nothing", () => {
    expect(() =>
      createJudge({
        model: replying(""),
        criteria: [
          { ...(CRITERIA[0] as (typeof CRITERIA)[number]), weight: 0 },
        ],
      }),
    ).toThrow(/weight/);
  });

  test("refuses a negative weight", () => {
    expect(() =>
      createJudge({
        model: replying(""),
        criteria: [
          { ...(CRITERIA[0] as (typeof CRITERIA)[number]), weight: -1 },
          CRITERIA[1] as (typeof CRITERIA)[number],
        ],
      }),
    ).toThrow(/weight/);
  });

  test("forbids the feedback from restating the expected answer", async () => {
    // The feedback is written into a reusable instruction. "The instruction
    // never says to state the thirty-day refund window" is addressed to the
    // instructions, as asked, and copies the answer key into the prompt: the
    // validation score climbs and nothing generalizes.
    let seen = "";
    const judge = createJudge({
      model: async ({ prompt }) => {
        seen = prompt;
        return '<score name="accuracy">5</score>\n<feedback>ok</feedback>';
      },
      criteria: [CRITERIA[0] as (typeof CRITERIA)[number]],
    });

    await judge({ input: "q", output: "a", expected: "Thirty days." });

    expect(seen).toContain("expected_answer");
    expect(seen).toMatch(/do not restate|must not restate/i);
  });

  test("says nothing about an expected answer when there is none", async () => {
    let seen = "";
    const judge = createJudge({
      model: async ({ prompt }) => {
        seen = prompt;
        return '<score name="accuracy">5</score>\n<feedback>ok</feedback>';
      },
      criteria: [CRITERIA[0] as (typeof CRITERIA)[number]],
    });

    await judge({ input: "q", output: "a" });

    expect(seen).not.toMatch(/restate/i);
  });

  test("asks for feedback about the instructions rather than the output", async () => {
    let seen = "";
    const judge = createJudge({
      model: async ({ prompt }) => {
        seen = prompt;
        return '<score name="accuracy">5</score>\n<feedback>ok</feedback>';
      },
      criteria: [CRITERIA[0] as (typeof CRITERIA)[number]],
    });

    await judge({ input: "q", output: "a" });

    expect(seen).toContain("instructions");
  });
});

function replying(response: string): TextModel {
  return async () => response;
}
