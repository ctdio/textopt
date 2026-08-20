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
