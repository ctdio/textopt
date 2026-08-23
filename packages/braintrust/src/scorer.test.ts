import { ExactMatch, Levenshtein } from "autoevals";
import { describe, expect, test } from "vitest";
import { createBraintrustScorer } from "./scorer.js";

describe("createBraintrustScorer", () => {
  test("runs a real autoevals scorer and returns its score", async () => {
    const score = createBraintrustScorer({ scorers: [Levenshtein] });

    const result = await score({ output: "hello", expected: "hello" });

    expect(result.score).toBe(1);
  });

  test("exposes each scorer's score as a named objective", async () => {
    const score = createBraintrustScorer({
      scorers: [Levenshtein, ExactMatch],
    });

    const result = await score({ output: "hello", expected: "hallo" });

    expect(Object.keys(result.objectiveScores ?? {}).sort()).toEqual([
      "ExactMatch",
      "Levenshtein",
    ]);
    expect(result.objectiveScores?.ExactMatch).toBe(0);
  });

  test("averages across scorers", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "a", score: 1 }),
        () => ({ name: "b", score: 0 }),
      ],
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(0.5);
  });

  test("applies per-scorer weights", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "a", score: 1 }),
        () => ({ name: "b", score: 0 }),
      ],
      weights: { a: 3, b: 1 },
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(0.75);
  });

  test("ignores scorers that return a null score", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "a", score: 1 }),
        () => ({ name: "skipped", score: null }),
      ],
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(1);
    expect(result.objectiveScores).not.toHaveProperty("skipped");
  });

  test("accepts a scorer that returns a bare number", async () => {
    const score = createBraintrustScorer({ scorers: [() => 0.25] });

    const result = await score({ output: "x" });

    expect(result.score).toBe(0.25);
  });

  test("builds feedback from scorer names and rationales", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => ({
          name: "Factuality",
          score: 0.2,
          metadata: { rationale: "The answer invented a statistic." },
        }),
      ],
    });

    const result = await score({ output: "x" });

    expect(result.feedback).toContain("Factuality");
    expect(result.feedback).toContain("The answer invented a statistic.");
  });

  test("turns a thrown scorer into feedback instead of failing the rollout", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => {
          throw new Error("judge unavailable");
        },
        () => ({ name: "b", score: 1 }),
      ],
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(1);
    expect(result.feedback).toContain("judge unavailable");
  });

  test("scores zero when every scorer fails", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => {
          throw new Error("down");
        },
      ],
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(0);
  });

  test("marks the result transient when every scorer failed on infrastructure", async () => {
    // A rate limited judge is not the candidate's fault. Without the flag the
    // engine caches this zero against the candidate permanently.
    const score = createBraintrustScorer({
      scorers: [
        () => {
          throw new Error("429 rate limited");
        },
      ],
      isTransient: (err) => String(err).includes("429"),
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(0);
    expect(result.transient).toBe(true);
  });

  test("marks the result transient when only some scorers failed on infrastructure", async () => {
    // The composite is now computed from the survivors alone, and the failed
    // scorer's objective has vanished from the breakdown. Caching that as the
    // candidate's score records a number that was never measured.
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "a", score: 1 }),
        () => {
          throw new Error("429 rate limited");
        },
      ],
      isTransient: (err) => String(err).includes("429"),
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(1);
    expect(result.transient).toBe(true);
  });

  test("leaves a scorer failure the caller does not classify as the candidate's", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => {
          throw new Error("the output could not be parsed");
        },
      ],
    });

    const result = await score({ output: "x" });

    expect(result.transient).toBeUndefined();
  });

  test("reports a scorer failure as a failure whether or not it was classified", async () => {
    // Unclassified means "not worth retrying", not "worth remembering": the
    // zero below is a stand-in for a score nothing measured.
    const score = createBraintrustScorer({
      scorers: [
        () => {
          throw new Error("the output could not be parsed");
        },
      ],
    });

    const result = await score({ output: "x" });

    expect(result.failed).toBe(true);
    expect(result.transient).toBeUndefined();
  });

  test("reports a partly failed composite as a failure", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "a", score: 1 }),
        () => {
          throw new Error("the output could not be parsed");
        },
      ],
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(1);
    expect(result.failed).toBe(true);
  });

  test("leaves failed unset when every scorer produced a score", async () => {
    const score = createBraintrustScorer({
      scorers: [() => ({ name: "a", score: 1 })],
    });

    const result = await score({ output: "x" });

    expect(result.failed).toBeUndefined();
  });

  test("rejects a negative weight", () => {
    expect(() =>
      createBraintrustScorer({
        scorers: [() => ({ name: "a", score: 1 })],
        weights: { a: -2 },
      }),
    ).toThrow(/weight/i);
  });

  test("rejects a non-finite weight", () => {
    expect(() =>
      createBraintrustScorer({
        scorers: [() => ({ name: "a", score: 1 })],
        weights: { a: Number.NaN },
      }),
    ).toThrow(/weight/i);
  });

  test("rejects a weight configuration that zeroes out every scorer", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "a", score: 1 }),
        () => ({ name: "b", score: 1 }),
      ],
      weights: { a: 0, b: 0 },
    });

    await expect(score({ output: "x" })).rejects.toThrow(/weight/i);
  });

  test("rejects two scorers reporting the same name", async () => {
    // The composite would count both while objectiveScores kept only the
    // last, and `weights.accuracy` would be ambiguous between them.
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "accuracy", score: 1 }),
        () => ({ name: "accuracy", score: 0 }),
      ],
    });

    await expect(score({ output: "x" })).rejects.toThrow(/accuracy/);
  });

  test("still scores when a duplicate name comes from a scorer that failed", async () => {
    const score = createBraintrustScorer({
      scorers: [
        () => ({ name: "accuracy", score: 1 }),
        () => ({ name: "accuracy", score: null }),
      ],
    });

    const result = await score({ output: "x" });

    expect(result.score).toBe(1);
    expect(result.objectiveScores).toEqual({ accuracy: 1 });
  });
});
