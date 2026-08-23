import { describe, expect, test } from "vitest";
import {
  failureWarnings,
  resolveValidationSet,
  seedScoreWarnings,
} from "./warnings.js";

const TRAINING = [{ id: "a" }, { id: "b" }];

describe("resolveValidationSet", () => {
  test("warns when selection falls back onto the instances reflection reads", () => {
    const resolved = resolveValidationSet({
      validationSet: undefined,
      trainingSet: TRAINING,
    });

    expect(resolved.validationSet).toBe(TRAINING);
    expect(resolved.warnings.map((warning) => warning.code)).toEqual([
      "validationSetReusesTraining",
    ]);
  });

  test("stays quiet when the caller asked for the reuse by name", () => {
    const resolved = resolveValidationSet({
      validationSet: "reuseTraining",
      trainingSet: TRAINING,
    });

    expect(resolved.validationSet).toBe(TRAINING);
    expect(resolved.warnings).toEqual([]);
  });

  test("stays quiet when a separate validation set was given", () => {
    const held = [{ id: "c" }];

    const resolved = resolveValidationSet({
      validationSet: held,
      trainingSet: TRAINING,
    });

    expect(resolved.validationSet).toBe(held);
    expect(resolved.warnings).toEqual([]);
  });
});

describe("seedScoreWarnings", () => {
  test("warns when the seed already scores perfectly everywhere", () => {
    const warnings = seedScoreWarnings({
      scores: [1, 1, 1],
      perfectScore: 1,
    });

    expect(warnings.map((warning) => warning.code)).toEqual([
      "seedScoreSaturated",
    ]);
  });

  test("warns when the seed scores zero everywhere", () => {
    const warnings = seedScoreWarnings({ scores: [0, 0, 0], perfectScore: 1 });

    expect(warnings.map((warning) => warning.code)).toEqual([
      "seedScoreFloored",
    ]);
  });

  test("stays quiet when the seed separates the instances", () => {
    const warnings = seedScoreWarnings({
      scores: [0, 0.5, 1],
      perfectScore: 1,
    });

    expect(warnings).toEqual([]);
  });

  test("reads only the instances that were measured", () => {
    // An unmeasured instance is unknown, not zero, and a row of holes is a
    // transient failure to report rather than a floored seed.
    const warnings = seedScoreWarnings({
      scores: [undefined, 0.5, undefined],
      perfectScore: 1,
    });

    expect(warnings).toEqual([]);
  });

  test("stays quiet when nothing was measured at all", () => {
    const warnings = seedScoreWarnings({
      scores: [undefined, undefined],
      perfectScore: 1,
    });

    expect(warnings).toEqual([]);
  });
});

describe("failureWarnings", () => {
  test("warns when failures were scored against the candidate unclassified", () => {
    const warnings = failureWarnings({ unclassified: 14 });

    expect(warnings.map((warning) => warning.code)).toEqual([
      "unclassifiedFailures",
    ]);
    expect(warnings[0]?.message).toContain("14");
  });

  test("stays quiet when nothing failed", () => {
    expect(failureWarnings({ unclassified: 0 })).toEqual([]);
  });
});
