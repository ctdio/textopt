import { describe, expect, test } from "vitest";
import { buildAdvicePrompt, parseAdvice } from "./advice.js";

describe("buildAdvicePrompt", () => {
  test("names every component the advice is wanted for", () => {
    const prompt = buildAdvicePrompt({
      components: ["planner", "writer"],
      input: { question: "How do I reset a device?" },
      better: {
        output: "hold ten seconds",
        score: 1,
        feedback: "All present.",
      },
      worse: { output: "turn it off", score: 0, feedback: "Missing: hold." },
    });

    expect(prompt).toContain("planner");
    expect(prompt).toContain("writer");
  });

  test("shows both trajectories with their rewards", () => {
    const prompt = buildAdvicePrompt({
      components: ["writer"],
      input: { question: "How do I reset a device?" },
      better: {
        output: "hold ten seconds",
        score: 1,
        feedback: "All present.",
      },
      worse: { output: "turn it off", score: 0, feedback: "Missing: hold." },
    });

    expect(prompt).toContain("hold ten seconds");
    expect(prompt).toContain("turn it off");
    expect(prompt).toContain("Missing: hold.");
  });

  test("omits the better trajectory when there is none to contrast", () => {
    const prompt = buildAdvicePrompt({
      components: ["writer"],
      input: { question: "How do I reset a device?" },
      worse: { output: "turn it off", score: 0, feedback: "Missing: hold." },
    });

    expect(prompt).not.toContain("<better_trajectory>");
    expect(prompt).toContain("<worse_trajectory>");
  });
});

describe("parseAdvice", () => {
  test("reads one entry per component", () => {
    const advice = parseAdvice(
      '<advice component="writer">Say hold.</advice>\n<advice component="planner">Plan first.</advice>',
    );

    expect(advice).toEqual({ writer: "Say hold.", planner: "Plan first." });
  });

  test("keeps line breaks inside an entry", () => {
    const advice = parseAdvice(
      '<advice component="writer">Say hold.\nThen say ten seconds.</advice>',
    );

    expect(advice.writer).toBe("Say hold.\nThen say ten seconds.");
  });

  test("ignores prose written around the entries", () => {
    const advice = parseAdvice(
      'Here is my analysis.\n<advice component="writer">Say hold.</advice>\nThat is all.',
    );

    expect(advice).toEqual({ writer: "Say hold." });
  });

  test("returns nothing when the response carries no entries", () => {
    expect(parseAdvice("I have no suggestions.")).toEqual({});
  });

  test("drops an entry whose advice is empty", () => {
    expect(parseAdvice('<advice component="writer">   </advice>')).toEqual({});
  });
});
