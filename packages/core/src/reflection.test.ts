import { describe, expect, test } from "vitest";
import {
  buildReflectionPrompt,
  createDefaultProposer,
  limitReflectiveRecords,
  parseProposedText,
} from "./reflection.js";
import type { ReflectiveRecord } from "./types.js";

describe("buildReflectionPrompt", () => {
  test("includes the component name and its current text", () => {
    const prompt = buildReflectionPrompt({
      componentName: "classifier",
      currentText: "Classify the ticket.",
      records: [],
    });

    expect(prompt).toContain("classifier");
    expect(prompt).toContain("Classify the ticket.");
  });

  test("serializes each reflective record into the prompt", () => {
    const prompt = buildReflectionPrompt({
      componentName: "classifier",
      currentText: "Classify the ticket.",
      records: [
        {
          inputs: { ticket: "printer on fire" },
          generatedOutputs: "billing",
          feedback: "Expected hardware, got billing.",
        },
      ],
    });

    expect(prompt).toContain("printer on fire");
    expect(prompt).toContain("Expected hardware, got billing.");
  });

  test("lists instructions that were already tried and rejected", () => {
    const prompt = buildReflectionPrompt({
      componentName: "classifier",
      currentText: "Classify the ticket.",
      records: [],
      rejected: [
        { text: "Guess the category.", parentScore: 0.5, childScore: 0.1 },
      ],
    });

    expect(prompt).toContain("Guess the category.");
    expect(prompt).toContain("<rejected_instructions>");
  });

  test("says nothing about rejected instructions when there are none", () => {
    const prompt = buildReflectionPrompt({
      componentName: "classifier",
      currentText: "Classify the ticket.",
      records: [],
    });

    expect(prompt).not.toContain("<rejected_instructions>");
  });
});

describe("parseProposedText", () => {
  test("extracts text from a fenced block", () => {
    expect(parseProposedText("Here you go:\n```\nNew instruction\n```")).toBe(
      "New instruction",
    );
  });

  test("ignores a language tag on the fence", () => {
    expect(parseProposedText("```text\nNew instruction\n```")).toBe(
      "New instruction",
    );
  });

  test("keeps nested fences intact instead of truncating to the last block", () => {
    const response =
      "```\nRun this:\n```python\nprint(1)\n```\nThen explain it.\n```";

    expect(parseProposedText(response)).toBe(
      "Run this:\n```python\nprint(1)\n```\nThen explain it.",
    );
  });

  test("strips a dangling opening fence from a truncated response", () => {
    expect(parseProposedText("```\nNew instruction, cut off mid")).toBe(
      "New instruction, cut off mid",
    );
  });

  test("strips a dangling opening fence carrying a language tag", () => {
    expect(parseProposedText("```markdown\nNew instruction, cut off mid")).toBe(
      "New instruction, cut off mid",
    );
  });

  test("strips a stray closing fence when the opening one is missing", () => {
    expect(parseProposedText("New instruction\n```")).toBe("New instruction");
  });

  test("falls back to the trimmed response when unfenced", () => {
    expect(parseProposedText("  New instruction  ")).toBe("New instruction");
  });

  test("preserves internal newlines", () => {
    expect(parseProposedText("```\nline one\nline two\n```")).toBe(
      "line one\nline two",
    );
  });
});

describe("createDefaultProposer", () => {
  test("returns new text for each requested component", async () => {
    const propose = createDefaultProposer();

    const proposed = await propose({
      candidate: { alpha: "old alpha", beta: "old beta" },
      reflectiveDataset: {
        alpha: [
          { inputs: {}, generatedOutputs: "", feedback: "be more specific" },
        ],
      },
      componentsToUpdate: ["alpha"],
      reflect: async () => "```\nnew alpha\n```",
    });

    expect(proposed).toEqual({ alpha: "new alpha" });
  });

  test("skips components with no reflective records", async () => {
    const propose = createDefaultProposer();

    const proposed = await propose({
      candidate: { alpha: "old alpha" },
      reflectiveDataset: {},
      componentsToUpdate: ["alpha"],
      reflect: async () => "```\nnew alpha\n```",
    });

    expect(proposed).toEqual({});
  });

  test("drops a proposal identical to the current text", async () => {
    const propose = createDefaultProposer();

    const proposed = await propose({
      candidate: { alpha: "old alpha" },
      reflectiveDataset: {
        alpha: [{ inputs: {}, generatedOutputs: "", feedback: "fine" }],
      },
      componentsToUpdate: ["alpha"],
      reflect: async () => "```\nold alpha\n```",
    });

    expect(proposed).toEqual({});
  });

  test("drops an empty proposal", async () => {
    const propose = createDefaultProposer();

    const proposed = await propose({
      candidate: { alpha: "old alpha" },
      reflectiveDataset: {
        alpha: [{ inputs: {}, generatedOutputs: "", feedback: "fine" }],
      },
      componentsToUpdate: ["alpha"],
      reflect: async () => "```\n\n```",
    });

    expect(proposed).toEqual({});
  });

  test("shows the reflection model the proposals already rejected", async () => {
    const propose = createDefaultProposer();
    const prompts: string[] = [];

    await propose({
      candidate: { alpha: "old alpha" },
      reflectiveDataset: {
        alpha: [{ inputs: {}, generatedOutputs: "", feedback: "be specific" }],
      },
      componentsToUpdate: ["alpha"],
      rejectedProposals: {
        alpha: [{ text: "tried alpha", parentScore: 1, childScore: 0 }],
      },
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nnew alpha\n```";
      },
    });

    expect(prompts[0]).toContain("tried alpha");
  });

  test("uses a supplied prompt builder instead of the default template", async () => {
    const propose = createDefaultProposer({
      buildPrompt: ({ componentName, currentText }) =>
        `rewrite ${componentName}: ${currentText}`,
    });
    const prompts: string[] = [];

    await propose({
      candidate: { alpha: "old alpha" },
      reflectiveDataset: {
        alpha: [{ inputs: {}, generatedOutputs: "", feedback: "be specific" }],
      },
      componentsToUpdate: ["alpha"],
      reflect: async ({ prompt }) => {
        prompts.push(prompt);
        return "```\nnew alpha\n```";
      },
    });

    expect(prompts).toEqual(["rewrite alpha: old alpha"]);
  });

  test("applies record limits before the prompt builder sees them", async () => {
    const propose = createDefaultProposer({ limits: { maxRecords: 1 } });
    const seen: number[] = [];

    await propose({
      candidate: { alpha: "old alpha" },
      reflectiveDataset: {
        alpha: [
          { inputs: {}, generatedOutputs: "", feedback: "a", score: 1 },
          { inputs: {}, generatedOutputs: "", feedback: "b", score: 0 },
        ],
      },
      componentsToUpdate: ["alpha"],
      reflect: async ({ prompt }) => {
        seen.push((prompt.match(/"feedback"/g) ?? []).length);
        return "```\nnew alpha\n```";
      },
    });

    expect(seen).toEqual([1]);
  });
});

describe("limitReflectiveRecords", () => {
  const records: ReflectiveRecord[] = [
    { inputs: "one", generatedOutputs: "", feedback: "perfect", score: 1 },
    { inputs: "two", generatedOutputs: "", feedback: "wrong", score: 0 },
    { inputs: "three", generatedOutputs: "", feedback: "partial", score: 0.5 },
  ];

  test("returns every record when no limit is set", () => {
    expect(limitReflectiveRecords({ records })).toEqual(records);
  });

  test("keeps the worst scoring records", () => {
    const limited = limitReflectiveRecords({ records, maxRecords: 2 });

    expect(limited.map((record) => record.inputs)).toEqual(["two", "three"]);
  });

  test("keeps the records in their original order", () => {
    const limited = limitReflectiveRecords({
      records: [records[1] as ReflectiveRecord, records[0] as ReflectiveRecord],
      maxRecords: 2,
    });

    expect(limited.map((record) => record.inputs)).toEqual(["two", "one"]);
  });

  test("prefers scored failures over records with no score at all", () => {
    const limited = limitReflectiveRecords({
      records: [
        { inputs: "unscored", generatedOutputs: "", feedback: "" },
        { inputs: "failure", generatedOutputs: "", feedback: "", score: 0 },
      ],
      maxRecords: 1,
    });

    expect(limited.map((record) => record.inputs)).toEqual(["failure"]);
  });

  test("truncates a string longer than the per-record character share", () => {
    const limited = limitReflectiveRecords({
      records: [
        { inputs: "x".repeat(5000), generatedOutputs: "", feedback: "" },
        { inputs: "short", generatedOutputs: "", feedback: "" },
      ],
      maxCharacters: 600,
    });

    expect(String(limited[0]?.inputs).length).toBeLessThan(400);
    expect(String(limited[0]?.inputs)).toContain("truncated");
    expect(limited[1]?.inputs).toBe("short");
  });

  test("truncates strings nested inside a record", () => {
    const limited = limitReflectiveRecords({
      records: [
        {
          inputs: { trace: [{ output: "y".repeat(500) }] },
          generatedOutputs: "",
          feedback: "",
        },
      ],
      maxCharacters: 100,
    });

    const trace = (limited[0]?.inputs as { trace: { output: string }[] }).trace;
    expect(trace[0]?.output.length).toBeLessThan(120);
  });

  test("drops trailing records when many small ones still overflow", () => {
    const limited = limitReflectiveRecords({
      records: Array.from({ length: 40 }, (_, index) => ({
        inputs: `record ${index}`,
        generatedOutputs: "",
        feedback: "",
      })),
      maxCharacters: 400,
    });

    expect(limited.length).toBeLessThan(40);
    expect(limited.length).toBeGreaterThan(0);
    expect(limited[0]?.inputs).toBe("record 0");
  });

  test("keeps one record even when it cannot fit", () => {
    const limited = limitReflectiveRecords({
      records: [{ inputs: "z".repeat(50), generatedOutputs: "", feedback: "" }],
      maxCharacters: 10,
    });

    expect(limited).toHaveLength(1);
  });
});
