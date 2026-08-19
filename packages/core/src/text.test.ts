import { describe, expect, test } from "vitest";
import { parseProposedText } from "./text.js";

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
