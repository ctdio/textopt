import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { jsonlReporter } from "./file-reporter.js";
import type { ReportableEvent } from "./reporting.js";

const ACCEPTED: ReportableEvent = {
  type: "candidateAccepted",
  candidateId: 3,
  candidate: { instruction: "answer well" },
  aggregateScore: 0.77,
  instanceScores: [0.5, 1],
};

const ROLLOUT: ReportableEvent = {
  type: "rollout",
  iteration: 0,
  phase: "validation",
  split: "val",
  candidateId: 3,
  completed: 1,
  total: 2,
};

let directory: string;
let path: string;

beforeEach(() => {
  directory = mkdtempSync(join(tmpdir(), "textopt-events-"));
  path = join(directory, "run.jsonl");
});

afterEach(() => {
  rmSync(directory, { recursive: true, force: true });
});

function lines(): unknown[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line) as unknown);
}

describe("jsonlReporter", () => {
  test("writes one line per event as the run makes it", () => {
    const reporter = jsonlReporter({ path });

    reporter.onEvent?.(ACCEPTED);

    expect(lines()).toEqual([ACCEPTED]);
  });

  test("appends to the events an earlier run left in the file", () => {
    jsonlReporter({ path }).onEvent?.(ACCEPTED);
    jsonlReporter({ path }).onEvent?.(ACCEPTED);

    expect(lines()).toHaveLength(2);
  });

  test("writes only the events it was asked for", () => {
    const reporter = jsonlReporter({ path, only: ["candidateAccepted"] });

    reporter.onEvent?.(ROLLOUT);
    reporter.onEvent?.(ACCEPTED);

    expect(lines()).toEqual([ACCEPTED]);
  });

  test("keeps an event whose payload will not serialize", () => {
    const cycle: Record<string, unknown> = { type: "error", iteration: 1 };
    cycle.self = cycle;

    jsonlReporter({ path }).onEvent?.(cycle as { type: string });

    expect(lines()).toEqual([{ type: "error", unserializable: true }]);
  });
});
