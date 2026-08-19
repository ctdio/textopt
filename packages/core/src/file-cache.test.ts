import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { createFileCache } from "./file-cache.js";

let directory: string;
let path: string;

beforeEach(() => {
  directory = mkdtempSync(join(tmpdir(), "textopt-cache-"));
  path = join(directory, "scores.jsonl");
});

afterEach(() => {
  rmSync(directory, { recursive: true, force: true });
});

describe("createFileCache", () => {
  test("serves an entry a previous process wrote", () => {
    createFileCache({ path }).set("val:abc:0", { score: 0.5 });

    expect(createFileCache({ path }).get("val:abc:0")).toEqual({ score: 0.5 });
  });

  test("keeps per-objective scores alongside the metric", () => {
    createFileCache({ path }).set("val:abc:0", {
      score: 0.5,
      objectiveScores: { accuracy: 1, brevity: 0 },
    });

    expect(createFileCache({ path }).get("val:abc:0")).toEqual({
      score: 0.5,
      objectiveScores: { accuracy: 1, brevity: 0 },
    });
  });

  test("serves the newest value written for a key", () => {
    const cache = createFileCache({ path });

    cache.set("val:abc:0", { score: 0.1 });
    cache.set("val:abc:0", { score: 0.9 });

    expect(createFileCache({ path }).get("val:abc:0")).toEqual({ score: 0.9 });
  });

  test("reads what it can from a file a killed process truncated", () => {
    const cache = createFileCache({ path });
    cache.set("val:abc:0", { score: 0.5 });
    cache.set("val:abc:1", { score: 0.25 });
    const written = readFileSync(path, "utf8");
    writeFileSync(path, written.slice(0, written.length - 12));

    const reopened = createFileCache({ path });

    expect(reopened.get("val:abc:0")).toEqual({ score: 0.5 });
    expect(reopened.get("val:abc:1")).toBeUndefined();
  });

  test("offers no entries to a snapshot, because the file already holds them", () => {
    // Checkpoints exist to make scores survive a crash. This cache already
    // does, so copying them into every snapshot would only make it larger.
    expect(createFileCache({ path }).entries).toBeUndefined();
  });
});
