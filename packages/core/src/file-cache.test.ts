import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { createFileCache } from "./file-cache.js";

const NAMESPACE = "gpt-4o-mini@t0/scorer-v3";

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
    createFileCache({ path, namespace: NAMESPACE }).set("val:abc:0", {
      score: 0.5,
    });

    expect(
      createFileCache({ path, namespace: NAMESPACE }).get("val:abc:0"),
    ).toEqual({ score: 0.5 });
  });

  test("keeps per-objective scores alongside the metric", () => {
    createFileCache({ path, namespace: NAMESPACE }).set("val:abc:0", {
      score: 0.5,
      objectiveScores: { accuracy: 1, brevity: 0 },
    });

    expect(
      createFileCache({ path, namespace: NAMESPACE }).get("val:abc:0"),
    ).toEqual({
      score: 0.5,
      objectiveScores: { accuracy: 1, brevity: 0 },
    });
  });

  test("refuses a score measured under a different system", () => {
    // A durable log outlives the model, the decoding settings and the scorer
    // that produced its entries. Serving one of those across a change is a run
    // reading measurements of a system it is no longer running, and nothing in
    // the result says it happened.
    createFileCache({ path, namespace: NAMESPACE }).set("val:abc:0", {
      score: 0.5,
    });

    expect(
      createFileCache({ path, namespace: "gpt-4o-mini@t0/scorer-v4" }).get(
        "val:abc:0",
      ),
    ).toBeUndefined();
  });

  test("refuses to open a log without a namespace naming the system", () => {
    expect(() => createFileCache({ path, namespace: "  " })).toThrow(
      /namespace/,
    );
  });

  test("serves the newest value written for a key", () => {
    const cache = createFileCache({ path, namespace: NAMESPACE });

    cache.set("val:abc:0", { score: 0.1 });
    cache.set("val:abc:0", { score: 0.9 });

    expect(
      createFileCache({ path, namespace: NAMESPACE }).get("val:abc:0"),
    ).toEqual({ score: 0.9 });
  });

  test("reads what it can from a file a killed process truncated", () => {
    const cache = createFileCache({ path, namespace: NAMESPACE });
    cache.set("val:abc:0", { score: 0.5 });
    cache.set("val:abc:1", { score: 0.25 });
    const written = readFileSync(path, "utf8");
    writeFileSync(path, written.slice(0, written.length - 12));

    const reopened = createFileCache({ path, namespace: NAMESPACE });

    expect(reopened.get("val:abc:0")).toEqual({ score: 0.5 });
    expect(reopened.get("val:abc:1")).toBeUndefined();
  });

  test("keeps the first entry written after a truncated one", () => {
    // The half-written record is already lost. Appending onto the line it left
    // open loses the next one too, and that one is a score this run paid for.
    const cache = createFileCache({ path, namespace: NAMESPACE });
    cache.set("val:abc:0", { score: 0.5 });
    cache.set("val:abc:1", { score: 0.25 });
    const written = readFileSync(path, "utf8");
    writeFileSync(path, written.slice(0, written.length - 12));

    const reopened = createFileCache({ path, namespace: NAMESPACE });
    reopened.set("val:abc:2", { score: 0.75 });

    expect(
      createFileCache({ path, namespace: NAMESPACE }).get("val:abc:2"),
    ).toEqual({ score: 0.75 });
  });

  test("offers no entries to a snapshot, because the file already holds them", () => {
    // Checkpoints exist to make scores survive a crash. This cache already
    // does, so copying them into every snapshot would only make it larger.
    expect(
      createFileCache({ path, namespace: NAMESPACE }).entries,
    ).toBeUndefined();
  });
});
