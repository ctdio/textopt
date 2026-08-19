import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import type { CachedScore, EvaluationCache } from "./cache.js";

/**
 * An evaluation cache that outlives the process, as an append-only log.
 *
 * A long run against a real provider is measured in hours and dollars, and an
 * in-memory cache throws all of it away when the run ends — a crashed run, a
 * re-run with a changed budget, or a second experiment over the same
 * validation set all pay for identical rollouts again.
 *
 * Append-only rather than rewritten: a score is never invalidated (the key
 * names the candidate, the instance, and the environment), and a log survives
 * a process killed mid-write, which a file rewritten in place does not.
 */
export function createFileCache(args: {
  path: string;
  /** Entries kept in memory. The file itself is never trimmed. */
  maxEntries?: number;
}): EvaluationCache {
  const { path, maxEntries = 1_000_000 } = args;

  mkdirSync(dirname(path), { recursive: true });
  const entries = readLog(path);

  return {
    get: (key) => entries.get(key),
    set: (key, cached) => {
      if (entries.size >= maxEntries && !entries.has(key)) {
        const oldest = entries.keys().next();
        if (!oldest.done) {
          entries.delete(oldest.value);
        }
      }
      entries.set(key, cached);
      appendFileSync(path, `${JSON.stringify([key, cached])}\n`);
    },
    // Deliberately absent: `entries` exists so a checkpoint can carry scores
    // that would otherwise be lost, and these are already on disk.
  };
}

/**
 * Later records win, so a re-measured instance replaces its earlier reading.
 * A record that does not parse is dropped rather than fatal: the last line of
 * a log whose process was killed mid-write is routinely half-written, and
 * losing one cached score is not worth failing a run over.
 */
function readLog(path: string): Map<string, CachedScore> {
  const entries = new Map<string, CachedScore>();

  let contents: string;
  try {
    contents = readFileSync(path, "utf8");
  } catch {
    return entries;
  }

  for (const line of contents.split("\n")) {
    if (line.length === 0) {
      continue;
    }
    const entry = parseEntry(line);
    if (entry !== undefined) {
      entries.set(entry[0], entry[1]);
    }
  }
  return entries;
}

function parseEntry(line: string): [string, CachedScore] | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(line);
  } catch {
    return undefined;
  }

  if (!Array.isArray(parsed) || parsed.length !== 2) {
    return undefined;
  }
  const [key, cached] = parsed as [unknown, unknown];
  if (typeof key !== "string" || !isCachedScore(cached)) {
    return undefined;
  }
  return [key, cached];
}

function isCachedScore(value: unknown): value is CachedScore {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as CachedScore).score === "number"
  );
}
