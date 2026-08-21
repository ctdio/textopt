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
 *
 * `namespace` is what makes that invariant true. A cached score measures a
 * whole system, not a candidate, and this log outlives every part of that
 * system a run does not pass through the key: the model id behind an alias the
 * provider upgraded, the decoding settings, the scorer's own version. It is
 * required rather than optional because the failure it prevents is silent —
 * scores from one system served to a run of another, with a normal-looking
 * result and no way to read afterwards that it happened.
 */
export function createFileCache(args: {
  path: string;
  /**
   * Names the system these scores measure — model id, decoding settings,
   * scorer version. Change it whenever any of them changes.
   */
  namespace: string;
  /** Entries kept in memory. The file itself is never trimmed. */
  maxEntries?: number;
}): EvaluationCache {
  const { path, namespace, maxEntries = 1_000_000 } = args;

  if (namespace.trim() === "") {
    throw new Error(
      "createFileCache requires a non-empty namespace naming the system these scores measure",
    );
  }

  mkdirSync(dirname(path), { recursive: true });
  const log = readLog(path);
  const entries = log.entries;
  const scope = (key: string) => `${namespace}\u0000${key}`;
  // A process killed mid-write leaves its last record unterminated. That one is
  // already lost; closing the line keeps the next write from being appended
  // onto it and lost with it.
  if (log.unterminated) {
    appendFileSync(path, "\n");
  }

  return {
    get: (key) => entries.get(scope(key)),
    set: (key, cached) => {
      const scoped = scope(key);
      if (entries.size >= maxEntries && !entries.has(scoped)) {
        const oldest = entries.keys().next();
        if (!oldest.done) {
          entries.delete(oldest.value);
        }
      }
      entries.set(scoped, cached);
      appendFileSync(path, `${JSON.stringify([scoped, cached])}\n`);
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
function readLog(path: string): {
  entries: Map<string, CachedScore>;
  unterminated: boolean;
} {
  const entries = new Map<string, CachedScore>();

  let contents: string;
  try {
    contents = readFileSync(path, "utf8");
  } catch {
    return { entries, unterminated: false };
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
  return {
    entries,
    unterminated: contents.length > 0 && !contents.endsWith("\n"),
  };
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
