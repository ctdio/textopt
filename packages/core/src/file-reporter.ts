import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { OptimizerEvent, Reporter } from "./reporting.js";

/**
 * A run's events as a JSON Lines file, one event per line, written as they
 * happen.
 *
 * The counterpart to `consoleReporter`: a terminal answers "is it moving", and
 * this answers everything asked afterwards. A run costs hours and money, and
 * the numbers it printed are gone the moment the scrollback is — so the events
 * go somewhere a later script can read them back as data rather than parse
 * them out of prose.
 *
 * Appended rather than buffered, for the same reason `createFileCache` is: the
 * run with the most to say about itself is the one that was killed, and a
 * buffer flushed at the end says nothing about that run at all. Written
 * synchronously, like the cache log, because the write is orders of magnitude
 * cheaper than the model call it sits between.
 *
 * `node:fs` is why this is its own entry point. The root of the package runs
 * anywhere; this does not.
 */
export function jsonlReporter(args: {
  path: string;
  /**
   * Event names to keep. Every event by default, which for a long run means a
   * line per rollout — that is the file's job, but `["candidateAccepted",
   * "finish"]` is the shape of a record kept for comparison rather than for
   * debugging.
   */
  only?: readonly string[];
}): Reporter<OptimizerEvent> {
  const { path, only } = args;
  const kept = only === undefined ? undefined : new Set(only);

  mkdirSync(dirname(path), { recursive: true });

  // No `handles`: `only` names what is written, not what is understood, and a
  // record filtered down to two events is not a reporter that misread the
  // union it was attached to.
  return {
    onEvent: (event) => {
      if (kept !== undefined && !kept.has(event.type)) {
        return;
      }
      appendFileSync(path, `${line(event)}\n`);
    },
  };
}

/**
 * One event as the line it is written as. An event whose payload will not
 * serialize is still recorded by name: a run that failed on a circular `err`
 * is exactly the run whose log has to show that something happened there.
 */
function line(event: OptimizerEvent): string {
  try {
    return JSON.stringify(event, errorReplacer);
  } catch {
    return JSON.stringify({ type: event.type, unserializable: true });
  }
}

/** `JSON.stringify` renders an Error as `{}`, which reads as no error at all. */
function errorReplacer(_key: string, value: unknown): unknown {
  return value instanceof Error
    ? { name: value.name, message: value.message, stack: value.stack }
    : value;
}
