import type { Candidate } from "./types.js";
import type { RunWarning } from "./warnings.js";

/**
 * Where a run's progress goes, for any optimizer. Observability only:
 * persisting a run so it can be resumed is `onCheckpoint`, which is durability
 * and a separate concern.
 *
 * Generic over the event union rather than one union covering every optimizer:
 * a search emits what it actually has, and a reporter written against one
 * optimizer still type-checks against the events it reads.
 */
/**
 * The least an optimizer's event satisfies. A reporter typed against this
 * accepts every optimizer's union, because a literal tag is assignable to
 * `string` and the parameter position is contravariant.
 */
export interface OptimizerEvent {
  type: string;
}

export interface Reporter<Event> {
  /**
   * Called on the search's hot path, synchronously. A reporter that ships
   * anywhere over a network buffers here and uploads in `flush`, or it charges
   * every iteration for its latency.
   *
   * A reporter that throws is warned about and skipped: observability never
   * fails a run.
   */
  onEvent?: (event: Event) => void;
  /** Awaited once as the run ends, including when it ends by throwing. */
  flush?: () => Promise<void>;
}

/**
 * The payload every optimizer's `candidateAccepted` carries, so one reporter
 * can read an acceptance without knowing which search produced it. Each
 * optimizer intersects its own fields onto this — GEPA its lineage, MIPRO its
 * menu choices — the way the event unions already intersect `EvaluationEvent`.
 *
 * Emitted only when the incumbent moves and a full validation sweep measured
 * it. An optimizer that accepts on a minibatch reports the acceptance in its
 * own event and emits this one once the sweep that confirms it lands, so
 * `instanceScores` never means "a subset, and you work out which".
 */
export interface CandidateAccepted<K extends string = string> {
  /** Identifies the candidate within the run. */
  candidateId: number;
  /** The text that scored, so a move is readable next to the edit. */
  candidate: Candidate<K>;
  /** Mean over the validation set, which is what selection is decided on. */
  aggregateScore: number;
  /**
   * Per-instance scores, aligned with the validation set. `undefined` marks an
   * instance an infrastructure failure left unmeasured — unknown, not zero.
   *
   * Handed out by reference rather than copied: a run emits this once per
   * accepted candidate, and copying a validation-set-sized array that often to
   * guard against a listener that writes to it costs every run to protect a
   * listener that should not exist.
   */
  instanceScores: readonly (number | undefined)[];
  /** Aligned with `instanceScores`. Present only under `trackBestOutputs`. */
  outputs?: readonly unknown[];
}

/**
 * The payload every optimizer's `finish` carries. `reason` stays per-optimizer
 * because the stop reasons genuinely differ — only GEPA can exhaust a
 * reflection budget.
 */
export interface RunFinished {
  /** The winner, named the way `CandidateAccepted.candidateId` names it. */
  bestCandidateId: number;
  bestScore: number;
  metricCalls: number;
  /** The winner's held-out score, when a testSet was given. */
  testScore?: number;
  /**
   * The winner's per-instance held-out scores, aligned with the testSet and
   * present whenever `testScore` is. `undefined` marks an instance an
   * infrastructure failure left unmeasured, which is what `testScore` averages
   * over too.
   *
   * The mean is the number selection never saw; this is where the gap below
   * `bestScore` came from.
   */
  testInstanceScores?: readonly (number | undefined)[];
  /** Aligned with `testInstanceScores`. Only under `trackBestOutputs`. */
  testOutputs?: readonly unknown[];
  /**
   * What the run cannot say about itself from its own numbers. A reporter that
   * writes the score somewhere permanent writes these beside it, or the record
   * outlives the only place the caveat was ever stated.
   */
  warnings: readonly RunWarning[];
}

/**
 * The two events every optimizer emits with a payload a reporter can read
 * without knowing which search produced it. A cross-optimizer reporter takes
 * this as its event type: it is a supertype of every optimizer's own union, so
 * one reporter drops into any optimizer's `reporters` array.
 */
export type ReportableEvent<K extends string = string> =
  | ({ type: "candidateAccepted" } & CandidateAccepted<K>)
  | ({ type: "finish" } & RunFinished);

/**
 * Narrows an event off any optimizer's union to an acceptance. The tag is
 * enough: every optimizer's `candidateAccepted` intersects `CandidateAccepted`,
 * so carrying the payload is a compile-time obligation rather than a hope.
 */
export function isCandidateAccepted<K extends string = string>(
  event: OptimizerEvent,
): event is { type: "candidateAccepted" } & CandidateAccepted<K> {
  return event.type === "candidateAccepted";
}

/** Narrows an event off any optimizer's union to the end of the run. */
export function isRunFinished(
  event: OptimizerEvent,
): event is { type: "finish" } & RunFinished {
  return event.type === "finish";
}

/**
 * Fans one event out to every reporter, absorbing whatever they throw. A
 * reporter is an observer of the search, never a participant in it: a logging
 * endpoint that is down must not decide a run's outcome.
 */
export function createEmitter<Event extends { type: string }>(
  reporters: readonly Reporter<Event>[],
): (event: Event) => void {
  return function emit(event: Event): void {
    for (const reporter of reporters) {
      try {
        reporter.onEvent?.(event);
      } catch (err) {
        console.warn("[textopt] reporter threw while handling an event", {
          type: event.type,
          err,
        });
      }
    }
  };
}

/**
 * Gives every reporter its one chance to upload what it buffered. Called from
 * a `finally` rather than after the run: a reporter that buffers has the most
 * to say about a run that aborted or threw, and that is exactly the run that
 * never reaches its last line.
 */
export async function flushReporters<Event>(
  reporters: readonly Reporter<Event>[],
): Promise<void> {
  await Promise.all(
    reporters.map(async (reporter) => {
      try {
        await reporter.flush?.();
      } catch (err) {
        console.warn("[textopt] reporter threw while flushing", { err });
      }
    }),
  );
}

/**
 * A scored batch as the per-instance row a reporter should read: an instance
 * an infrastructure failure left unmeasured becomes `undefined` rather than
 * the zero the adapter reported for it.
 *
 * The same distinction `measuredMean` makes when it averages — a row of zeros
 * and a row of unknowns describe very different runs.
 */
export function instanceRow(batch: {
  scores: readonly number[];
  transient: readonly boolean[];
}): (number | undefined)[] {
  return batch.scores.map((score, index) =>
    batch.transient[index] === true ? undefined : score,
  );
}
