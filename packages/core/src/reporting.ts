import type { EvaluationEvent, RolloutProgress } from "./evaluation.js";
import type { Candidate } from "./types.js";
import type { RunWarning } from "./warnings.js";

/**
 * The least an optimizer's event satisfies. A reporter typed against this
 * accepts every optimizer's union, because a literal tag is assignable to
 * `string` and the parameter position is contravariant.
 */
export interface OptimizerEvent {
  type: string;
}

/**
 * Where a run's progress goes, for any optimizer. Observability only:
 * persisting a run so it can be resumed is `onCheckpoint`, which is durability
 * and a separate concern.
 *
 * Generic over the event union rather than one union covering every optimizer:
 * a search emits what it actually has, and a reporter written against one
 * optimizer still type-checks against the events it reads.
 */
export interface Reporter<Event> {
  /**
   * Event names this reporter reads, when it can say. `createReporter` fills
   * it from its handler keys, which is what lets a run say at once that a
   * handler is named after an event it will never receive — the failure that
   * otherwise reads as a search that printed nothing.
   *
   * A reporter that formats whatever arrives, as `consoleReporter` does,
   * leaves this unset: it has no names to be wrong about.
   */
  handles?: readonly string[];
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
  /**
   * Mean of each objective over the same instances `aggregateScore` averages,
   * when the adapter scored any. Absent for an objective some measured
   * instance did not report: a mean over a subset is not the same quantity.
   *
   * Here and not only on the result because this is the one place it is
   * actionable. A single objective collapsing while the aggregate holds is how
   * a degenerate metric channel announces itself, and a run that only reports
   * it at the end reports it after the budget is spent.
   */
  objectiveScores?: Readonly<Record<string, number>>;
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
 * The payload every optimizer's `start` carries: what is being optimized, and
 * how many instances the number that selects it will be averaged over.
 */
export interface RunStarted<K extends string = string> {
  components: K[];
  validationSetSize: number;
}

/**
 * The events every optimizer emits with a payload a reporter can read without
 * knowing which search produced it. A cross-optimizer reporter takes this as
 * its event type: `start` and `finish` bracket the run, `evaluation` and
 * `rollout` are what it spends, and `candidateAccepted` is what it bought.
 *
 * Not a supertype of any optimizer's union — each search emits more than this
 * — so a reporter that must drop into every `reporters` array is typed
 * `Reporter<OptimizerEvent>` and narrows with the guards below.
 */
export type ReportableEvent<K extends string = string> =
  | ({ type: "start" } & RunStarted<K>)
  | ({ type: "evaluation" } & EvaluationEvent)
  | ({ type: "rollout" } & RolloutProgress)
  | ({ type: "candidateAccepted" } & CandidateAccepted<K>)
  | ({ type: "finish" } & RunFinished);

/**
 * A handler per event name, each receiving that event alone rather than the
 * whole union. Keys are the union's own tags, so a handler named after an
 * event that does not exist is a compile error instead of a callback nothing
 * ever calls.
 */
export type EventHandlers<Event extends OptimizerEvent> = {
  [Type in Event["type"]]?: (event: Extract<Event, { type: Type }>) => void;
};

/** How much of a run `consoleReporter` prints. */
export type ConsoleReporterLevel = "quiet" | "normal" | "verbose";

/**
 * Every event name any optimizer in this library emits.
 *
 * It exists to keep the check below from punishing the thing this event
 * substrate is for. A reporter written once and attached to several searches
 * handles names a given run does not emit — `error` comes from GEPA and SIMBA
 * and from none of the other four — and warning about that would train the
 * reader to ignore the warning that matters. A name from nowhere in this list
 * is a different thing: nothing will ever send it.
 *
 * `reporting.types.test.ts` asserts this is exactly the union of the six
 * optimizers' own lists, so an optimizer that adds an event cannot leave it
 * stale.
 */
export const EVERY_EVENT_NAME: readonly string[] = [
  "attempt",
  "candidate",
  "candidateAccepted",
  "candidateRejected",
  "error",
  "evaluation",
  "finish",
  "iterationStart",
  "menu",
  "proposal",
  "rollout",
  "roundStart",
  "start",
  "stepStart",
  "trial",
];

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

/** Narrows an event off any optimizer's union to the start of the run. */
export function isRunStarted<K extends string = string>(
  event: OptimizerEvent,
): event is { type: "start" } & RunStarted<K> {
  return event.type === "start";
}

/** Narrows an event off any optimizer's union to one scored batch. */
export function isEvaluation(
  event: OptimizerEvent,
): event is { type: "evaluation" } & EvaluationEvent {
  return event.type === "evaluation";
}

/** Narrows an event off any optimizer's union to one finished rollout. */
export function isRollout(
  event: OptimizerEvent,
): event is { type: "rollout" } & RolloutProgress {
  return event.type === "rollout";
}

/**
 * A reporter from one handler per event, rather than one callback that
 * switches on a tag.
 *
 * This is the shape to write reporters in, because it is the shape a wrong
 * event name cannot survive. Handler keys are the event union's own tags, so
 * `createReporter<GepaEvent>({ on: { rolloutCompleted: … } })` does not
 * compile, and a name that no run emits — the same handler attached to an
 * optimizer that has no such event — is warned about when the run starts
 * rather than never firing. A reporter typed against `{ type: string }`, which
 * is what an integrator writes before knowing the union, has neither check.
 */
export function createReporter<Event extends OptimizerEvent>(args: {
  on: EventHandlers<Event>;
  /** Awaited once as the run ends, including when it ends by throwing. */
  flush?: () => Promise<void>;
}): Reporter<Event> {
  const { on, flush } = args;
  const handlers = on as Record<string, ((event: Event) => void) | undefined>;

  return {
    handles: Object.keys(on),
    onEvent: (event) => handlers[event.type]?.(event),
    ...(flush === undefined ? {} : { flush }),
  };
}

/**
 * A reporter that prints a run as it happens, for any optimizer.
 *
 * One line per event, never rewritten in place. A progress bar is unreadable
 * the moment a run is redirected to a file — which is where a run long enough
 * to need one ends up — and a line per event stays greppable, tailable, and
 * interleaves with a logger instead of fighting it.
 *
 * `"normal"` includes a line per rollout, which is the only signal fine
 * grained enough to answer "is this moving" during a validation sweep of a
 * slow model. `"quiet"` keeps the events that moved the search.
 */
export function consoleReporter(args?: {
  /** Where a line goes. Defaults to `console.log`. */
  log?: (line: string) => void;
  level?: ConsoleReporterLevel;
}): Reporter<OptimizerEvent> {
  const { log = defaultLog, level = "normal" } = args ?? {};

  // No `handles`: this formats whatever arrives, so it has no event names to
  // be wrong about, and claiming a set would warn on every optimizer whose
  // union is not exactly that set.
  return {
    onEvent: (event) => {
      for (const line of formatEvent({ event, level })) {
        log(`[textopt] ${line}`);
      }
    },
  };
}

/**
 * Fans one event out to every reporter, absorbing whatever they throw. A
 * reporter is an observer of the search, never a participant in it: a logging
 * endpoint that is down must not decide a run's outcome.
 */
export function createEmitter<Event extends OptimizerEvent>(args: {
  reporters: readonly Reporter<Event>[];
  /**
   * Every event name this run can emit. Checked against what the reporters
   * say they handle, because a handler named after an event that never
   * arrives is indistinguishable, from the outside, from a search that had
   * nothing to report.
   */
  emits: readonly Event["type"][];
}): (event: Event) => void {
  const { reporters, emits } = args;

  for (const reporter of reporters) {
    const handles = reporter.handles ?? [];
    const unnamed = handles.filter((name) => !EVERY_EVENT_NAME.includes(name));

    for (const name of unnamed) {
      console.warn(
        `[textopt] a reporter handles "${name}", which no optimizer emits, so that handler will never fire. This run emits: ${emits.join(", ")}`,
      );
    }

    // A reporter whose every handler misses is the failure worth naming even
    // when each name is real: it is a run observed by nothing, which reads
    // exactly like a search that had nothing to say. One warning covers it —
    // a typo already warned about above is the same mistake, not a second one.
    const fires = handles.some((name) => emits.includes(name as Event["type"]));

    if (unnamed.length === 0 && handles.length > 0 && !fires) {
      console.warn(
        `[textopt] a reporter handles only events this run never emits (${handles.join(", ")}), so it will report nothing. This run emits: ${emits.join(", ")}`,
      );
    }
  }

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
/**
 * A scored batch's objective means as `candidateAccepted` carries them, ready
 * to spread into the event: absent, rather than present and undefined, when
 * the adapter scored no objectives.
 *
 * Transient rows measured the infrastructure rather than the candidate, so
 * they are left out here exactly as `measuredMean` leaves them out of the
 * aggregate these sit beside.
 */
export function objectiveScoresOf(batch: {
  objectiveScores: readonly (Record<string, number> | undefined)[];
  transient: readonly boolean[];
}): { objectiveScores?: Record<string, number> } {
  const objectiveScores = meanObjectives({
    rows: batch.objectiveScores,
    measured: batch.objectiveScores.map(
      (_, index) => batch.transient[index] !== true,
    ),
  });

  return objectiveScores === undefined ? {} : { objectiveScores };
}

/**
 * Mean of each objective over the instances that measured the candidate.
 *
 * An objective missing from any of those instances is dropped rather than
 * averaged over the subset that has it: two candidates whose objective means
 * cover different instances are not comparable, and an objective frontier
 * selects on exactly this number.
 */
export function meanObjectives(args: {
  rows: readonly (Record<string, number> | undefined)[];
  /** Aligned with `rows`: whether that instance measured the candidate. */
  measured: readonly boolean[];
}): Record<string, number> | undefined {
  const { rows, measured } = args;

  const scored = rows.filter((_, index) => measured[index] === true);
  const totals = new Map<string, { total: number; count: number }>();

  for (const row of scored) {
    for (const [objective, value] of Object.entries(row ?? {})) {
      const running = totals.get(objective) ?? { total: 0, count: 0 };
      running.total += value;
      running.count += 1;
      totals.set(objective, running);
    }
  }

  const complete = [...totals].filter(
    ([, { count }]) => count === scored.length,
  );
  if (complete.length === 0) {
    return undefined;
  }
  return Object.fromEntries(
    complete.map(([objective, { total, count }]) => [objective, total / count]),
  );
}

export function instanceRow(batch: {
  scores: readonly number[];
  transient: readonly boolean[];
}): (number | undefined)[] {
  return batch.scores.map((score, index) =>
    batch.transient[index] === true ? undefined : score,
  );
}

function defaultLog(line: string): void {
  console.log(line);
}

/**
 * One event as the lines it prints, without the prefix. The shared events get
 * a shape a reader can scan down a column; anything else is named with its
 * payload, so a search's own events are never silently dropped.
 */
function formatEvent(args: {
  event: OptimizerEvent;
  level: ConsoleReporterLevel;
}): string[] {
  const { event, level } = args;
  const { type } = event;

  if (isRunFinished(event)) {
    return [
      `finish best=#${event.bestCandidateId} score=${score(event.bestScore)}` +
        ` calls=${event.metricCalls}` +
        (event.testScore === undefined
          ? ""
          : ` test=${score(event.testScore)}`),
      ...event.warnings.map(
        (warning) => `warning ${warning.code}: ${warning.message}`,
      ),
    ];
  }

  if (isCandidateAccepted(event)) {
    return [
      `accepted #${event.candidateId} score=${score(event.aggregateScore)}` +
        objectives(event.objectiveScores),
    ];
  }

  // An error is the one event a quiet run still prints: it is the reason a
  // result that follows may not mean what it says.
  if (type === "error") {
    return [generic(event)];
  }

  if (level === "quiet") {
    return [];
  }

  if (isRollout(event)) {
    return [`rollout ${event.phase} ${event.completed}/${event.total}`];
  }

  if (isEvaluation(event)) {
    return [
      `evaluation ${event.phase} mean=${score(event.meanScore)}` +
        ` calls=${event.metricCalls} cached=${event.cacheHits}`,
    ];
  }

  if (isRunStarted(event)) {
    return [
      `start components=[${event.components.join(", ")}]` +
        ` validation=${event.validationSetSize}`,
    ];
  }

  return level === "verbose" ? [generic(event)] : [];
}

function score(value: number): string {
  return value.toFixed(3);
}

function objectives(scores?: Readonly<Record<string, number>>): string {
  return Object.entries(scores ?? {})
    .map(([objective, value]) => ` ${objective}=${score(value)}`)
    .join("");
}

/**
 * An event this reporter has no line for, named with whatever it carried. A
 * payload that will not serialize — an `err` holding a cycle — must still
 * print its event: the type is the part worth having.
 */
function generic(event: OptimizerEvent): string {
  const { type, ...rest } = event;

  try {
    return `${type} ${JSON.stringify(rest, errorReplacer)}`;
  } catch {
    return type;
  }
}

/** `JSON.stringify` renders an Error as `{}`, which reads as no error at all. */
function errorReplacer(_key: string, value: unknown): unknown {
  return value instanceof Error ? `${value.name}: ${value.message}` : value;
}
