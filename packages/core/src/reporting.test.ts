import { describe, expect, it, vi } from "vitest";

import {
  consoleReporter,
  createEmitter,
  createReporter,
  flushReporters,
  objectiveScoresOf,
} from "./reporting.js";
import type { ReportableEvent } from "./reporting.js";
import type { Reporter } from "./reporting.js";

type TestEvent = { type: "start" } | { type: "finish"; score: number };

describe("createEmitter", () => {
  it("hands every event to every reporter in order", () => {
    const first: TestEvent[] = [];
    const second: TestEvent[] = [];
    const emit = createEmitter<TestEvent>({
      reporters: [
        { onEvent: (event) => first.push(event) },
        { onEvent: (event) => second.push(event) },
      ],
      emits: ["start", "finish"],
    });

    emit({ type: "start" });
    emit({ type: "finish", score: 0.5 });

    expect(first).toEqual([{ type: "start" }, { type: "finish", score: 0.5 }]);
    expect(second).toEqual(first);
  });

  it("keeps delivering to later reporters after an earlier one throws", () => {
    const seen: TestEvent[] = [];
    const emit = createEmitter<TestEvent>({
      reporters: [
        {
          onEvent: () => {
            throw new Error("logging endpoint is down");
          },
        },
        { onEvent: (event) => seen.push(event) },
      ],
      emits: ["start", "finish"],
    });

    emit({ type: "start" });

    expect(seen).toEqual([{ type: "start" }]);
  });

  it("tolerates a reporter that observes nothing", () => {
    const emit = createEmitter<TestEvent>({
      reporters: [{}],
      emits: ["start", "finish"],
    });

    expect(() => emit({ type: "start" })).not.toThrow();
  });
});

describe("flushReporters", () => {
  it("awaits every reporter's flush", async () => {
    const flushed: string[] = [];
    const reporters: Reporter<TestEvent>[] = [
      {
        flush: async () => {
          await Promise.resolve();
          flushed.push("first");
        },
      },
      {
        flush: async () => {
          flushed.push("second");
        },
      },
    ];

    await flushReporters(reporters);

    expect(flushed.toSorted()).toEqual(["first", "second"]);
  });

  it("flushes the remaining reporters when one rejects", async () => {
    const flushed: string[] = [];

    await flushReporters<TestEvent>([
      { flush: async () => Promise.reject(new Error("upload failed")) },
      {
        flush: async () => {
          flushed.push("second");
        },
      },
    ]);

    expect(flushed).toEqual(["second"]);
  });
});

describe("createReporter", () => {
  it("hands each event to the handler its type names", () => {
    const seen: TestEvent[] = [];
    const reporter = createReporter<TestEvent>({
      on: {
        finish: (event) => seen.push(event),
      },
    });

    reporter.onEvent?.({ type: "start" });
    reporter.onEvent?.({ type: "finish", score: 0.5 });

    expect(seen).toEqual([{ type: "finish", score: 0.5 }]);
  });

  it("reports the event names it was given handlers for", () => {
    const reporter = createReporter<TestEvent>({
      on: { start: () => undefined, finish: () => undefined },
    });

    expect(reporter.handles?.toSorted()).toEqual(["finish", "start"]);
  });

  it("carries the flush its caller gave it", async () => {
    const flushed: string[] = [];
    const reporter = createReporter<TestEvent>({
      on: {},
      flush: async () => {
        flushed.push("uploaded");
      },
    });

    await reporter.flush?.();

    expect(flushed).toEqual(["uploaded"]);
  });
});

describe("createEmitter event names", () => {
  it("warns about a handler for an event the run never emits", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    createEmitter<TestEvent>({
      reporters: [
        createReporter<TestEvent>({
          on: { start: () => undefined },
          // A reporter written against another optimizer's union, which is how
          // a whole run goes by without printing anything.
        }),
        { onEvent: () => undefined, handles: ["rolloutCompleted"] },
      ],
      emits: ["start", "finish"],
    });

    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0]?.join(" ")).toContain("rolloutCompleted");
    expect(warn.mock.calls[0]?.join(" ")).toContain("start, finish");

    warn.mockRestore();
  });

  it("says nothing about a reporter that never named what it handles", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    createEmitter<TestEvent>({
      reporters: [{ onEvent: () => undefined }],
      emits: ["start", "finish"],
    });

    expect(warn).not.toHaveBeenCalled();

    warn.mockRestore();
  });
});

describe("objectiveScoresOf", () => {
  it("averages each objective over the instances that measured the candidate", () => {
    const row = objectiveScoresOf({
      objectiveScores: [
        { tone: 1, burst: 0 },
        { tone: 0, burst: 1 },
      ],
      transient: [false, false],
    });

    expect(row).toEqual({ objectiveScores: { tone: 0.5, burst: 0.5 } });
  });

  it("leaves a transient instance out of every objective's mean", () => {
    const row = objectiveScoresOf({
      objectiveScores: [{ tone: 1 }, { tone: 0 }],
      transient: [false, true],
    });

    expect(row).toEqual({ objectiveScores: { tone: 1 } });
  });

  it("drops an objective some measured instance never reported", () => {
    const row = objectiveScoresOf({
      objectiveScores: [{ tone: 1, burst: 1 }, { tone: 0 }],
      transient: [false, false],
    });

    expect(row).toEqual({ objectiveScores: { tone: 0.5 } });
  });

  it("reports nothing when the adapter scored no objectives", () => {
    const row = objectiveScoresOf({
      objectiveScores: [undefined, undefined],
      transient: [false, false],
    });

    expect(row).toEqual({});
  });
});

describe("consoleReporter", () => {
  const ROLLOUT: ReportableEvent = {
    type: "rollout",
    iteration: 2,
    phase: "validation",
    split: "val",
    candidateId: 3,
    completed: 12,
    total: 54,
  };

  function accepted(args: {
    candidateId: number;
    aggregateScore: number;
    objectiveScores?: Record<string, number>;
  }): ReportableEvent {
    return {
      type: "candidateAccepted",
      candidate: { instruction: "answer well" },
      instanceScores: [0.5, 1],
      ...args,
    };
  }

  it("writes a line naming the candidate an acceptance moved to", () => {
    const lines: string[] = [];
    const reporter = consoleReporter({ log: (line) => lines.push(line) });

    reporter.onEvent?.(accepted({ candidateId: 3, aggregateScore: 0.7774 }));

    expect(lines).toEqual(["[textopt] accepted #3 score=0.777"]);
  });

  it("writes each objective beside the score it moved", () => {
    const lines: string[] = [];
    const reporter = consoleReporter({ log: (line) => lines.push(line) });

    reporter.onEvent?.(
      accepted({
        candidateId: 1,
        aggregateScore: 0.8,
        objectiveScores: { burst: 0, tone: 0.9 },
      }),
    );

    expect(lines).toEqual([
      "[textopt] accepted #1 score=0.800 burst=0.000 tone=0.900",
    ]);
  });

  it("writes the rollouts of a sweep as they land", () => {
    const lines: string[] = [];
    const reporter = consoleReporter({ log: (line) => lines.push(line) });

    reporter.onEvent?.(ROLLOUT);

    expect(lines).toEqual(["[textopt] rollout validation 12/54"]);
  });

  it("keeps a quiet run to the events that moved the search", () => {
    const lines: string[] = [];
    const reporter = consoleReporter({
      log: (line) => lines.push(line),
      level: "quiet",
    });

    reporter.onEvent?.(ROLLOUT);
    reporter.onEvent?.(accepted({ candidateId: 3, aggregateScore: 0.5 }));

    expect(lines).toEqual(["[textopt] accepted #3 score=0.500"]);
  });

  it("writes a warning on its own line, where the score it qualifies is", () => {
    const lines: string[] = [];
    const reporter = consoleReporter({ log: (line) => lines.push(line) });

    const finished: ReportableEvent = {
      type: "finish",
      bestCandidateId: 3,
      bestScore: 0.777,
      metricCalls: 594,
      warnings: [
        { code: "validationSetReusesTraining", message: "no validationSet" },
      ],
    };

    reporter.onEvent?.(finished);

    expect(lines).toEqual([
      "[textopt] finish best=#3 score=0.777 calls=594",
      "[textopt] warning validationSetReusesTraining: no validationSet",
    ]);
  });

  it("names an event it has no line for rather than dropping it", () => {
    const lines: string[] = [];
    const reporter = consoleReporter({
      log: (line) => lines.push(line),
      level: "verbose",
    });

    // Inferred rather than annotated: an event this reporter has no line for
    // is still an `OptimizerEvent`, and annotating it as one would check the
    // literal against a type that holds only the tag.
    const proposal = { type: "proposal", iteration: 1, changed: true };

    reporter.onEvent?.(proposal);

    expect(lines).toEqual([
      '[textopt] proposal {"iteration":1,"changed":true}',
    ]);
  });
});
