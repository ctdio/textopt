import { describe, expect, it } from "vitest";

import { createEmitter, flushReporters } from "./reporting.js";
import type { Reporter } from "./reporting.js";

type TestEvent = { type: "start" } | { type: "finish"; score: number };

describe("createEmitter", () => {
  it("hands every event to every reporter in order", () => {
    const first: TestEvent[] = [];
    const second: TestEvent[] = [];
    const emit = createEmitter<TestEvent>([
      { onEvent: (event) => first.push(event) },
      { onEvent: (event) => second.push(event) },
    ]);

    emit({ type: "start" });
    emit({ type: "finish", score: 0.5 });

    expect(first).toEqual([{ type: "start" }, { type: "finish", score: 0.5 }]);
    expect(second).toEqual(first);
  });

  it("keeps delivering to later reporters after an earlier one throws", () => {
    const seen: TestEvent[] = [];
    const emit = createEmitter<TestEvent>([
      {
        onEvent: () => {
          throw new Error("logging endpoint is down");
        },
      },
      { onEvent: (event) => seen.push(event) },
    ]);

    emit({ type: "start" });

    expect(seen).toEqual([{ type: "start" }]);
  });

  it("tolerates a reporter that observes nothing", () => {
    const emit = createEmitter<TestEvent>([{}]);

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
