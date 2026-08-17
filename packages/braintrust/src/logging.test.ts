import { KEYWORD_EXAMPLES, createKeywordAdapter } from "@ctdio/gepa/testing";
import { describe, expect, test } from "vitest";
import type { BraintrustEvent, BraintrustLoggerLike } from "./logging.js";
import { withBraintrustLogging } from "./logging.js";

function createRecordingLogger(): BraintrustLoggerLike & {
  events: BraintrustEvent[];
} {
  const events: BraintrustEvent[] = [];
  return {
    events,
    log: (event) => {
      events.push(event);
      return undefined;
    },
  };
}

const CANDIDATE = { instruction: "hold ten seconds" };

describe("withBraintrustLogging", () => {
  test("logs one event per evaluated instance", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 2),
      candidate: CANDIDATE,
      captureTraces: false,
    });

    expect(logger.events).toHaveLength(2);
  });

  test("logs the score and the candidate that produced it", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
    });

    const event = logger.events[0];

    expect(event?.scores?.score).toBe(1);
    expect(event?.metadata?.candidate).toEqual(CANDIDATE);
  });

  test("logs the feedback string as metadata", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(1, 2),
      candidate: CANDIDATE,
      captureTraces: false,
    });

    expect(logger.events[0]?.metadata?.feedback).toContain(
      "Missing required terms",
    );
  });

  test("attaches caller-supplied metadata to every event", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
      metadata: { run: "nightly" },
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 2),
      candidate: CANDIDATE,
      captureTraces: false,
    });

    for (const event of logger.events) {
      expect(event.metadata?.run).toBe("nightly");
    }
  });

  test("returns the wrapped adapter's evaluation unchanged", async () => {
    const inner = createKeywordAdapter();
    const wrapped = withBraintrustLogging({
      adapter: inner,
      logger: createRecordingLogger(),
    });
    const args = {
      batch: KEYWORD_EXAMPLES,
      candidate: CANDIDATE,
      captureTraces: false,
    };

    const direct = await inner.evaluate(args);
    const throughWrapper = await wrapped.evaluate(args);

    expect(throughWrapper.scores).toEqual(direct.scores);
    expect(throughWrapper.outputs).toEqual(direct.outputs);
  });

  test("keeps evaluating when the logger throws", async () => {
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger: {
        log: () => {
          throw new Error("braintrust unreachable");
        },
      },
    });

    const result = await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 2),
      candidate: CANDIDATE,
      captureTraces: false,
    });

    expect(result.scores).toHaveLength(2);
  });

  test("delegates reflective dataset construction to the wrapped adapter", async () => {
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger: createRecordingLogger(),
    });
    const batch = KEYWORD_EXAMPLES.slice(0, 2);

    const evaluation = await adapter.evaluate({
      batch,
      candidate: CANDIDATE,
      captureTraces: true,
    });
    const dataset = await adapter.makeReflectiveDataset({
      candidate: CANDIDATE,
      batch,
      evaluation,
      componentsToUpdate: ["instruction"],
    });

    expect(dataset.instruction).toHaveLength(2);
  });

  test("logs the expected value when a mapper is provided", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
      toExpected: (datum) => datum.required,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
    });

    expect(logger.events[0]?.expected).toEqual(["hold", "ten seconds"]);
  });
});
