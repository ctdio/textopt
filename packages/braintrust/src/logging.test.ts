import type { EvaluateArgs, EvaluationContext } from "textopt";
import type { KeywordExample } from "textopt/testing";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "textopt/testing";
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

const RUN: EvaluationContext = {
  iteration: 0,
  phase: "minibatch",
  split: "train",
  candidateId: 0,
};

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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
    });

    for (const event of logger.events) {
      expect(event.metadata?.run).toBe("nightly");
    }
  });

  test("logs where in the run each event came from", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
      run: {
        iteration: 7,
        phase: "validation",
        split: "val",
        candidateId: 3,
      },
    });

    expect(logger.events[0]?.metadata).toMatchObject({
      iteration: 7,
      phase: "validation",
      split: "val",
      candidateId: 3,
    });
  });

  test("logs a proposal being screened as having no candidate id", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
      run: { ...RUN, candidateId: null },
    });

    expect(logger.events[0]?.metadata?.candidateId).toBeNull();
  });

  test("keeps the run context when the caller supplies its own metadata", async () => {
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
      metadata: { iteration: "nightly" },
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
      run: { ...RUN, iteration: 2 },
    });

    expect(logger.events[0]?.metadata?.iteration).toBe(2);
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
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
      run: RUN,
    });

    expect(logger.events[0]?.expected).toEqual(["hold", "ten seconds"]);
  });

  test("logs what a rollout spent under the metric names braintrust reads", async () => {
    // Braintrust derives an experiment's token and cost columns from `metrics`
    // under these exact names. Reporting them anywhere else leaves the columns
    // empty, which is the whole reason to log rollouts there.
    const logger = createRecordingLogger();
    const keyword = createKeywordAdapter();
    const adapter = withBraintrustLogging({
      adapter: {
        ...keyword,
        evaluate: async (args: EvaluateArgs<KeywordExample>) => ({
          ...(await keyword.evaluate(args)),
          usage: args.batch.map(() => ({
            inputTokens: 120,
            outputTokens: 30,
            totalTokens: 150,
            costUsd: 0.002,
          })),
        }),
      },
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
      run: RUN,
    });

    expect(logger.events[0]?.metrics).toEqual({
      prompt_tokens: 120,
      completion_tokens: 30,
      tokens: 150,
      cost_usd: 0.002,
    });
  });

  test("logs no metrics for an adapter that reports no usage", async () => {
    // An absent reading is not a zero one: a zero token count would read as a
    // free rollout rather than an unmeasured one.
    const logger = createRecordingLogger();
    const adapter = withBraintrustLogging({
      adapter: createKeywordAdapter(),
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
      run: RUN,
    });

    expect(logger.events[0]?.metrics).toBeUndefined();
  });

  test("logs only the readings an adapter actually reported", async () => {
    const logger = createRecordingLogger();
    const keyword = createKeywordAdapter();
    const adapter = withBraintrustLogging({
      adapter: {
        ...keyword,
        evaluate: async (args: EvaluateArgs<KeywordExample>) => ({
          ...(await keyword.evaluate(args)),
          usage: args.batch.map(() => ({ costUsd: 0.5 })),
        }),
      },
      logger,
    });

    await adapter.evaluate({
      batch: KEYWORD_EXAMPLES.slice(0, 1),
      candidate: CANDIDATE,
      captureTraces: false,
      run: RUN,
    });

    expect(logger.events[0]?.metrics).toEqual({ cost_usd: 0.5 });
  });
});
