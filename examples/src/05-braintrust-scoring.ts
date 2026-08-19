/**
 * Braintrust as the eval layer: autoevals scorers become the optimizer's
 * metric, and every rollout is logged as an experiment row.
 *
 * Two independent pieces, usable separately:
 *   - `createBraintrustScorer` turns a list of autoevals scorers into one
 *     `ScoreResult`, carrying each scorer's rationale through as feedback and
 *     each scorer's number through as `objectiveScores`.
 *   - `withBraintrustLogging` decorates *any* adapter, so it composes with the
 *     AI SDK adapter used here, the LangChain one, or your own.
 *
 * Without BRAINTRUST_API_KEY it prints the events instead of shipping them, so
 * the example still runs.
 *
 *   OPENAI_API_KEY=... pnpm --filter textopt-examples braintrust
 */
import { openai } from "@ai-sdk/openai";
import { GepaOptimizer } from "textopt/gepa";
import { createAiSdkAdapter } from "@textopt/ai-sdk";
import {
  createBraintrustScorer,
  withBraintrustLogging,
  type BraintrustLoggerLike,
} from "@textopt/braintrust";
import { generateText } from "ai";
import { ExactMatch, Levenshtein } from "autoevals";
import { initLogger } from "braintrust";
import { createReflector, requireApiKey } from "./shared/reflector.js";
import { logEvent, printResult } from "./shared/report.js";
import {
  TICKET_LABELS,
  TRAIN_TICKETS,
  VAL_TICKETS,
  type Ticket,
} from "./shared/tickets.js";

requireApiKey("OPENAI_API_KEY");

const taskModel = openai("gpt-5.6-luna");
const reflect = createReflector({
  model: openai("gpt-5.6"),
  providerOptions: { openai: { reasoningEffort: "high" } },
});

/**
 * Three objectives on one rollout: the label must be right (weighted heaviest),
 * near-misses get partial credit, and the answer must be a bare label. The run
 * optimizes the weighted mean but keeps every component score for inspection.
 */
const scoreTicket = createBraintrustScorer<string>({
  scorers: [
    ExactMatch,
    Levenshtein,
    ({ metadata }) => {
      const raw = String(metadata?.raw ?? "");
      const isBareLabel = /^[a-z_]+$/.test(raw.trim());

      return {
        name: "format",
        score: isBareLabel ? 1 : 0,
        metadata: isBareLabel
          ? {}
          : { rationale: `Expected a bare label, got "${raw.slice(0, 60)}".` },
      };
    },
  ],
  weights: { ExactMatch: 4, Levenshtein: 1, format: 1 },
});

const braintrustLogger: BraintrustLoggerLike =
  process.env.BRAINTRUST_API_KEY === undefined
    ? {
        log: (event) =>
          console.log("    [braintrust:offline]", JSON.stringify(event.scores)),
      }
    : initLogger({ projectName: "textopt-examples" });

const adapter = withBraintrustLogging({
  adapter: createAiSdkAdapter<Ticket>({
    run: ({ candidate, datum, signal }) =>
      generateText({
        model: taskModel,
        system: candidate.system ?? "",
        prompt: datum.text,
        providerOptions: {
          openai: { reasoningEffort: "none", textVerbosity: "low" },
        },
        maxOutputTokens: 512,
        abortSignal: signal,
      }),

    score: async ({ datum, output }) => {
      const raw = output ?? "";
      const predicted =
        TICKET_LABELS.find((label) => raw.toLowerCase().includes(label)) ?? raw;

      const scored = await scoreTicket({
        output: predicted,
        expected: datum.label,
        input: datum.text,
        metadata: { raw },
      });

      // Scorer rationales alone say what was wrong. The dataset's `why` says
      // what the prompt is missing — the reflection model needs both.
      return {
        ...scored,
        feedback: `${scored.feedback}\n${datum.why}`,
      };
    },

    // Mirror the system's real input. Passing the whole row — `label` included
    // — taught the reflection model to write rules about a field the task
    // model never sees.
    buildRecord: ({ datum, output, score, feedback }) => ({
      inputs: datum.text,
      generatedOutputs: output,
      feedback,
      score,
    }),

    concurrency: 4,
  }),
  logger: braintrustLogger,
  metadata: { experiment: "textopt-ticket-routing" },
  toInput: (datum) => datum.text,
  toExpected: (datum) => datum.label,
});

const gepa = new GepaOptimizer({
  minibatchSize: 3,
  seed: 11,
});

const result = await gepa.optimize({
  seedCandidate: {
    system: "Classify the support ticket. Answer with one word.",
  },
  trainingSet: TRAIN_TICKETS,
  validationSet: VAL_TICKETS,
  adapter,
  reflect,
  maxMetricCalls: 150,
  instanceId: ({ datum }) => datum.id,
  onEvent: logEvent,
});

printResult(result);
