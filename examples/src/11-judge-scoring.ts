/**
 * Optimizing against a model-graded metric.
 *
 * Every other example scores by string match, because their tasks have a right
 * answer. Most real tasks do not: a customer reply is good or bad along several
 * axes at once, and no keyword check reads that. `createJudge` builds the
 * metric out of criteria and returns written feedback with the score.
 *
 * The feedback is the point. A judge that hands back only a number reduces a
 * paragraph of diagnosis to one scalar, and reflective search runs on exactly
 * that diagnosis — so the judge grades the *instruction*, not the answer:
 * "the instruction never says to state the refund window" is something a
 * rewriting model can act on, and "this answer should have mentioned the refund
 * window" is not.
 *
 * Two models, two roles: a cheap one writes the replies and is what gets
 * optimized, a frontier one judges and reflects.
 *
 * Cost note: 120 rollouts of the task model, each with a judge call behind it,
 * plus roughly a dozen reflection calls.
 *
 *   ANTHROPIC_API_KEY=... pnpm --filter textopt-examples judge
 */
import { anthropic } from "@ai-sdk/anthropic";
import { createJudge } from "textopt";
import { GepaOptimizer } from "textopt/gepa";
import { createAiSdkAdapter } from "@textopt/ai-sdk";
import { generateText } from "ai";
import { createReflector, requireApiKey } from "./shared/reflector.js";
import { logEvent, printResult } from "./shared/report.js";
import { TRAIN_TICKETS, VAL_TICKETS, type Ticket } from "./shared/tickets.js";

requireApiKey("ANTHROPIC_API_KEY");

const taskModel = anthropic("claude-haiku-4-5-20251001");
const judgeModel = createReflector({ model: anthropic("claude-sonnet-5") });
const reflect = createReflector({
  model: anthropic("claude-opus-5"),
  providerOptions: { anthropic: { thinking: { type: "adaptive" } } },
});

/**
 * Named criteria rather than one "is this good?" question. A model asked for a
 * single overall grade averages its own impressions and returns the middle of
 * the scale for everything; asked for three, it has to commit to each.
 *
 * The scale is deliberately small. Models discriminate between 2 and 4 far
 * more reliably than between 0.4 and 0.8, and the judge normalizes afterwards.
 */
const judge = createJudge<Ticket, string>({
  model: judgeModel,
  scale: 5,
  criteria: [
    {
      name: "routing",
      description:
        "States the queue the ticket belongs in, and it is the right one.",
    },
    {
      name: "reasoning",
      description:
        "Names the specific detail in the ticket that decided the queue, rather than restating the ticket.",
    },
    {
      name: "brevity",
      description:
        "Two sentences at most, with no apology, no filler, and no invented policy.",
    },
  ],
  renderInput: (ticket) => ticket.text,
});

const adapter = createAiSdkAdapter<Ticket>({
  run: ({ candidate, datum, signal }) =>
    generateText({
      model: taskModel,
      system: candidate.system ?? "",
      prompt: datum.text,
      maxOutputTokens: 512,
      abortSignal: signal,
    }),

  // The judge returns a `ScoreResult` already — score, per-criterion
  // `objectiveScores`, and feedback — which is exactly what `score` owes the
  // optimizer. The per-criterion scores are what the Pareto frontier selects
  // on, so a candidate that is best at reasoning survives even if another is
  // better overall.
  score: ({ datum, output }) => judge({ input: datum, output: output ?? "" }),

  pricing: { inputPerMillionUsd: 1, outputPerMillionUsd: 5 },
});

const result = await new GepaOptimizer({
  minibatchSize: 3,
  seed: 11,
}).optimize({
  seedCandidate: {
    system: "Reply to the support ticket.",
  },
  trainingSet: TRAIN_TICKETS,
  validationSet: VAL_TICKETS,
  adapter,
  reflect,
  maxMetricCalls: 120,
  // A dollar ceiling bounds the run in the unit it is actually paid for, which
  // a rollout ceiling cannot: a growing instruction makes late rollouts cost
  // more than early ones. It counts what the adapter reports — the task model
  // here. The judge and the reflection model go through the `TextModel` seam,
  // which is text in and text out and carries no usage, so their spend is real
  // and outside this number.
  maxCostUsd: 5,
  onEvent: logEvent,
});

printResult(result);
console.log(
  `\nspent $${(result.usage.costUsd ?? 0).toFixed(4)} over ${result.usage.rollouts} rollouts` +
    ` (${result.usage.totalTokens} tokens), stopped because ${result.stopReason}`,
);
