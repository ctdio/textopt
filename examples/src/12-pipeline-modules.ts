/**
 * Optimizing a system made of several prompts instead of one.
 *
 * The system here is two modules in sequence: one reads a ticket and writes
 * down what it found, the other reads those notes and picks a queue. Each has
 * its own instruction, and both are components of the same candidate — so the
 * optimizer searches over the pair, and can discover that the fix for a
 * misrouted ticket belongs in the *first* module.
 *
 * That attribution is the whole reason `createPipelineAdapter` exists. Written
 * by hand, a multi-module adapter usually shows the reflection model the
 * pipeline's input and its final answer, which is the one view that tells you
 * nothing about which module lost the point. This one records what each module
 * received and produced and hands that back as the trajectory.
 *
 * Cost note: two model calls per rollout, capped at 90 rollouts.
 *
 *   ANTHROPIC_API_KEY=... pnpm --filter textopt-examples pipeline
 */
import { anthropic } from "@ai-sdk/anthropic";
import { GepaOptimizer } from "textopt/gepa";
import { createPipelineAdapter } from "textopt/gepa";
import { generateText } from "ai";
import { createReflector, requireApiKey } from "./shared/reflector.js";
import { logEvent, printResult } from "./shared/report.js";
import {
  TICKET_LABELS,
  TRAIN_TICKETS,
  VAL_TICKETS,
  type Ticket,
} from "./shared/tickets.js";

requireApiKey("ANTHROPIC_API_KEY");

const taskModel = anthropic("claude-haiku-4-5-20251001");
const reflect = createReflector({
  model: anthropic("claude-opus-5"),
  providerOptions: { anthropic: { thinking: { type: "adaptive" } } },
});

async function ask(args: {
  instruction: string;
  input: string;
  signal?: AbortSignal;
}): Promise<string> {
  const { instruction, input, signal } = args;

  const result = await generateText({
    model: taskModel,
    system: instruction,
    prompt: input,
    maxOutputTokens: 512,
    abortSignal: signal,
  });

  return result.text.trim();
}

const adapter = createPipelineAdapter<Ticket, string, "triage" | "route">({
  // The order here is the order they run in, and each module's `component`
  // names the candidate field holding its instruction.
  modules: [
    {
      component: "triage",
      run: ({ instruction, input, signal }) =>
        ask({
          instruction,
          input: String(input),
          ...(signal ? { signal } : {}),
        }),
    },
    {
      component: "route",
      run: ({ instruction, input, signal }) =>
        ask({
          instruction,
          input: String(input),
          ...(signal ? { signal } : {}),
        }),
    },
  ],

  // What the first module receives. Without this the module would be handed the
  // whole row, label included, and would be scored on copying it back.
  input: (ticket) => ticket.text,

  // `steps` is the trace: every module's input and output, in order. The score
  // is end-to-end, but the feedback can name which step went wrong — and that
  // is the sentence the reflection model rewrites an instruction from.
  score: ({ datum, output, steps }) => {
    const predicted = TICKET_LABELS.find((label) =>
      output.toLowerCase().includes(label),
    );
    const notes = String(steps[0]?.output ?? "");

    if (predicted === datum.label) {
      return { score: 1, feedback: `Correct: ${datum.label}.` };
    }

    return {
      score: 0,
      feedback: [
        predicted === undefined
          ? `The router answered "${output.slice(0, 80)}", which names no queue. It must answer with exactly one of: ${TICKET_LABELS.join(", ")}.`
          : `The router chose "${predicted}" but the correct queue is "${datum.label}".`,
        `Triage passed on: "${notes.slice(0, 200)}"`,
        datum.why,
      ].join("\n"),
    };
  },

  concurrency: 4,
});

const result = await new GepaOptimizer({
  minibatchSize: 3,
  seed: 12,
  // Two components, so merging is on by default: a candidate that improved
  // triage and one that improved routing can be combined into a third that has
  // both, without either lineage having discovered the other's change.
}).optimize({
  seedCandidate: {
    triage: "Summarize what the customer is reporting.",
    route: "Answer with the queue this ticket belongs in.",
  },
  trainingSet: TRAIN_TICKETS,
  validationSet: VAL_TICKETS,
  adapter,
  reflect,
  maxMetricCalls: 90,
  reporters: [{ onEvent: logEvent }],
});

printResult(result);
