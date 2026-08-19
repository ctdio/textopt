/**
 * Optimizing a Vercel AI SDK call.
 *
 * The system under optimization is one `generateText` call; the candidate's
 * `system` component is its system prompt. Nothing else changes between
 * candidates, so any score movement is attributable to the prompt.
 *
 * Cost note: capped at 150 rollouts of the task model plus roughly a dozen
 * reflection calls. Lower `maxMetricCalls` to spend less.
 *
 *   ANTHROPIC_API_KEY=... pnpm --filter textopt-examples ai-sdk
 */
import { anthropic } from "@ai-sdk/anthropic";
import { GepaOptimizer } from "textopt/gepa";
import { createAiSdkAdapter } from "@textopt/ai-sdk";
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

// Nothing below this block is vendor-specific. To run the same optimization on
// OpenAI, import `openai` from "@ai-sdk/openai" and use:
//   const taskModel = openai("gpt-5.4-mini");
//   const reflect = createReflector({
//     model: openai("gpt-5.6"),
//     providerOptions: { openai: { reasoningEffort: "high" } },
//   });
const taskModel = anthropic("claude-sonnet-5");
const reflect = createReflector({
  model: anthropic("claude-opus-5"),
  providerOptions: { anthropic: { thinking: { type: "adaptive" } } },
});

const adapter = createAiSdkAdapter<Ticket>({
  run: ({ candidate, datum, signal }) =>
    generateText({
      model: taskModel,
      system: candidate.system ?? "",
      prompt: datum.text,
      // Generous enough for a reasoning model's hidden tokens if you swap one
      // in; the visible answer is a single word either way.
      maxOutputTokens: 512,
      abortSignal: signal,
    }),

  score: ({ datum, output }) => {
    const predicted = TICKET_LABELS.find((label) =>
      (output ?? "").toLowerCase().includes(label),
    );

    if (predicted === datum.label) {
      return { score: 1, feedback: `Correct: ${datum.label}.` };
    }

    // Feedback, not just a number. This paragraph is what the reflection model
    // reads, and it is the difference between GEPA and random prompt search.
    return {
      score: 0,
      feedback: [
        predicted === undefined
          ? `No label found in the response ("${(output ?? "").slice(0, 80)}"). The response must be exactly one of: ${TICKET_LABELS.join(", ")}.`
          : `Predicted "${predicted}" but the correct queue is "${datum.label}".`,
        datum.why,
      ].join(" "),
    };
  },

  // `inputs` has to mirror what the system actually receives. The default
  // record passes the whole row, and a row that carries `label` teaches the
  // reflection model to write rules about a field the task model never sees.
  // Ground truth still reaches reflection through `feedback`, where it belongs.
  buildRecord: ({ datum, output, score, feedback }) => ({
    inputs: datum.text,
    generatedOutputs: output,
    feedback,
    score,
  }),

  concurrency: 4,
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

console.log("\nper-instance validation scores for the best candidate:");
const best = result.candidates[result.bestCandidateId];
VAL_TICKETS.forEach((ticket, index) => {
  const score = best?.instanceScores[index] ?? 0;
  console.log(
    `  ${score === 1 ? "✓" : "✗"} ${ticket.id} (${ticket.label}) ${ticket.text}`,
  );
});
