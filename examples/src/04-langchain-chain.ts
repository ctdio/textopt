/**
 * Optimizing a LangChain chain.
 *
 * The adapter rebuilds the runnable for each candidate, so anything that reads
 * candidate text is optimizable — prompts, tool descriptions, routing rules.
 * Traces are captured through a plain callback handler, which means the same
 * events LangSmith records are available to the reflection model without a
 * LangSmith account. With LANGSMITH_TRACING=1 set, every rollout also carries
 * `textopt_iteration`, `textopt_phase`, `textopt_split` and
 * `textopt_candidate_id`, so a score that moved can be filtered back to the
 * rollouts that moved it.
 *
 * This one runs on OpenAI to make the point that none of the machinery is
 * vendor-specific — the AI SDK example next door is identical in structure and
 * runs on Claude.
 *
 *   OPENAI_API_KEY=... pnpm --filter textopt-examples langchain
 */
import { GepaOptimizer } from "textopt/gepa";
import { createLangChainAdapter } from "@textopt/langchain";
import { SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { openai } from "@ai-sdk/openai";
import { createReflector, requireApiKey } from "./shared/reflector.js";
import { logEvent, printResult } from "./shared/report.js";
import {
  TICKET_LABELS,
  TRAIN_TICKETS,
  VAL_TICKETS,
  type Ticket,
} from "./shared/tickets.js";

requireApiKey("OPENAI_API_KEY");

// The adapter takes a `Runnable`, so the chat model is an ordinary LangChain
// choice — swap in `new ChatAnthropic({ model: "claude-sonnet-5" })` and
// nothing else in this file changes.
const model = new ChatOpenAI({
  model: "gpt-5.6-luna",
  reasoning: { effort: "none" },
  maxCompletionTokens: 512,
});

// Reflection does not have to run on the same stack as the system under
// optimization: this is an AI SDK call driving a LangChain program.
const reflect = createReflector({
  model: openai("gpt-5.6"),
  providerOptions: { openai: { reasoningEffort: "high" } },
});

const adapter = createLangChainAdapter<Ticket, string>({
  buildRunnable: (candidate) =>
    ChatPromptTemplate.fromMessages([
      // A SystemMessage instance, not a ["system", text] tuple: candidate text
      // is arbitrary and will eventually contain braces, which a template
      // string would try to interpret as input variables.
      new SystemMessage(candidate.system ?? ""),
      ["human", "{ticket}"],
    ])
      .pipe(model)
      .pipe(new StringOutputParser()),

  toInput: (datum) => ({ ticket: datum.text }),

  score: ({ datum, output }) => {
    const predicted = TICKET_LABELS.find((label) =>
      (output ?? "").toLowerCase().includes(label),
    );

    if (predicted === datum.label) {
      return { score: 1, feedback: `Correct: ${datum.label}.` };
    }

    return {
      score: 0,
      feedback: [
        predicted === undefined
          ? `No label found in the response ("${(output ?? "").slice(0, 80)}"). Answer with exactly one of: ${TICKET_LABELS.join(", ")}.`
          : `Predicted "${predicted}" but the correct queue is "${datum.label}".`,
        datum.why,
      ].join(" "),
    };
  },

  // The default record carries inputs, output, feedback and the captured trace.
  // Override it when you want the reflection model to see something specific —
  // anything beyond the four standard fields goes in `evidence`, which is the
  // record's one adapter-owned slot.
  buildRecord: ({ datum, output, trace, score, feedback }) => ({
    inputs: datum.text,
    generatedOutputs: output,
    feedback,
    score,
    evidence: {
      llmCalls: trace.steps.filter((step) => step.type === "llm").length,
    },
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
  reporters: [{ onEvent: logEvent }],
});

printResult(result);
