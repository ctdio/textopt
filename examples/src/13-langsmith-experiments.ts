/**
 * Watching the hill climb in LangSmith.
 *
 * The reporter writes one experiment per accepted candidate over one fixed
 * dataset — the validation split, uploaded once. Selecting several of those
 * experiments in LangSmith's comparison view renders the candidate x instance
 * score matrix GEPA's Pareto selection reads: which instances a candidate won,
 * and which it paid for. A dataset per iteration would give you N datasets
 * with one experiment each and nothing to compare.
 *
 * The held-out sweep gets exactly one experiment, for the winner, on its own
 * dataset. That separation is deliberate: GEPA scores the held-out set only
 * after selection is over, and a per-candidate view of it is how a held-out
 * number stops meaning anything.
 *
 * Traces are the other half. With LANGSMITH_TRACING=1 the LangChain adapter
 * tags every rollout with its iteration, phase, split and candidate id, so the
 * project holds the individual rollouts while the datasets hold the scores.
 *
 *   OPENAI_API_KEY=... LANGSMITH_API_KEY=... pnpm --filter textopt-examples langsmith
 */
import { consoleReporter } from "textopt";
import { GepaOptimizer } from "textopt/gepa";
import { createLangChainAdapter } from "@textopt/langchain";
import { createLangSmithReporter } from "@textopt/langsmith";
import { SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { openai } from "@ai-sdk/openai";
import { Client } from "langsmith";
import { createReflector, requireApiKey } from "./shared/reflector.js";
import { printResult } from "./shared/report.js";
import {
  TICKET_LABELS,
  TRAIN_TICKETS,
  VAL_TICKETS,
  type Ticket,
} from "./shared/tickets.js";

requireApiKey("OPENAI_API_KEY");
requireApiKey("LANGSMITH_API_KEY");

/** Never scored until the search is over, and never selected against. */
const HELD_OUT_TICKETS: Ticket[] = [
  {
    id: "h1",
    text: "Charged twice for the same invoice after retrying a failed payment.",
    label: "billing",
    why: "Duplicate charge is billing.",
  },
  {
    id: "h2",
    text: "Export to CSV truncates every description at 80 characters.",
    label: "bug",
    why: "Existing feature behaving incorrectly.",
  },
  {
    id: "h3",
    text: "Can we get SSO through Okta for the whole org?",
    label: "feature_request",
    why: "Capability that does not exist yet.",
  },
  {
    id: "h4",
    text: "Password reset emails never arrive for one of our teammates.",
    label: "account",
    why: "Access to an account.",
  },
];

const model = new ChatOpenAI({
  model: "gpt-5.6-luna",
  reasoning: { effort: "none" },
  maxCompletionTokens: 512,
});

const reflect = createReflector({
  model: openai("gpt-5.6"),
  providerOptions: { openai: { reasoningEffort: "high" } },
});

const adapter = createLangChainAdapter<Ticket, string>({
  buildRunnable: (candidate) =>
    ChatPromptTemplate.fromMessages([
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
          ? `No label found in the response. Answer with exactly one of: ${TICKET_LABELS.join(", ")}.`
          : `Predicted "${predicted}" but the correct queue is "${datum.label}".`,
        datum.why,
      ].join(" "),
    };
  },

  concurrency: 4,
});

/**
 * The reporter is given the same splits and the same `instanceId` as the run.
 * The id is what a row is keyed by, so a later run against the same dataset
 * writes to the rows it already has instead of a fresh copy of them.
 */
const instanceId = ({ datum }: { datum: Ticket }) => datum.id;

const langsmith = createLangSmithReporter<Ticket>({
  client: new Client(),
  dataset: "textopt-ticket-triage",
  experimentPrefix: `gepa-${process.env["USER"] ?? "local"}`,
  validationSet: VAL_TICKETS,
  testSet: HELD_OUT_TICKETS,
  instanceId,
  toInput: (datum) => ({ ticket: datum.text }),
  toExpected: (datum) => ({ label: datum.label }),
});

const gepa = new GepaOptimizer({
  minibatchSize: 3,
  seed: 11,
  // Without it the experiments carry scores but no outputs, and half of what
  // an experiment is worth opening for is what the candidate actually said.
  trackBestOutputs: true,
});

const result = await gepa.optimize({
  seedCandidate: {
    system: "Classify the support ticket. Answer with one word.",
  },
  trainingSet: TRAIN_TICKETS,
  validationSet: VAL_TICKETS,
  testSet: HELD_OUT_TICKETS,
  adapter,
  reflect,
  maxMetricCalls: 150,
  instanceId,
  reporters: [consoleReporter(), langsmith],
});

printResult(result);

console.log(
  `\nheld out: ${result.testScore?.toFixed(3) ?? "n/a"}` +
    ` over ${HELD_OUT_TICKETS.length} instances the search never saw`,
);
