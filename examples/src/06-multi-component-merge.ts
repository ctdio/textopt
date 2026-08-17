/**
 * Two prompts, one system — and GEPA's system-aware merge.
 *
 * The pipeline routes a ticket and then drafts a reply, with a separate text
 * component for each stage. Because the two components fail for different
 * reasons, lineages tend to improve them independently: one branch learns the
 * routing rules, another learns the reply format. Merge is what recombines
 * them, taking the `router` from one ancestor and the `replyStyle` from another
 * without re-running the search.
 *
 * Merge is enabled by default whenever a candidate has more than one component.
 *
 *   ANTHROPIC_API_KEY=... pnpm --filter @ctdio/gepa-examples merge
 */
import { anthropic } from "@ai-sdk/anthropic";
import { optimize } from "@ctdio/gepa";
import { createAiSdkAdapter } from "@ctdio/gepa-ai-sdk";
import { generateText } from "ai";
import { createReflector, requireApiKey } from "./shared/reflector.js";
import { logEvent, printResult } from "./shared/report.js";
import {
  TICKET_LABELS,
  TRAIN_TICKETS,
  VAL_TICKETS,
  type Ticket,
  type TicketLabel,
} from "./shared/tickets.js";

requireApiKey("ANTHROPIC_API_KEY");

interface Draft {
  label: string;
  reply: string;
}

/** The action a reply must promise, per queue. */
const REQUIRED_ACTION: Record<TicketLabel, string[]> = {
  billing: ["refund", "invoice", "credit"],
  bug: ["engineering", "reproduce", "fix"],
  account: ["verify", "identity", "access"],
  feature_request: ["roadmap", "product team", "track"],
};

const taskModel = anthropic("claude-sonnet-5");
const reflect = createReflector({
  model: anthropic("claude-opus-5"),
  providerOptions: { anthropic: { thinking: { type: "adaptive" } } },
});

const adapter = createAiSdkAdapter<Ticket, Draft>({
  run: async ({ candidate, datum, signal }) => {
    const routing = await generateText({
      model: taskModel,
      system: candidate.router ?? "",
      prompt: datum.text,
      maxOutputTokens: 512,
      abortSignal: signal,
    });

    const reply = await generateText({
      model: taskModel,
      system: candidate.replyStyle ?? "",
      prompt: `Ticket: ${datum.text}\nQueue: ${routing.text}\n\nWrite the reply.`,
      maxOutputTokens: 800,
      abortSignal: signal,
    });

    // `AiSdkResultLike` is structural, so two calls can be stitched into one
    // result. The reflection model then sees both stages in the trace.
    return {
      text: reply.text,
      finishReason: reply.finishReason,
      steps: [...routing.steps, ...reply.steps],
      usage: reply.usage,
    };
  },

  toOutput: (result) => ({
    label: result.steps?.[0]?.text ?? "",
    reply: result.text ?? "",
  }),

  score: ({ datum, output }) => {
    const predicted = TICKET_LABELS.find((label) =>
      (output?.label ?? "").toLowerCase().includes(label),
    );
    const reply = (output?.reply ?? "").toLowerCase();
    const words = reply.split(/\s+/).filter(Boolean).length;

    const routingScore = predicted === datum.label ? 1 : 0;
    const actionScore = REQUIRED_ACTION[datum.label].some((action) =>
      reply.includes(action),
    )
      ? 1
      : 0;
    const brevityScore = words > 0 && words <= 60 ? 1 : 0;

    // Per-component feedback in one string: the reflection model is told which
    // component each sentence is about, so a routing failure does not push the
    // reply prompt around.
    const notes = [
      routingScore === 1
        ? `router: correct (${datum.label}).`
        : `router: answered "${output?.label ?? ""}" but the correct queue is "${datum.label}". ${datum.why}`,
      actionScore === 1
        ? "replyStyle: promised the right next action."
        : `replyStyle: a ${datum.label} reply must commit to a concrete next action (one of: ${REQUIRED_ACTION[datum.label].join(", ")}).`,
      brevityScore === 1
        ? `replyStyle: length ok (${words} words).`
        : `replyStyle: ${words} words — replies must be 60 words or fewer.`,
    ];

    return {
      score: 0.5 * routingScore + 0.3 * actionScore + 0.2 * brevityScore,
      feedback: notes.join("\n"),
      objectiveScores: {
        routing: routingScore,
        action: actionScore,
        brevity: brevityScore,
      },
    };
  },

  concurrency: 3,
});

const result = await optimize({
  seedCandidate: {
    router: "Classify the support ticket. Answer with one word.",
    replyStyle: "Write a reply to the customer.",
  },
  trainset: TRAIN_TICKETS,
  valset: VAL_TICKETS,
  adapter,
  reflect,
  maxMetricCalls: 120,
  minibatchSize: 3,
  seed: 4,
  instanceId: ({ datum }) => datum.id,
  merge: { enabled: true, maxInvocations: 4 },
  onEvent: logEvent,
});

printResult(result);

console.log("\nlineage:");
for (const candidate of result.candidates) {
  console.log(
    `  #${candidate.id}  ${candidate.source.padEnd(8)}` +
      ` parents=[${candidate.parentIds.join(", ")}]` +
      ` changed=[${candidate.updatedComponents.join(", ")}]` +
      ` score=${candidate.aggregateScore.toFixed(3)}`,
  );
}

const merged = result.candidates.filter(
  (candidate) => candidate.source === "merge",
);
console.log(
  merged.length === 0
    ? "\nno merge was accepted this run — no two lineages improved disjoint components."
    : `\n${merged.length} merged candidate(s) accepted.`,
);
