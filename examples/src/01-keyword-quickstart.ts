/**
 * The smallest end-to-end GEPA run: no API keys, no network, no LLM.
 *
 * `@ctdio/gepa/testing` ships a deterministic stand-in for both halves of the
 * loop — a system under optimization and a reflection model — so you can see the
 * whole mechanism before spending a single token.
 *
 *   pnpm --filter @ctdio/gepa-examples keyword
 */
import { optimize } from "@ctdio/gepa";
import {
  KEYWORD_EXAMPLES,
  createKeywordAdapter,
  createKeywordReflector,
} from "@ctdio/gepa/testing";

const result = await optimize({
  seedCandidate: {
    instruction: "Answer the customer's question.",
  },
  trainset: KEYWORD_EXAMPLES,
  adapter: createKeywordAdapter(),
  reflect: createKeywordReflector(),
  maxMetricCalls: 120,
  minibatchSize: 2,
  seed: 7,
  onEvent: (event) => {
    if (event.type === "candidateAccepted") {
      console.log(
        `  accepted #${event.candidateId} (${event.source}) score=${event.aggregateScore.toFixed(3)}`,
      );
    }
    if (event.type === "candidateRejected") {
      console.log(
        `  rejected child of #${event.parentId}: ${event.childScore.toFixed(3)} <= ${event.parentScore.toFixed(3)}`,
      );
    }
  },
});

console.log("\nseed score:", result.candidates[0]?.aggregateScore.toFixed(3));
console.log("best score:", result.bestScore.toFixed(3));
console.log("best candidate:", result.bestCandidate);
console.log(
  `\n${result.metricCalls} metric calls, ${result.cacheHits} cache hits, ` +
    `${result.iterations} iterations, stopped because ${result.stopReason}`,
);
