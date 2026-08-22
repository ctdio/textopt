/**
 * The smallest end-to-end run of the GEPA optimizer: no API keys, no network,
 * no LLM.
 *
 * `textopt/testing` ships a deterministic stand-in for both halves of
 * the loop — a system under optimization and a reflection model — so you can see
 * the whole mechanism before spending a single token.
 *
 * The two objects below are the whole API: the constructor takes the settings
 * that are stateless and free of your types, and `optimize` takes one problem.
 * An optimizer holds no run state, so one instance can run any number of them.
 *
 *   pnpm --filter textopt-examples keyword
 */
import { consoleReporter, createReporter } from "textopt";
import { GepaOptimizer } from "textopt/gepa";
import type { GepaEvent } from "textopt/gepa";
import {
  KEYWORD_EXAMPLES,
  createKeywordAdapter,
  createKeywordReflector,
} from "textopt/testing";

const gepa = new GepaOptimizer({
  minibatchSize: 2,
  seed: 7,
});

// The component names come from `seedCandidate`, so `result.bestCandidate`
// below has an `instruction` field and a misspelling here is a compile error.
const result = await gepa.optimize({
  seedCandidate: {
    instruction: "Answer the customer's question.",
  },
  trainingSet: KEYWORD_EXAMPLES,
  adapter: createKeywordAdapter(),
  reflect: createKeywordReflector(),
  maxMetricCalls: 120,
  reporters: [
    // Prints the run as it happens, one line per event, for any optimizer.
    consoleReporter({ level: "quiet" }),

    // Your own reporter: one handler per event rather than one callback that
    // switches on a tag. The keys are `GepaEvent`'s own tags, so a handler
    // named after an event that does not exist is a compile error instead of
    // a callback nothing ever calls.
    createReporter<GepaEvent>({
      on: {
        candidateRejected: (event) =>
          console.log(
            `  rejected child of #${event.parentId}:` +
              ` ${event.childScore.toFixed(3)} <= ${event.parentScore.toFixed(3)}`,
          ),
      },
    }),
  ],
});

console.log("\nseed score:", result.candidates[0]?.aggregateScore.toFixed(3));
console.log("best score:", result.bestScore.toFixed(3));
console.log("best candidate:", result.bestCandidate);
console.log(
  `\n${result.metricCalls} metric calls, ${result.cacheHits} cache hits, ` +
    `${result.iterations} iterations, stopped because ${result.stopReason}`,
);
