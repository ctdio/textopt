/**
 * SIMBA, on the kind of metric it exists for: one that returns a slightly
 * different number every time you ask it.
 *
 * Every other optimizer here scores a candidate over the whole validation set
 * before deciding anything. That is the right call when the metric is exact,
 * and wasteful when it is not — most of a sweep goes into re-measuring noise.
 * SIMBA works a minibatch at a time and spends its attention where the rollouts
 * disagreed with each other, on the theory that an instance every attempt gets
 * right teaches nothing.
 *
 * No API keys and no network: the metric, the system, and the advice model are
 * all deterministic stand-ins.
 *
 *   pnpm --filter textopt-examples simba
 */
import { SimbaOptimizer } from "textopt/simba";
import { KEYWORD_EXAMPLES } from "textopt/testing";
import {
  createAdviceModel,
  createNoisyKeywordAdapter,
} from "./shared/noisy-keyword.js";

const simba = new SimbaOptimizer({
  // Both defaults are far higher (32 and 6). This task has four instances, so
  // the minibatch is the whole set and two candidates per step is enough to
  // produce a disagreement to read.
  minibatchSize: 4,
  candidates: 2,
  maxSteps: 12,
  seed: 7,
});

const result = await simba.optimize({
  seedCandidate: { instruction: "Answer the customer's question." },
  trainingSet: KEYWORD_EXAMPLES,
  adapter: createNoisyKeywordAdapter(),
  // Advice is written per component, so a system with several instructions gets
  // a separate rule for each rather than one note addressed to all of them.
  reflect: createAdviceModel(),
  maxMetricCalls: 400,
  onEvent: (event) => {
    if (event.type === "stepStart") {
      console.log(`\nstep ${event.step} (pool of ${event.poolSize} programs)`);
    }
    if (event.type === "candidate") {
      console.log(
        `  ${event.strategy} from program #${event.sourceProgram}` +
          ` scored ${event.minibatchScore.toFixed(3)} on the minibatch`,
      );
    }
  },
});

console.log(
  `\nseed ${result.seedScore.toFixed(3)} -> best ${result.bestScore.toFixed(3)}` +
    ` after ${result.steps} steps`,
);
console.log(
  `${result.metricCalls} metric calls, ${result.reflectionCalls} advice calls,` +
    ` stopped because ${result.stopReason}`,
);

// The finalists are the programs the pool ended up holding, each scored over
// the full validation set. The winner is the top one — and the spread below it
// is what a single mini-batch measurement could not have told you.
console.log("\nfinalists:");
for (const finalist of result.finalists) {
  console.log(
    `  ${finalist.score.toFixed(3)}  ${finalist.candidate.instruction}`,
  );
}
