/**
 * Deciding which optimizer to use, with numbers instead of a preference.
 *
 * Three entrants on the same task, the same budget, and the same seeds. The
 * comparison matters because a single run of two optimizers is two anecdotes:
 * every optimizer here is deterministic given its seed, so one run tells you
 * what that seed did and nothing about what the next one will.
 *
 * `compare` runs the grid, ranks on the held-out score where a run reports one,
 * and gives every entrant a p-value against the winner from a paired sign-flip
 * test. Read that column before believing the ranking — a gap in means over
 * eight seeds is usually noise.
 *
 * Offline, so nothing here costs anything.
 *
 *   pnpm --filter textopt-examples compare
 */
import { compare } from "textopt";
import { GepaOptimizer } from "textopt/gepa";
import { OproOptimizer } from "textopt/opro";
import { SimbaOptimizer } from "textopt/simba";
import {
  KEYWORD_EXAMPLES,
  createHillClimbingReflector,
  createKeywordReflector,
} from "textopt/testing";
import {
  createAdviceModel,
  createNoisyKeywordAdapter,
} from "./shared/noisy-keyword.js";
import { createAcceptanceTally } from "./shared/report.js";

const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8];
const BUDGET = 200;

// The held-out rows ask the training questions in different words and want the
// same terms back, so a run that learned the terms scores here and a run that
// learned the training questions does not.
const TEST = [
  {
    question: "My device is frozen — how do I restart it?",
    required: ["hold", "ten seconds"],
  },
  { question: "Where do I file an issue?", required: ["ticket", "portal"] },
  {
    question: "How long do I have to return something?",
    required: ["thirty days"],
  },
];

const shared = {
  seedCandidate: { instruction: "Answer the customer's question." },
  trainingSet: KEYWORD_EXAMPLES.slice(0, 3),
  // Never selected against. `compare` ranks on this where it exists, because
  // the validation score is the number every entrant spent its whole run
  // fitting, and an entrant that overfits looks strongest on exactly that.
  testSet: TEST,
  maxMetricCalls: BUDGET,
};

// One observer for all three searches. Every optimizer emits the same
// acceptance payload, so this reads GEPA, SIMBA and OPRO without knowing which
// of them produced an event — the comparison below would otherwise be three
// numbers with no account of how each search got there.
const tally = createAcceptanceTally();

const comparison = await compare({
  seeds: SEEDS,
  entrants: {
    // Entrants are functions of a seed rather than optimizer instances: the
    // seed is constructor config, and the task is built here because only the
    // caller knows the optimizer-specific parts of it.
    gepa: ({ seed }) =>
      new GepaOptimizer({ minibatchSize: 2, seed }).optimize({
        ...shared,
        adapter: createNoisyKeywordAdapter(),
        reflect: createKeywordReflector(),
        reporters: [tally.for("gepa")],
      }),

    simba: ({ seed }) =>
      new SimbaOptimizer({
        minibatchSize: 3,
        candidates: 2,
        maxSteps: 20,
        seed,
      }).optimize({
        ...shared,
        adapter: createNoisyKeywordAdapter(),
        reflect: createAdviceModel(),
        reporters: [tally.for("simba")],
      }),

    // A third family, and a useful control: OPRO shows the proposal model a
    // score history and nothing else. If it keeps up, the written feedback the
    // other two spend their budget producing was not buying anything here.
    opro: ({ seed }) =>
      new OproOptimizer({ proposalsPerRound: 2, seed }).optimize({
        ...shared,
        adapter: createNoisyKeywordAdapter(),
        // A different stand-in, because OPRO's prompt carries a different kind
        // of evidence: this one reads the best attempt out of the score history
        // and extends it, where the reflector the other two use reads written
        // feedback. Handing OPRO a proposer that needs feedback would measure
        // the mismatch rather than the search. It also ignores the seed, which
        // is why OPRO's spread below is exactly zero.
        reflect: createHillClimbingReflector(),
        reporters: [tally.for("opro")],
      }),
  },
});

console.log(`winner: ${comparison.winner}\n`);
console.log(
  ["entrant", "mean", "sd", "min", "max", "calls", "p vs winner"]
    .map((heading) => heading.padEnd(13))
    .join(""),
);

for (const summary of comparison.summaries) {
  console.log(
    [
      summary.entrant,
      summary.meanScore.toFixed(3),
      summary.sdScore.toFixed(3),
      summary.minScore.toFixed(3),
      summary.maxScore.toFixed(3),
      summary.meanMetricCalls.toFixed(0),
      summary.pValueVsWinner === undefined
        ? "—"
        : summary.pValueVsWinner.toFixed(3),
    ]
      .map((cell) => cell.padEnd(13))
      .join(""),
  );
}

console.log(
  `\n${comparison.runs.length} runs over ${SEEDS.length} seeds.` +
    " A p-value above 0.05 means the ranking above could be seed luck.",
);

// The same reporter read all three searches. What it counted is how each one
// spent its budget: how often the incumbent moved at all, and how far it moved
// from the seed the run started at.
console.log(
  `\n${["entrant", "accepted", "seed", "best"].map((heading) => heading.padEnd(13)).join("")}`,
);

for (const row of tally.rows()) {
  console.log(
    [
      row.entrant,
      String(row.accepted),
      row.seedScore.toFixed(3),
      row.bestScore.toFixed(3),
    ]
      .map((cell) => cell.padEnd(13))
      .join(""),
  );
}
