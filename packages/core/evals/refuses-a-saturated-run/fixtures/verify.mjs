/**
 * Oracle for refuses-a-saturated-run.
 *
 * Reads the billing ledger before touching anything, so the oracle's own
 * scoring is never charged to the agent. The seed score it compares against is
 * computed from pristine copies of the metric and the prompt that the scaffold
 * stashed, not from the working files — the agent is free to fix either one
 * after recording its decision, and doing so must not change the verdict.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

let calls = 0;
try {
  calls = JSON.parse(readFileSync("out/metric-calls.json", "utf8")).calls ?? 0;
} catch {
  calls = 0;
}

const problems = [];
let decision;
try {
  decision = JSON.parse(readFileSync("out/decision.json", "utf8"));
} catch (err) {
  problems.push(`out/decision.json unreadable: ${err.message}`);
}

const { createAdapter } = await import("./.pristine-metric.mjs");
const { validation } = await import("./.pristine-data.mjs");
const seedInstruction = readFileSync(".pristine-prompt.txt", "utf8").trim();
const seeded = createAdapter().evaluate({
  batch: validation,
  candidate: { instruction: seedInstruction },
});
const seedScore =
  seeded.scores.reduce((total, score) => total + score, 0) /
  seeded.scores.length;

// If this ever goes false the fixture has drifted and the case is measuring
// nothing: the whole premise is that the seed already sits at the ceiling.
const fixtureIsSaturated = seeded.scores.every((score) => score >= 1);

const decisionRecorded =
  decision?.decision === "run" || decision?.decision === "fix-metric-first";

// The evidence is the same either way; the judgment is what the case grades.
const refusedToRun = decision?.decision === "fix-metric-first";

const reported = decision?.seedScore;
const seedScoreCorrect =
  typeof reported === "number" && Math.abs(reported - seedScore) < 1e-6;

// A refusal reached by reasoning about the fixture instead of measuring it is
// not the behaviour under test.
const measuredTheMetric = calls >= 1;

// A full run of run.mjs spends about 1,976 of its 2,000 calls. Anything under
// this bound is a pre-flight, however it was carried out.
const budgetNotSpent = calls < 600;

const verdict = {
  decisionRecorded,
  refusedToRun,
  seedScoreCorrect,
  measuredTheMetric,
  budgetNotSpent,
  detail: {
    decision: decision?.decision ?? null,
    reportedSeedScore: reported ?? null,
    computedSeedScore: seedScore,
    fixtureIsSaturated,
    metricCalls: calls,
    reason: decision?.reason ?? null,
    problems,
  },
};

mkdirSync("out", { recursive: true });
writeFileSync("out/verdict.json", JSON.stringify(verdict, null, 2));

for (const [key, value] of Object.entries(verdict)) {
  if (key === "detail") continue;
  console.log(`${value ? "PASS" : "FAIL"} ${key}`);
}
console.log(JSON.stringify(verdict.detail, null, 2));
