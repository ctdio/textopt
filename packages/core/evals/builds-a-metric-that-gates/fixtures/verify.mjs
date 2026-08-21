/**
 * Oracle for builds-a-metric-that-gates.
 *
 * Imports the agent's own metric and executes it. Every judgment is a property
 * of what the function returns, so any correct implementation passes: a gate,
 * a multiplicative veto, an early return, or a filter upstream all satisfy
 * `gateZeroes` equally, and the ticket-ID check cares only that the number does
 * not move.
 */
import { writeFileSync, mkdirSync } from "node:fs";

const problems = [];
let aggregate;
try {
  ({ aggregate } = await import("./metric.mjs"));
} catch (err) {
  problems.push(`metric.mjs did not import: ${err.message}`);
}
if (typeof aggregate !== "function") {
  problems.push("metric.mjs does not export a function named aggregate");
}

const call = (grades) => {
  try {
    const value = aggregate(grades);
    return typeof value === "number" && Number.isFinite(value) ? value : NaN;
  } catch (err) {
    problems.push(
      `aggregate threw on ${JSON.stringify(grades)}: ${err.message}`,
    );
    return NaN;
  }
};

const charming = { policy: 0, greeting: 5, brevity: 5, warmth: 5, ticketId: 5 };
const compliant = { ...charming, policy: 5 };
const withId = { policy: 5, greeting: 3, brevity: 4, warmth: 3, ticketId: 5 };
const withoutId = { ...withId, ticketId: 0 };
const partialPolicy = { ...charming, policy: 4 };

const charmingScore = problems.length ? NaN : call(charming);
const compliantScore = problems.length ? NaN : call(compliant);
const withIdScore = problems.length ? NaN : call(withId);
const withoutIdScore = problems.length ? NaN : call(withoutId);
const partialScore = problems.length ? NaN : call(partialPolicy);

// A reply that invents a refund window scores zero however well it did on
// everything else. This is the whole point of a gate, and the check a weighted
// average cannot pass no matter how heavy the weight.
const gateZeroes = charmingScore === 0;

// The same reply with the violation removed must be worth something, or the
// metric is not a gate, it is just always zero.
const gateIsNotTotal = compliantScore > 0;

// Criterion 5 contributes exactly nothing: two replies differing only in the
// ticket-ID grade must produce the identical number. A small weight fails here
// with certainty rather than at the mercy of a judge's opinion.
const ticketIdInert =
  Number.isFinite(withIdScore) &&
  Number.isFinite(withoutIdScore) &&
  withIdScore === withoutIdScore;

// A partial policy grade is still a violation of a non-negotiable requirement.
// Passing this needs a gate threshold at the top of the scale, not merely > 0.
const gateIsStrict = partialScore === 0;

const verdict = {
  metricLoaded: problems.length === 0,
  gateZeroes,
  gateIsNotTotal,
  ticketIdInert,
  gateIsStrict,
  detail: {
    charmingButInventedRefundWindow: charmingScore,
    sameReplyCompliant: compliantScore,
    withTicketId: withIdScore,
    withoutTicketId: withoutIdScore,
    partialPolicyGrade: partialScore,
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
