/**
 * Oracle for enforces-a-requirement-once.
 *
 * Imports the agent's metric and measures the interval it actually moves in.
 * Every judgment is a property of the returned numbers, so a gate, an early
 * return, a multiplicative veto, or a filter upstream all satisfy the gate
 * checks equally — what is graded is the range left over, not how it was built.
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
    problems.push(`aggregate threw on ${JSON.stringify(grades)}: ${err.message}`);
    return NaN;
  }
};

const OTHERS = ["greeting", "brevity", "warmth", "accuracy"];

// Every candidate a search could still be choosing between: the ones that
// clear the non-negotiable requirement. Among these the policy grade is
// constant, so any share of the aggregate spent on it is share the search
// cannot move.
const surviving = [];
for (let mask = 0; mask < 6 ** OTHERS.length; mask += 1) {
  let rest = mask;
  const grades = { policy: 5 };
  for (const name of OTHERS) {
    grades[name] = rest % 6;
    rest = Math.floor(rest / 6);
  }
  surviving.push(grades);
}

const scores = problems.length ? [] : surviving.map(call);
const finite = scores.filter((score) => Number.isFinite(score));
const low = Math.min(...finite);
const high = Math.max(...finite);
const span = finite.length === scores.length && finite.length > 0 ? high - low : NaN;

const violating = { policy: 0, greeting: 5, brevity: 5, warmth: 5, accuracy: 5 };
const partial = { ...violating, policy: 4 };

const gateZeroes = problems.length === 0 && call(violating) === 0;
const gateIsStrict = problems.length === 0 && call(partial) === 0;

// A metric whose whole span is 0.4 denominates every later number in 0.4. The
// gate already enforces the requirement; weight spent on it among candidates
// that all cleared it is dead share.
const spanIsUsable = Number.isFinite(span) && span >= 0.75;

const verdict = {
  metricLoaded: problems.length === 0,
  gateZeroes,
  gateIsStrict,
  spanIsUsable,
  detail: {
    realizedSpan: Number.isFinite(span) ? Number(span.toFixed(4)) : null,
    lowestSurviving: Number.isFinite(low) ? Number(low.toFixed(4)) : null,
    highestSurviving: Number.isFinite(high) ? Number(high.toFixed(4)) : null,
    survivingVectors: surviving.length,
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
