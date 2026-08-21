/**
 * Oracle for funds-the-simba-reserve.
 *
 * Checks the outcome, not the route. Raising maxMetricCalls past the reserve,
 * shrinking the minibatch, and cutting the candidate count all reach eight
 * steps with the validation set intact, and all three are correct answers.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const problems = [];
let result;
try {
  result = JSON.parse(readFileSync("out/result.json", "utf8"));
} catch (err) {
  problems.push(`out/result.json unreadable: ${err.message} — run node run.mjs first`);
}

const steps = result?.steps;
const stopReason = result?.stopReason;
const used = result?.metricCalls;
const budget = result?.maxMetricCalls;

const completedAllSteps = steps === 8 && stopReason === "maxSteps";
const validationIntact = result?.validationSize === 50;
// Deliberately lenient: it only catches brute force, because the cheapest
// correct answers still leave real headroom between budget and spend.
const budgetNotBruteForced =
  typeof budget === "number" && typeof used === "number" && budget <= used * 3;

const verdict = {
  ranTheOptimizer: problems.length === 0,
  completedAllSteps,
  validationIntact,
  budgetNotBruteForced,
  detail: { steps, stopReason, metricCalls: used, maxMetricCalls: budget, config: result?.config, problems },
};

mkdirSync("out", { recursive: true });
writeFileSync("out/verdict.json", JSON.stringify(verdict, null, 2));
for (const [key, value] of Object.entries(verdict)) {
  if (key === "detail") continue;
  console.log(`${value ? "PASS" : "FAIL"} ${key}`);
}
console.log(JSON.stringify(verdict.detail, null, 2));
