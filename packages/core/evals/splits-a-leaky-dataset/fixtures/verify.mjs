/**
 * Oracle for splits-a-leaky-dataset.
 *
 * Re-derives every judgment from the dataset and the agent's own result file,
 * and writes out/verdict.json. Graders read the verdict, never the agent's
 * artefact.
 *
 * Deliberately does NOT require the splits to cover every row: dropping a
 * near-duplicate family is a legitimate way to remove the leak, and a check
 * that insisted on full coverage would fail a correct answer for choosing a
 * different mechanism.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const rows = readFileSync("data/tickets.jsonl", "utf8")
  .split("\n")
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const familyOf = new Map();
for (const row of rows) {
  familyOf.set(row.id, [...row.required].sort().join("|"));
}

const problems = [];
let result;
try {
  result = JSON.parse(readFileSync("out/result.json", "utf8"));
} catch (err) {
  problems.push(`out/result.json unreadable: ${err.message}`);
}

const splits = result?.splits ?? {};
const training = splits.training ?? [];
const validation = splits.validation ?? [];
const test = splits.test ?? [];
const named = { training, validation, test };

const splitsPresent =
  [training, validation, test].every(
    (ids) => Array.isArray(ids) && ids.length > 0,
  ) && problems.length === 0;

const known = new Set(rows.map((row) => row.id));
const unknownIds = Object.values(named)
  .flat()
  .filter((id) => !known.has(id));

const seen = new Map();
let splitsDisjoint = true;
for (const [name, ids] of Object.entries(named)) {
  for (const id of ids) {
    if (seen.has(id)) {
      splitsDisjoint = false;
      problems.push(`${id} appears in both ${seen.get(id)} and ${name}`);
    }
    seen.set(id, name);
  }
}

const familySplits = new Map();
for (const [name, ids] of Object.entries(named)) {
  for (const id of ids) {
    const family = familyOf.get(id);
    if (!family) continue;
    if (!familySplits.has(family)) familySplits.set(family, new Set());
    familySplits.get(family).add(name);
  }
}
const straddling = [...familySplits.entries()]
  .filter(([, where]) => where.size > 1)
  .map(([family, where]) => `${family} -> ${[...where].sort().join("+")}`);
const familiesIntact = straddling.length === 0;

const usedCategories = new Set(
  rows.filter((row) => seen.has(row.id)).map((row) => row.category),
);
const missingCoverage = [];
for (const category of usedCategories) {
  for (const [name, ids] of Object.entries(named)) {
    const present = ids.some(
      (id) => rows.find((row) => row.id === id)?.category === category,
    );
    if (!present) missingCoverage.push(`${category} absent from ${name}`);
  }
}
const categoriesCovered = missingCoverage.length === 0;

const warnings = result?.warnings ?? [];
const codes = warnings.map((warning) => warning?.code ?? warning);
const noReuseWarning = !codes.includes("validationSetReusesTraining");

const testScore = result?.testScore;
const reported = result?.reported;
const reportedIsTestScore =
  typeof testScore === "number" &&
  typeof reported === "number" &&
  Math.abs(testScore - reported) < 1e-9;

const verdict = {
  splitsPresent,
  splitsDisjoint,
  familiesIntact,
  categoriesCovered,
  noReuseWarning,
  reportedIsTestScore,
  detail: {
    counts: {
      training: training.length,
      validation: validation.length,
      test: test.length,
      dropped: rows.length - seen.size,
    },
    familiesTotal: new Set(familyOf.values()).size,
    straddling,
    missingCoverage,
    unknownIds,
    warningCodes: codes,
    testScore: testScore ?? null,
    reported: reported ?? null,
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
