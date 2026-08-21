import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { GepaOptimizer } from "textopt/gepa";
import { createKeywordReflector } from "textopt/testing";
import { createAdapter } from "./metric.mjs";
import { training, validation } from "./data.mjs";

const result = await new GepaOptimizer({ seed: 7 }).optimize({
  seedCandidate: { instruction: readFileSync("prompt.txt", "utf8").trim() },
  trainingSet: training,
  validationSet: validation,
  adapter: createAdapter(),
  reflect: createKeywordReflector(),
  maxMetricCalls: 2000,
});

mkdirSync("out", { recursive: true });
writeFileSync(
  "out/result.json",
  JSON.stringify(
    {
      bestScore: result.bestScore,
      stopReason: result.stopReason,
      metricCalls: result.metricCalls,
      warnings: result.warnings,
      best: result.bestCandidate,
    },
    null,
    2,
  ),
);
console.log(
  `bestScore=${result.bestScore} stopReason=${result.stopReason} used=${result.metricCalls}`,
);
