import { writeFileSync, mkdirSync } from "node:fs";
import { SimbaOptimizer } from "textopt/simba";
import { createKeywordAdapter } from "textopt/testing";
import config from "./config.mjs";
import { training, validation } from "./data.mjs";

const { maxMetricCalls, ...options } = config;
const result = await new SimbaOptimizer({ seed: 3, ...options }).optimize({
  seedCandidate: { instruction: "Answer the customer question." },
  trainingSet: training,
  validationSet: validation,
  adapter: createKeywordAdapter(),
  maxMetricCalls,
});

mkdirSync("out", { recursive: true });
writeFileSync(
  "out/result.json",
  JSON.stringify(
    {
      steps: result.steps,
      stopReason: result.stopReason,
      metricCalls: result.metricCalls,
      maxMetricCalls,
      validationSize: validation.length,
      config,
    },
    null,
    2,
  ),
);
console.log(`steps=${result.steps} stopReason=${result.stopReason} used=${result.metricCalls}`);
