import type { TextModel } from "textopt";
import type { GepaAdapter } from "textopt/gepa";
import type { KeywordExample, KeywordTrajectory } from "textopt/testing";
import { createKeywordAdapter } from "textopt/testing";

/**
 * The keyword fixture with a metric that cannot make up its mind.
 *
 * Real metrics are rarely exact. A judge model, a flaky integration test, a
 * human rater — each returns a slightly different number for the same rollout,
 * and a search that reads a single measurement as truth spends its budget
 * chasing that difference. The noise here is deterministic in the instance and
 * the candidate, so a run is still reproducible from its seed: two candidates
 * are perturbed differently, which is the part that misleads a search, but the
 * same candidate always reads the same on the same instance.
 */
export function createNoisyKeywordAdapter(
  amplitude = 0.25,
): GepaAdapter<KeywordExample, KeywordTrajectory, string> {
  const keyword = createKeywordAdapter();

  return {
    ...keyword,

    evaluate: async (args) => {
      const evaluation = await keyword.evaluate(args);
      const candidateText = Object.values(args.candidate).join(" ");

      return {
        ...evaluation,
        scores: evaluation.scores.map((score, index) => {
          const offset =
            (unitHash(`${candidateText}|${args.batch[index]?.question}`) -
              0.5) *
            2 *
            amplitude;
          return Math.min(1, Math.max(0, score + offset));
        }),
      };
    },
  };
}

/**
 * A stand-in for SIMBA's advice model, in the same spirit as
 * `createKeywordReflector`: it reads the failures the prompt carries and hands
 * every component the terms they were missing.
 *
 * The protocol is the interesting part. SIMBA asks for one `<advice>` block per
 * component, so the advice a module gets is specific to that module's job
 * rather than to the system as a whole — a real model is given the better and
 * worse trajectories and writes the contrast between them.
 */
export function createAdviceModel(): TextModel {
  return async ({ prompt }) => {
    const components = (
      prompt.match(/<components>\n([\s\S]*?)\n<\/components>/)?.[1] ?? ""
    )
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    const missing = new Set<string>();
    for (const match of prompt.matchAll(/Missing required terms: ([^"\n]+)/g)) {
      for (const term of (match[1] ?? "").split(",")) {
        missing.add(term.trim());
      }
    }

    return components
      .map(
        (component) =>
          `<advice component="${component}">${[...missing].join(" ")}</advice>`,
      )
      .join("\n");
  };
}

/** FNV-1a, folded into [0, 1). Any stable hash would do. */
function unitHash(text: string): number {
  let hash = 2166136261;

  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 2 ** 32;
}
