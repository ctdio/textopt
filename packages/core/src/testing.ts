import { buildReflectionPrompt } from "./gepa/reflection.js";
import type { GepaAdapter } from "./gepa/types.js";
import type { TextModel } from "./types.js";

/**
 * A deterministic, LLM-free system under optimization.
 *
 * It exists so the engine, and any adapter you write, can be exercised end to
 * end in milliseconds: the "model" answers with the candidate text itself and
 * the metric rewards covering the required terms. Optimization therefore has a
 * real gradient to climb without a single network call.
 */
export interface KeywordExample {
  question: string;
  required: string[];
}

export interface KeywordTrajectory {
  question: string;
  instruction: string;
  answer: string;
  missing: string[];
}

export const KEYWORD_EXAMPLES: KeywordExample[] = [
  { question: "How do I reset a device?", required: ["hold", "ten seconds"] },
  { question: "How do I contact support?", required: ["ticket", "portal"] },
  { question: "What is the refund window?", required: ["thirty days"] },
  { question: "How do I upgrade a plan?", required: ["billing", "prorated"] },
];

/** Terms the keyword metric rewards, interleaved with terms it ignores. */
export const SAMPLING_POOL = [
  "hold",
  "sprocket",
  "ten seconds",
  "lorem",
  "ticket",
  "widget",
  "portal",
  "colour",
  "thirty days",
  "ipsum",
  "billing",
  "gizmo",
  "prorated",
  "flange",
];

export function createKeywordAdapter(): GepaAdapter<
  KeywordExample,
  KeywordTrajectory,
  string
> {
  return {
    evaluate: ({ batch, candidate, onRollout }) => {
      const answer = Object.values(candidate).join(" ");
      const trajectories: KeywordTrajectory[] = [];
      const scores: number[] = [];
      const feedback: string[] = [];

      for (const example of batch) {
        const missing = example.required.filter(
          (term) => !answer.toLowerCase().includes(term.toLowerCase()),
        );
        const score =
          (example.required.length - missing.length) / example.required.length;

        scores.push(score);
        feedback.push(
          missing.length === 0
            ? "All required terms present."
            : `Missing required terms: ${missing.join(", ")}`,
        );
        trajectories.push({
          question: example.question,
          instruction: answer,
          answer,
          missing,
        });
        onRollout?.();
      }

      return {
        outputs: trajectories.map((trajectory) => trajectory.answer),
        scores,
        feedback,
        trajectories,
      };
    },

    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) => {
      const records = batch.map((example, index) => ({
        inputs: { question: example.question },
        generatedOutputs: evaluation.outputs[index] ?? "",
        feedback: evaluation.feedback?.[index] ?? "",
        score: evaluation.scores[index],
      }));

      return Object.fromEntries(
        componentsToUpdate.map((component) => [component, records]),
      );
    },
  };
}

/**
 * A deterministic stand-in for a reflection model: it reads the feedback in the
 * prompt and folds the missing terms into the current instruction.
 */
export function createKeywordReflector(): TextModel {
  return async ({ prompt }) => {
    const current = extractCurrentInstruction(prompt);
    const missing = extractMissingTerms(prompt);
    const additions = missing.filter(
      (term) => !current.toLowerCase().includes(term.toLowerCase()),
    );

    return `\`\`\`\n${[current, ...additions].filter(Boolean).join(" ")}\n\`\`\``;
  };
}

/**
 * A stand-in for a model asked to rewrite text it has been told nothing about.
 * It appends one term per call, cycling a pool in which only every other entry
 * is useful — which is what blind proposal actually is: a draw from a space
 * where some samples happen to help.
 *
 * Feedback-driven reflectors read the prompt; this one deliberately does not,
 * so a search using it cannot benefit from evidence even if it is offered.
 */
export function createSamplingReflector(
  args: { pool?: readonly string[] } = {},
): TextModel {
  const { pool = SAMPLING_POOL } = args;
  let cursor = 0;

  return async ({ prompt }) => {
    const term = pool[cursor % pool.length] as string;
    cursor += 1;
    return `\`\`\`\n${[extractCurrentInstruction(prompt), term].filter(Boolean).join(" ")}\n\`\`\``;
  };
}

/**
 * A stand-in for a model that reads a score history and climbs it: it takes
 * the highest-scoring attempt it is shown, keeps it, and extends it by one
 * term. Unlike `createSamplingReflector` it depends on the prompt carrying
 * scores, so a search that shows it none makes no progress with it.
 */
export function createHillClimbingReflector(
  args: { pool?: readonly string[] } = {},
): TextModel {
  const { pool = SAMPLING_POOL } = args;
  let cursor = 0;

  return async ({ prompt }) => {
    const best = extractBestAttempt(prompt);
    const term = pool[cursor % pool.length] as string;
    cursor += 1;
    return `\`\`\`\n${[best, term].filter(Boolean).join(" ")}\n\`\`\``;
  };
}

/** A reflection model that always proposes something strictly worse. */
export function createDegradingReflector(): TextModel {
  return async () => "```\nno useful information\n```";
}

export { buildReflectionPrompt };

function extractCurrentInstruction(prompt: string): string {
  const match = prompt.match(
    /<current_instruction>\n([\s\S]*?)\n<\/current_instruction>/,
  );
  return match?.[1]?.trim() ?? "";
}

/** The instruction beside the highest score in a scored-attempt prompt. */
function extractBestAttempt(prompt: string): string {
  let bestScore = Number.NEGATIVE_INFINITY;
  let best = "";

  for (const match of prompt.matchAll(
    /score:\s*([\d.]+)[\s\S]*?<instruction>\n?([\s\S]*?)\n?<\/instruction>/g,
  )) {
    const score = Number(match[1]);
    if (score > bestScore) {
      bestScore = score;
      best = (match[2] ?? "").trim();
    }
  }
  return best;
}

function extractMissingTerms(prompt: string): string[] {
  const terms = new Set<string>();
  for (const match of prompt.matchAll(/Missing required terms: ([^"\\\n]+)/g)) {
    for (const term of (match[1] ?? "").split(",")) {
      const trimmed = term.trim();
      if (trimmed.length > 0) {
        terms.add(trimmed);
      }
    }
  }
  return [...terms];
}
