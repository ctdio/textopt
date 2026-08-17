import { buildReflectionPrompt } from "./reflection.js";
import type { Adapter, Reflector } from "./types.js";

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

export function createKeywordAdapter(): Adapter<
  KeywordExample,
  KeywordTrajectory,
  string
> {
  return {
    evaluate: ({ batch, candidate }) => {
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
export function createKeywordReflector(): Reflector {
  return async ({ prompt }) => {
    const current = extractCurrentInstruction(prompt);
    const missing = extractMissingTerms(prompt);
    const additions = missing.filter(
      (term) => !current.toLowerCase().includes(term.toLowerCase()),
    );

    return `\`\`\`\n${[current, ...additions].filter(Boolean).join(" ")}\n\`\`\``;
  };
}

/** A reflection model that always proposes something strictly worse. */
export function createDegradingReflector(): Reflector {
  return async () => "```\nno useful information\n```";
}

export { buildReflectionPrompt };

function extractCurrentInstruction(prompt: string): string {
  const match = prompt.match(
    /<current_instruction>\n([\s\S]*?)\n<\/current_instruction>/,
  );
  return match?.[1]?.trim() ?? "";
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
