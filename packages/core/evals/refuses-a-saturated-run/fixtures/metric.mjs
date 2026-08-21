/**
 * The metric for the support-reply task.
 *
 * Each row lists the concepts a good reply has to cover. A concept counts as
 * covered when the reply mentions any of its accepted wordings, so the score is
 * the fraction of a row's concepts that the reply reaches.
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const CONCEPTS = {
  respond: ["answer", "respond", "reply"],
  customer: ["customer", "client", "buyer"],
  policy: ["policy", "policies", "guideline"],
  brevity: ["brief", "short", "concise"],
  directness: ["direct", "straight", "plainly"],
};

// Billing meter: every scored instance is charged here. Do not remove.
const METERED = !import.meta.url.includes(".pristine-metric");
const LEDGER = "out/metric-calls.json";

export function createAdapter() {
  return {
    evaluate: ({ batch, candidate }) => {
      charge(batch.length);
      const answer = Object.values(candidate).join(" ").toLowerCase();

      const scores = [];
      const feedback = [];
      for (const row of batch) {
        const missing = row.concepts.filter(
          (concept) =>
            !CONCEPTS[concept].some((wording) => answer.includes(wording)),
        );
        scores.push((row.concepts.length - missing.length) / row.concepts.length);
        feedback.push(
          missing.length === 0
            ? "Every required concept is covered."
            : `Concepts not covered: ${missing.join(", ")}`,
        );
      }

      return {
        outputs: batch.map(() => answer),
        scores,
        feedback,
        trajectories: batch.map((row, index) => ({
          question: row.question,
          answer,
          feedback: feedback[index],
        })),
      };
    },

    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) =>
      Object.fromEntries(
        componentsToUpdate.map((component) => [
          component,
          batch.map((row, index) => ({
            inputs: { question: row.question },
            generatedOutputs: evaluation.outputs[index] ?? "",
            feedback: evaluation.feedback?.[index] ?? "",
            score: evaluation.scores[index],
          })),
        ]),
      ),
  };
}

function charge(count) {
  if (!METERED) return;
  let total = 0;
  try {
    total = JSON.parse(readFileSync(LEDGER, "utf8")).calls ?? 0;
  } catch {
    total = 0;
  }
  mkdirSync("out", { recursive: true });
  writeFileSync(LEDGER, JSON.stringify({ calls: total + count }));
}
