/**
 * What the Pareto frontier actually is, printed.
 *
 * GEPA's frontier is not over objectives — it is over *validation instances*.
 * The whole selection state is one matrix: `scoreMatrix[candidate][instance]`.
 * A candidate stays alive if it is best on at least one instance, even when its
 * mean is mediocre, and parents are sampled in proportion to how many instances
 * they win. That is what stops the search collapsing onto one local optimum.
 *
 * To make that visible you need a task with a real trade-off, so this one has
 * an unavoidable conflict: a single `tone` component, and tickets that want
 * different tones. No candidate can win everything.
 *
 * Runs offline — the adapter implements `proposeNewTexts`, replacing the
 * reflection LLM with a deterministic rule.
 *
 *   pnpm --filter @ctdio/gepa-examples pareto
 */
import {
  buildInstanceFronts,
  optimize,
  pruneDominatedFronts,
} from "@ctdio/gepa";
import type { Adapter, ReflectiveRecord } from "@ctdio/gepa";

interface Ticket {
  question: string;
  tone: "terse" | "formal";
  facts: string[];
}

interface TicketTrace {
  toneOk: boolean;
  missingFacts: string[];
}

const TICKETS: Ticket[] = [
  {
    question: "Reset steps?",
    tone: "terse",
    facts: ["hold ten seconds"],
  },
  {
    question: "Where do I file a ticket?",
    tone: "terse",
    facts: ["support portal"],
  },
  {
    question: "Please explain your refund policy.",
    tone: "formal",
    facts: ["thirty days"],
  },
  {
    question: "Could you clarify plan upgrade billing?",
    tone: "formal",
    facts: ["prorated"],
  },
];

const adapter: Adapter<Ticket, TicketTrace, string> = {
  evaluate: ({ batch, candidate }) => {
    const tone = (candidate.tone ?? "").trim().toLowerCase();
    const facts = (candidate.facts ?? "").toLowerCase();

    const trajectories = batch.map((ticket) => ({
      toneOk: tone === ticket.tone,
      missingFacts: ticket.facts.filter((fact) => !facts.includes(fact)),
    }));

    return {
      outputs: batch.map(() => `[${tone}] ${candidate.facts ?? ""}`),
      // Half the score is a trade-off the candidate cannot win everywhere,
      // half is knowledge it can always accumulate.
      scores: trajectories.map(
        (trace, index) =>
          0.5 * (trace.toneOk ? 1 : 0) +
          0.5 *
            (1 - trace.missingFacts.length / (batch[index]?.facts.length ?? 1)),
      ),
      feedback: trajectories.map((trace, index) =>
        [
          trace.toneOk
            ? "Tone matched."
            : `Wrong tone: this ticket wants "${batch[index]?.tone}".`,
          trace.missingFacts.length === 0
            ? "All facts present."
            : `Missing facts: ${trace.missingFacts.join(", ")}.`,
        ].join(" "),
      ),
      trajectories,
    };
  },

  makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) => {
    const records: ReflectiveRecord[] = batch.map((ticket, index) => ({
      inputs: ticket.question,
      generatedOutputs: evaluation.outputs[index],
      feedback: evaluation.feedback?.[index] ?? "",
      score: evaluation.scores[index],
      expectedTone: ticket.tone,
      missingFacts: evaluation.trajectories?.[index]?.missingFacts ?? [],
    }));

    return Object.fromEntries(
      componentsToUpdate.map((component) => [component, records]),
    );
  },

  // Adapters may replace the reflection call entirely. Here it is a rule, which
  // keeps the example deterministic and free.
  proposeNewTexts: ({ candidate, reflectiveDataset, componentsToUpdate }) => {
    const patch: Record<string, string> = {};

    for (const component of componentsToUpdate) {
      const records = reflectiveDataset[component] ?? [];

      if (component === "tone") {
        const wanted = records
          .filter((record) => (record.score ?? 1) < 1)
          .map((record) => String(record.expectedTone))[0];
        if (wanted !== undefined && wanted !== candidate.tone) {
          patch.tone = wanted;
        }
      }

      if (component === "facts") {
        const missing = [
          ...new Set(
            records.flatMap((record) =>
              Array.isArray(record.missingFacts)
                ? record.missingFacts.map(String)
                : [],
            ),
          ),
        ];
        if (missing.length > 0) {
          patch.facts = [candidate.facts, ...missing]
            .filter(Boolean)
            .join("; ");
        }
      }
    }

    return patch;
  },
};

const result = await optimize({
  seedCandidate: { tone: "neutral", facts: "" },
  trainset: TICKETS,
  adapter,
  reflect: async () => "",
  maxMetricCalls: 400,
  maxIterations: 14,
  // One example per minibatch, so each lineage specializes on one ticket —
  // exactly the condition that produces a branching frontier.
  minibatchSize: 1,
  seed: 5,
  merge: { enabled: false },
  instanceId: ({ datum }) => datum.question,
});

const aggregateScores = result.candidates.map(
  (candidate) => candidate.aggregateScore,
);

console.log("score matrix (rows = candidates, columns = val instances)\n");
console.log(
  [
    "cand".padEnd(6),
    ...TICKETS.map((_, index) => `i${index}`.padStart(6)),
    "  mean",
  ].join(""),
);
for (const candidate of result.candidates) {
  console.log(
    [
      `#${candidate.id}`.padEnd(6),
      ...candidate.instanceScores.map((score) =>
        (score === undefined ? "—" : score.toFixed(2)).padStart(6),
      ),
      `  ${candidate.aggregateScore.toFixed(3)}`,
    ].join(""),
  );
}

console.log("\ninstances:");
TICKETS.forEach((ticket, index) => {
  console.log(`  i${index}  [${ticket.tone}] ${ticket.question}`);
});

// The same two functions the engine calls internally, run here on the finished
// result so you can watch selection pressure being derived.
const fronts = buildInstanceFronts({ scoreMatrix: result.scoreMatrix });
const pruned = pruneDominatedFronts({ fronts, aggregateScores });

console.log("\nper-instance winners (ties keep every winner):");
fronts.forEach((front, index) => {
  console.log(`  i${index}  ${[...front].map((id) => `#${id}`).join(", ")}`);
});

const wins = new Map<number, number>();
for (const front of pruned) {
  for (const id of front) {
    wins.set(id, (wins.get(id) ?? 0) + 1);
  }
}
const totalWins = [...wins.values()].reduce((sum, count) => sum + count, 0);

console.log(
  "\nafter dropping candidates whose wins are fully covered by another,",
  "parents are sampled in proportion to instances won:",
);
for (const [id, count] of [...wins].sort((a, b) => b[1] - a[1])) {
  console.log(
    `  #${id}  ${count}/${totalWins} = ${((count / totalWins) * 100).toFixed(0)}%` +
      `   mean ${aggregateScores[id]?.toFixed(3)}   ${JSON.stringify(
        result.candidates[id]?.candidate,
      )}`,
  );
}

console.log(
  `\nbest by mean is #${result.bestCandidateId} at ${result.bestScore.toFixed(3)} —`,
  "but note it is not the only candidate kept alive.",
);
