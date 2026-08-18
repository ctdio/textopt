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
 *   pnpm --filter @ctdio/textopt-examples pareto
 */
import { GepaOptimizer } from "@ctdio/textopt/gepa";
import type {
  ComponentPatch,
  GepaAdapter,
  ReflectiveRecord,
} from "@ctdio/textopt/gepa";

interface Ticket {
  question: string;
  tone: "terse" | "formal";
  facts: string[];
}

interface TicketTrace {
  toneOk: boolean;
  missingFacts: string[];
}

/** The two text components this system is made of. */
type TicketComponent = "tone" | "facts";

/**
 * What `makeReflectiveDataset` hands the proposer beyond the feedback string.
 * It travels in `ReflectiveRecord.evidence`, the one adapter-owned slot.
 */
interface TicketEvidence {
  expectedTone: Ticket["tone"];
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

const adapter: GepaAdapter<Ticket, TicketTrace, string, TicketComponent> = {
  // Naming the components in the adapter's type is what makes `candidate.tone`
  // a checked `string` rather than an index lookup that silently yields
  // `undefined` for a typo.
  evaluate: ({ batch, candidate }) => {
    const tone = candidate.tone.trim().toLowerCase();
    const facts = candidate.facts.toLowerCase();

    const trajectories = batch.map((ticket) => ({
      toneOk: tone === ticket.tone,
      missingFacts: ticket.facts.filter((fact) => !facts.includes(fact)),
    }));

    return {
      outputs: batch.map(() => `[${tone}] ${candidate.facts}`),
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
    const records: ReflectiveRecord<TicketEvidence>[] = batch.map(
      (ticket, index) => ({
        inputs: ticket.question,
        generatedOutputs: evaluation.outputs[index],
        feedback: evaluation.feedback?.[index] ?? "",
        score: evaluation.scores[index],
        evidence: {
          expectedTone: ticket.tone,
          missingFacts: evaluation.trajectories?.[index]?.missingFacts ?? [],
        },
      }),
    );

    return Object.fromEntries(
      componentsToUpdate.map((component) => [component, records]),
    );
  },

  // Adapters may replace the reflection call entirely. Here it is a rule, which
  // keeps the example deterministic and free.
  proposeNewTexts: ({ candidate, reflectiveDataset, componentsToUpdate }) => {
    const patch: ComponentPatch<TicketComponent> = {};

    for (const component of componentsToUpdate) {
      const records = reflectiveDataset[component] ?? [];

      if (component === "tone") {
        const wanted = records
          .filter((record) => (record.score ?? 1) < 1)
          .flatMap((record) => readEvidence(record)?.expectedTone ?? [])[0];
        if (wanted !== undefined && wanted !== candidate.tone) {
          patch.tone = wanted;
        }
      }

      if (component === "facts") {
        const missing = [
          ...new Set(
            records.flatMap(
              (record) => readEvidence(record)?.missingFacts ?? [],
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

const gepa = new GepaOptimizer({
  maxIterations: 14,
  // One example per minibatch, so each lineage specializes on one ticket —
  // exactly the condition that produces a branching frontier.
  minibatchSize: 1,
  seed: 5,
  merge: { enabled: false },
});

const result = await gepa.optimize({
  seedCandidate: { tone: "neutral", facts: "" },
  trainset: TICKETS,
  adapter,
  reflect: async () => "",
  maxMetricCalls: 400,
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

// Selection pressure, derived here from the two public views of the run: the
// score matrix says who wins each instance, and `paretoFrontier` is the set the
// engine kept — everything else won nothing another survivor did not also win.
const fronts = instanceWinners(result.scoreMatrix);
const survivors = new Set(result.paretoFrontier.map((record) => record.id));

console.log("\nper-instance winners (ties keep every winner):");
fronts.forEach((front, index) => {
  console.log(`  i${index}  ${front.map((id) => `#${id}`).join(", ")}`);
});

const wins = new Map<number, number>();
for (const front of fronts) {
  for (const id of front.filter((candidate) => survivors.has(candidate))) {
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

/** For each validation instance, every candidate tied for the best score on it. */
function instanceWinners(
  scoreMatrix: readonly (readonly (number | undefined)[])[],
): number[][] {
  const instanceCount = scoreMatrix[0]?.length ?? 0;

  return Array.from({ length: instanceCount }, (_unused, instance) => {
    const column = scoreMatrix.map(
      (row) => row[instance] ?? Number.NEGATIVE_INFINITY,
    );
    const best = Math.max(...column);

    return column.flatMap((score, id) => (score === best ? [id] : []));
  });
}

/**
 * `ReflectiveDataset` is not parameterized by the evidence type, so evidence
 * this adapter wrote as `TicketEvidence` arrives back as `unknown` and has to
 * be re-narrowed before the rule above can read it.
 */
function readEvidence(record: ReflectiveRecord): TicketEvidence | undefined {
  const { evidence } = record;

  if (
    typeof evidence !== "object" ||
    evidence === null ||
    !("expectedTone" in evidence) ||
    !("missingFacts" in evidence)
  ) {
    return undefined;
  }

  const { expectedTone, missingFacts } = evidence;

  if (
    (expectedTone !== "terse" && expectedTone !== "formal") ||
    !Array.isArray(missingFacts)
  ) {
    return undefined;
  }

  return { expectedTone, missingFacts: missingFacts.map(String) };
}
