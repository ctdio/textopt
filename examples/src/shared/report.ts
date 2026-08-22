import { isCandidateAccepted } from "textopt";
import type { OptimizerEvent, Reporter } from "textopt";
import type { GepaResult } from "textopt/gepa";

export interface AcceptanceRow {
  entrant: string;
  /** Candidates that displaced the incumbent, the seed excluded. */
  accepted: number;
  seedScore: number;
  bestScore: number;
}

/**
 * One tally, shared by every entrant in a comparison.
 *
 * `for` hands out a reporter per entrant, and each is a `Reporter<
 * OptimizerEvent>` rather than a reporter for one search: it narrows with
 * `isCandidateAccepted` and reads only the acceptance payload every optimizer
 * emits — the candidate text, its aggregate, and its row over the validation
 * set. That is what lets the same observer attach to GEPA, SIMBA and OPRO and
 * produce numbers that mean the same thing across all three.
 *
 * Candidate 0 is the seed everywhere, which is what makes the lift readable:
 * without a baseline in the same units, "best 0.81" says nothing about what
 * the run bought.
 */
export function createAcceptanceTally(): {
  for: (entrant: string) => Reporter<OptimizerEvent>;
  rows: () => AcceptanceRow[];
} {
  const tallies = new Map<string, AcceptanceRow>();
  const seedRuns = new Map<string, number>();

  return {
    for: (entrant) => ({
      onEvent: (event) => {
        if (!isCandidateAccepted(event)) {
          return;
        }

        const row = tallies.get(entrant) ?? {
          entrant,
          accepted: 0,
          seedScore: 0,
          bestScore: 0,
        };

        // An entrant runs once per seed, and a noisy metric scores the same
        // seed candidate differently every time. The baseline is the running
        // mean of those readings, not whichever run reported last.
        if (event.candidateId === 0) {
          const runs = (seedRuns.get(entrant) ?? 0) + 1;
          seedRuns.set(entrant, runs);
          row.seedScore += (event.aggregateScore - row.seedScore) / runs;
        } else {
          row.accepted += 1;
        }
        row.bestScore = Math.max(row.bestScore, event.aggregateScore);

        tallies.set(entrant, row);
      },
    }),
    rows: () => [...tallies.values()],
  };
}

export function printResult(result: GepaResult): void {
  const seedScore = result.candidates[0]?.aggregateScore ?? 0;

  console.log(
    `\nseed ${seedScore.toFixed(3)} -> best ${result.bestScore.toFixed(3)}` +
      ` over ${result.candidates.length} candidates` +
      ` (${result.metricCalls} metric calls, ${result.cacheHits} cache hits)`,
  );
  console.log(
    `frontier: ${result.paretoFrontier.map((record) => `#${record.id}`).join(", ")}`,
  );

  for (const [component, text] of Object.entries(result.bestCandidate)) {
    console.log(`\n--- ${component} ---\n${text}`);
  }
}
