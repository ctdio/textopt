import type { GepaEvent, GepaResult } from "textopt/gepa";

/**
 * A live view of the search. `metricCalls` is the currency the optimizer
 * spends — every line that reports it is a line reporting cost.
 */
export function logEvent(event: GepaEvent): void {
  switch (event.type) {
    case "start":
      console.log(
        `optimizing [${event.components.join(", ")}] over ${event.valsetSize} val instances`,
      );
      break;
    case "evaluation":
      console.log(
        `  ${event.phase.padEnd(10)} mean=${event.meanScore.toFixed(3)}` +
          `  calls=${event.metricCalls} cached=${event.cacheHits}`,
      );
      break;
    case "proposal":
      console.log(
        `  proposed [${event.componentsToUpdate.join(", ")}] changed=${event.changed}`,
      );
      break;
    case "candidateAccepted":
      console.log(
        `  ✓ accepted #${event.candidateId} (${event.source}, parents ${event.parentIds.join("+")})` +
          ` score=${event.aggregateScore.toFixed(3)}`,
      );
      break;
    case "candidateRejected":
      console.log(
        `  ✗ ${event.reason === "worse" ? "rejected" : "passed over"}` +
          ` ${event.source} of #${event.parentId}:` +
          ` ${event.childScore.toFixed(3)} vs ${event.parentScore.toFixed(3)}`,
      );
      break;
    case "error":
      console.log(`  ! iteration ${event.iteration} failed:`, event.err);
      break;
    case "finish":
      console.log(
        `finished: ${event.reason}, best #${event.bestCandidateId}, ${event.metricCalls} metric calls`,
      );
      break;
    case "iterationStart":
      console.log(
        `\niteration ${event.iteration}` +
          ` (parents ${event.parentIds.map((id) => `#${id}`).join(", ")})`,
      );
      break;
  }
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
