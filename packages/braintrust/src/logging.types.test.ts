import type { TextModel } from "textopt";
import type { GepaAdapter } from "textopt/gepa";
import { GepaOptimizer } from "textopt/gepa";
import { describe, expect, test } from "vitest";
import type { BraintrustEvent, BraintrustLoggerLike } from "./logging.js";
import { withBraintrustLogging } from "./logging.js";

interface Ticket {
  id: string;
  text: string;
}

type TicketComponent = "systemPrompt" | "rubric";

const SEED = {
  systemPrompt: "Answer the ticket.",
  rubric: "Cite the policy.",
};

const TICKETS: Ticket[] = [
  { id: "a", text: "printer on fire" },
  { id: "b", text: "refund please" },
];

const REFLECT: TextModel = async () => "```\nCite the policy verbatim.\n```";

/**
 * The `@ts-expect-error` is the regression test: if `withBraintrustLogging`
 * stops threading the adapter's component names, the typo below becomes legal,
 * the directive goes unused and `tsc` reports it.
 */
export function decoratedComponentKeysAreChecked(): unknown {
  const decorated = withBraintrustLogging({
    adapter: createTicketAdapter(),
    logger: { log: () => undefined },
  });

  const asKeyed: GepaAdapter<Ticket, string, string, TicketComponent> =
    decorated;
  void asKeyed;

  return decorated.evaluate({
    batch: [],
    // @ts-expect-error "rubrik" is not a component the adapter declares
    candidate: { systemPrompt: "x", rubrik: "y" },
    captureTraces: false,
    run: {
      iteration: 0,
      phase: "minibatch",
      split: "train",
      candidateId: 0,
    },
  });
}

describe("withBraintrustLogging under a real optimization run", () => {
  test("logs a decorated adapter's rollouts with the engine's run context", async () => {
    const events: BraintrustEvent[] = [];
    const logger: BraintrustLoggerLike = {
      log: (event) => events.push(event),
    };

    await new GepaOptimizer({
      minibatchSize: 1,
      maxIterations: 1,
      seed: 1,
    }).optimize({
      seedCandidate: SEED,
      trainingSet: TICKETS,
      adapter: withBraintrustLogging({
        adapter: createTicketAdapter(),
        logger,
      }),
      reflect: REFLECT,
      maxMetricCalls: 20,
    });

    const phases = new Set(events.map((event) => event.metadata?.phase));

    expect(events.length).toBeGreaterThan(0);
    expect(phases).toContain("validation");
    expect(Object.keys(events[0]?.metadata?.candidate ?? {})).toEqual([
      "systemPrompt",
      "rubric",
    ]);
  });
});

function createTicketAdapter(): GepaAdapter<
  Ticket,
  string,
  string,
  TicketComponent
> {
  return {
    evaluate: ({ batch, candidate }) => {
      const answer = `${candidate.systemPrompt} ${candidate.rubric}`;

      return {
        outputs: batch.map((ticket) => `${answer} ${ticket.text}`),
        scores: batch.map(() => (answer.includes("verbatim") ? 1 : 0)),
        feedback: batch.map(() => "cite the policy verbatim"),
        trajectories: batch.map((ticket) => ticket.id),
      };
    },

    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) => {
      const records = batch.map((ticket, index) => ({
        inputs: { text: ticket.text },
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
