/**
 * No framework at all: a hand-written `GepaAdapter` over a vendor's own SDK.
 * Both are implemented here — flip `VENDOR` below — because the point of the
 * example is that swapping them touches exactly one function.
 *
 * The adapter is the only integration seam textopt has. Implement two
 * methods — `evaluate` (per-instance scores plus textual feedback) and
 * `makeReflectiveDataset` (what the reflection model gets to read) — and any
 * system becomes optimizable: an HTTP call, a retrieval pipeline, a compiler
 * pass, a shell script.
 *
 * Writing both by hand is the case where the trajectory matters. Here it is a
 * parse failure: the reflection model has to see the raw response that would
 * not parse, which no score carries. A system whose evidence is just its input
 * and its output wants `createPromptAdapter`, which writes both methods.
 *
 * The task model here is the cheapest tier, deliberately. The interesting
 * result of an optimization run is usually not "the big model got better" but
 * "the small model caught up once the prompt carried the domain knowledge".
 *
 *   ANTHROPIC_API_KEY=... pnpm --filter textopt-examples custom
 */
import { anthropic } from "@ai-sdk/anthropic";
import Anthropic from "@anthropic-ai/sdk";
import { consoleReporter, mapWithConcurrency } from "textopt";
import { GepaOptimizer } from "textopt/gepa";
import type { GepaAdapter } from "textopt/gepa";
import { openai } from "@ai-sdk/openai";
import OpenAI from "openai";
import { createReflector, requireApiKey } from "./shared/reflector.js";
import { printResult } from "./shared/report.js";

type Vendor = "anthropic" | "openai";

interface InvoiceLine {
  id: string;
  raw: string;
  expected: { vendor: string; amount: string; currency: string };
}

interface ExtractionTrace {
  response: string;
  parsed: Record<string, unknown> | null;
  parseError?: string;
}

/**
 * Every row carries a formatting convention a one-line prompt gets wrong:
 * European decimal commas, trailing currency symbols, legal-suffix noise,
 * thousands separators. The reflection model has to discover these from
 * feedback and write them into the instruction.
 */
const TRAIN: InvoiceLine[] = [
  {
    id: "a1",
    raw: "ACME Widgets Inc. — INV-8891 — 1.234,50 EUR",
    expected: { vendor: "ACME Widgets", amount: "1234.50", currency: "EUR" },
  },
  {
    id: "a2",
    raw: "Northwind Traders LLC / invoice 2231 / $980",
    expected: {
      vendor: "Northwind Traders",
      amount: "980.00",
      currency: "USD",
    },
  },
  {
    id: "a3",
    raw: "Globex Corporation | 12 500 JPY | ref 77-2",
    expected: { vendor: "Globex", amount: "12500.00", currency: "JPY" },
  },
  {
    id: "a4",
    raw: "Initech, Ltd.  £2,050.00  (Q3 retainer)",
    expected: { vendor: "Initech", amount: "2050.00", currency: "GBP" },
  },
  {
    id: "a5",
    raw: "Umbrella Health GmbH 89,00€ Rechnung 4410",
    expected: { vendor: "Umbrella Health", amount: "89.00", currency: "EUR" },
  },
  {
    id: "a6",
    raw: "Stark Industries Co — 45000 — USD — PO 9",
    expected: {
      vendor: "Stark Industries",
      amount: "45000.00",
      currency: "USD",
    },
  },
];

const VAL: InvoiceLine[] = [
  {
    id: "b1",
    raw: "Wayne Enterprises, Inc. :: 3.000,00 EUR :: DE-1187",
    expected: {
      vendor: "Wayne Enterprises",
      amount: "3000.00",
      currency: "EUR",
    },
  },
  {
    id: "b2",
    raw: "Cyberdyne Systems Ltd — 7,499.99 USD",
    expected: {
      vendor: "Cyberdyne Systems",
      amount: "7499.99",
      currency: "USD",
    },
  },
  {
    id: "b3",
    raw: "Tyrell Corp. ¥ 320 000 — annual",
    expected: { vendor: "Tyrell", amount: "320000.00", currency: "JPY" },
  },
  {
    id: "b4",
    raw: "Soylent Foods GmbH 12,90€",
    expected: { vendor: "Soylent Foods", amount: "12.90", currency: "EUR" },
  },
  {
    id: "b5",
    raw: "Hooli LLC | £15,000 | contract renewal",
    expected: { vendor: "Hooli", amount: "15000.00", currency: "GBP" },
  },
];

const FIELDS = ["vendor", "amount", "currency"] as const;

/** Edit this line to run the identical adapter against the other vendor. */
const VENDOR = "anthropic" as Vendor;

requireApiKey(VENDOR === "openai" ? "OPENAI_API_KEY" : "ANTHROPIC_API_KEY");

// Only the selected vendor's client is constructed — both SDKs throw at
// construction time when their key is missing.
const client =
  VENDOR === "openai"
    ? ({ kind: "openai", openai: new OpenAI() } as const)
    : ({ kind: "anthropic", anthropic: new Anthropic() } as const);

const reflect =
  VENDOR === "openai"
    ? createReflector({
        model: openai("gpt-5.6"),
        providerOptions: { openai: { reasoningEffort: "high" } },
      })
    : createReflector({
        model: anthropic("claude-opus-5"),
        providerOptions: { anthropic: { thinking: { type: "adaptive" } } },
      });

const adapter: GepaAdapter<
  InvoiceLine,
  ExtractionTrace,
  string,
  "instruction"
> = {
  evaluate: async ({ batch, candidate, onRollout, signal }) => {
    const results = await mapWithConcurrency({
      items: batch,
      limit: 4,
      // One `rollout` event per settled instance. Without it a run reports
      // nothing between the start of a validation sweep and its end.
      onSettled: onRollout,
      task: async (line) => {
        const response = await complete({
          instruction: candidate.instruction,
          input: line.raw,
          signal,
        });

        const extraction = { line, response, ...parseJson(response) };
        return { ...extraction, scored: scoreFields(extraction) };
      },
    });

    return {
      outputs: results.map((result) => result.response),
      scores: results.map((result) => result.scored.score),
      feedback: results.map((result) => result.scored.feedback),
      trajectories: results.map((result) => ({
        response: result.response,
        parsed: result.parsed,
        ...(result.parseError === undefined
          ? {}
          : { parseError: result.parseError }),
      })),
    };
  },

  makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) => {
    const records = batch.map((line, index) => ({
      inputs: line.raw,
      generatedOutputs: evaluation.outputs[index] ?? "",
      feedback: evaluation.feedback?.[index] ?? "",
      score: evaluation.scores[index],
      // Showing the target next to the output is what lets the reflection model
      // state the convention explicitly instead of guessing at it. Anything the
      // adapter adds beyond the four standard fields goes in `evidence`.
      evidence: { expected: line.expected },
    }));

    return Object.fromEntries(
      componentsToUpdate.map((component) => [component, records]),
    );
  },
};

const gepa = new GepaOptimizer({
  minibatchSize: 3,
  seed: 2,
});

const result = await gepa.optimize({
  seedCandidate: {
    instruction:
      "Extract the vendor, amount and currency from the invoice line.",
  },
  trainingSet: TRAIN,
  validationSet: VAL,
  adapter,
  reflect,
  maxMetricCalls: 120,
  instanceId: ({ datum }) => datum.id,
  reporters: [consoleReporter()],
});

printResult(result);

/**
 * The vendor call, and the only provider-specific code in the file. Everything
 * above it — the adapter, the scoring, the reflective dataset — is unchanged
 * between OpenAI and Anthropic.
 */
async function complete(args: {
  instruction: string;
  input: string;
  signal?: AbortSignal;
}): Promise<string> {
  const { instruction, input, signal } = args;

  if (client.kind === "openai") {
    const response = await client.openai.responses.create(
      {
        model: "gpt-5.6-luna",
        instructions: instruction,
        input,
        max_output_tokens: 1024,
        reasoning: { effort: "none" },
      },
      { signal },
    );

    return response.output_text.trim();
  }

  const message = await client.anthropic.messages.create(
    {
      model: "claude-haiku-4-5",
      max_tokens: 300,
      system: instruction,
      messages: [{ role: "user", content: input }],
    },
    { signal },
  );

  return message.content
    .map((block) => (block.type === "text" ? block.text : ""))
    .join("")
    .trim();
}

function parseJson(response: string): {
  parsed: Record<string, unknown> | null;
  parseError?: string;
} {
  const block = response.match(/\{[\s\S]*\}/);

  if (block === null) {
    return { parsed: null, parseError: "no JSON object in the response" };
  }

  try {
    return { parsed: JSON.parse(block[0]) as Record<string, unknown> };
  } catch (err) {
    return {
      parsed: null,
      parseError: err instanceof Error ? err.message : String(err),
    };
  }
}

function scoreFields(args: {
  line: InvoiceLine;
  response: string;
  parsed: Record<string, unknown> | null;
  parseError?: string;
}): { score: number; feedback: string } {
  const { line, response, parsed, parseError } = args;

  if (parsed === null) {
    return {
      score: 0,
      feedback:
        `Could not read a JSON object from the response (${parseError}). ` +
        `Return only a JSON object with the keys ${FIELDS.join(", ")}. ` +
        `Got: ${response.slice(0, 120)}`,
    };
  }

  const wrong = FIELDS.filter(
    (field) => String(parsed[field] ?? "") !== line.expected[field],
  );

  return {
    score: (FIELDS.length - wrong.length) / FIELDS.length,
    feedback:
      wrong.length === 0
        ? "All fields correct."
        : wrong
            .map(
              (field) =>
                `${field}: got "${String(parsed[field] ?? "")}", expected "${line.expected[field]}".`,
            )
            .join(" "),
  };
}
