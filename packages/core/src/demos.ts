import { createBudget } from "./budget.js";
import { createEvaluator } from "./evaluation.js";
import type { Rng } from "./rng.js";
import type { Adapter, Candidate } from "./types.js";

/**
 * One worked example: what went in, what a run of the system produced, and
 * how that output scored. A demo is harvested, never written — its value is
 * that the output is one the system actually produced and the metric actually
 * rewarded.
 */
export interface Demo<Datum = unknown, Output = unknown> {
  input: Datum;
  output: Output;
  /** Present on harvested demos, absent on ones recovered from a block. */
  score?: number;
}

export type DemoRenderer<Datum = unknown, Output = unknown> = (args: {
  demo: Demo<Datum, Output>;
  index: number;
}) => string;

export interface BootstrapResult<Datum, Output> {
  demos: Demo<Datum, Output>[];
  /** The demos as a candidate component, ready to seed a run with. */
  block: string;
  /** Rollouts this cost. Bootstrapping is cheap, not free. */
  metricCalls: number;
  attempted: number;
}

const DEMO_OPEN = "<demo>";
const DEMO_CLOSE = "</demo>";
const DEMO_BLOCK = /<demo>\s*([\s\S]*?)\s*<\/demo>/g;
const DEMO_PARTS =
  /<input>\s*([\s\S]*?)\s*<\/input>\s*<output>\s*([\s\S]*?)\s*<\/output>/;
const DEFAULT_MAX_DEMOS = 4;
const DEFAULT_MIN_SCORE = 1;

/**
 * Harvest demonstrations by running a candidate over the training set and keeping
 * the rollouts the metric rewarded.
 *
 * The cheapest signal in the whole library: a rollout that scored well is
 * already paid for, and turning it into a few-shot block costs one pass over
 * the data rather than a search. Instruction search and demonstrations pull on
 * different parts of a model's behaviour — instructions on what to do,
 * examples on what the output should look like — so a seed carrying both
 * starts somewhere neither reaches alone.
 */
export async function bootstrapDemos<
  Datum,
  Trajectory,
  Output,
  K extends string = string,
>(args: {
  adapter: Adapter<Datum, Trajectory, Output, K>;
  /** The candidate to run. Usually the seed, sometimes a run's winner. */
  candidate: Candidate<K>;
  trainingSet: readonly Datum[];
  /** Score a rollout must reach to be kept. Default 1. */
  minScore?: number;
  /** Demos to collect before stopping. Default 4. */
  maxDemos?: number;
  /**
   * Instances per rollout batch. Smaller batches stop closer to the moment
   * enough demos exist, at the cost of less concurrency inside the adapter.
   */
  batchSize?: number;
  /** Ceiling on rollouts. Defaults to one pass over the trainingSet. */
  maxMetricCalls?: number;
  /** Shuffles the trainingSet first, so demos are not all drawn from its head. */
  rng?: Rng;
  renderDemo?: DemoRenderer<Datum, Output>;
  signal?: AbortSignal;
}): Promise<BootstrapResult<Datum, Output>> {
  const {
    adapter,
    candidate,
    trainingSet,
    minScore = DEFAULT_MIN_SCORE,
    maxDemos = DEFAULT_MAX_DEMOS,
    batchSize = maxDemos,
    maxMetricCalls = trainingSet.length,
    rng,
    renderDemo,
    signal,
  } = args;

  if (trainingSet.length === 0) {
    throw new Error("bootstrapDemos requires a non-empty trainingSet");
  }

  const budget = createBudget({ maxMetricCalls });
  // Uncached on purpose: the cache stores scores, and a demo needs the output
  // the rollout produced, which a cache hit cannot return.
  const evaluator = createEvaluator<Datum, Trajectory, Output, K>({
    adapter,
    budget,
    ...(signal === undefined ? {} : { signal }),
  });

  const order = rng === undefined ? [...trainingSet] : rng.shuffle(trainingSet);
  const demos: Demo<Datum, Output>[] = [];
  let attempted = 0;

  for (let start = 0; start < order.length; start += batchSize) {
    if (demos.length >= maxDemos || signal?.aborted) {
      break;
    }

    const batch = order.slice(
      start,
      start + Math.min(batchSize, budget.remaining()),
    );
    if (batch.length === 0) {
      break;
    }

    const evaluation = await evaluator.evaluateTraced({
      candidate,
      batch,
      split: "train",
      phase: "seed",
      candidateId: null,
      iteration: 0,
    });
    if (evaluation === null) {
      break;
    }
    attempted += batch.length;

    for (let index = 0; index < batch.length; index += 1) {
      const score = evaluation.scores[index] as number;
      if (score < minScore || demos.length >= maxDemos) {
        continue;
      }
      demos.push({
        input: batch[index] as Datum,
        output: evaluation.outputs[index] as Output,
        score,
      });
    }
  }

  return {
    demos,
    block: formatDemos(
      demos,
      renderDemo === undefined ? {} : { render: renderDemo },
    ),
    metricCalls: budget.spent(),
    attempted,
  };
}

/**
 * Render demos as the text a candidate component holds.
 *
 * Delimited rather than free-form so `parseDemos` can read them back: a demo
 * component is edited over the course of a run, and a block that cannot be
 * parsed can only be replaced wholesale, throwing away every example found
 * before it.
 */
export function formatDemos<Datum, Output>(
  demos: readonly Demo<Datum, Output>[],
  options: { render?: DemoRenderer<Datum, Output> } = {},
): string {
  const { render = renderDefault } = options;

  if (demos.length === 0) {
    return "";
  }

  return demos
    .map(
      (demo, index) =>
        `${DEMO_OPEN}\n${render({ demo, index })}\n${DEMO_CLOSE}`,
    )
    .join("\n");
}

/**
 * Recover the demos from a formatted block, ignoring anything written around
 * them. Text a model rewrote and mangled yields the demos it left intact
 * rather than throwing: a malformed example is worth less than the rest of the
 * block, not more than it.
 */
export function parseDemos(text: string): Demo[] {
  const demos: Demo[] = [];

  for (const match of text.matchAll(DEMO_BLOCK)) {
    const parts = (match[1] ?? "").match(DEMO_PARTS);
    if (parts === null) {
      continue;
    }
    const input = parseValue(parts[1] ?? "");
    const output = parseValue(parts[2] ?? "");
    demos.push({ input, output });
  }
  return demos;
}

function renderDefault<Datum, Output>(args: {
  demo: Demo<Datum, Output>;
  index: number;
}): string {
  const { demo } = args;

  return [
    "<input>",
    serialize(demo.input),
    "</input>",
    "<output>",
    serialize(demo.output),
    "</output>",
  ].join("\n");
}

/** Strings stay as they are; anything else is shown as JSON. */
function serialize(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

function parseValue(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
