import { harvestRollouts } from "./harvest.js";
import type { Rng } from "./rng.js";
import type { Adapter, Candidate, UsageTotals } from "./types.js";

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
  /** Tokens and dollars this cost, for a caller that bounds spend. */
  usage: UsageTotals;
  attempted: number;
  /** Rollouts nothing classified, for a caller that reports what it missed. */
  unclassifiedFailures: number;
}

const DEMO_OPEN = "<demo>";
const DEMO_CLOSE = "</demo>";
const DEMO_BLOCK = /<demo>\s*([\s\S]*?)\s*<\/demo>/g;
const DEMO_PARTS =
  /<input>\s*([\s\S]*?)\s*<\/input>\s*<output>\s*([\s\S]*?)\s*<\/output>/;
const DELIMITER_TAG = /<(\/?)(demo|input|output)>/g;
const ESCAPED_DELIMITER_TAG = /&lt;(\/?)(demo|input|output)>/g;
const DEFAULT_MAX_DEMOS = 4;

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
export async function harvestFewShotExamples<
  Datum,
  Trajectory,
  Output,
  K extends string = string,
>(args: {
  adapter: Adapter<Datum, Trajectory, Output, K>;
  /** The candidate to run. Usually the seed, sometimes a run's winner. */
  candidate: Candidate<K>;
  trainingSet: readonly Datum[];
  /**
   * Score a rollout must reach to be kept. Unset keeps every rollout the
   * metric rewarded at all, which is what MIPROv2's bootstrapper does without
   * a `metric_threshold`: it keeps a trace on any truthy score and only
   * compares against a number once one is configured.
   *
   * Demanding a perfect score instead is the right call for a boolean metric
   * and the wrong one for a graded metric, where it throws away every rollout
   * that was most of the way there — which on a hard task is all of them.
   */
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
  /**
   * Ceiling on dollars this pass may spend, checked between batches. Harvesting
   * runs on its own evaluator, so a caller bounding spend has to say so here.
   */
  maxCostUsd?: number;
  /** Shuffles the trainingSet first, so demos are not all drawn from its head. */
  rng?: Rng;
  renderDemo?: DemoRenderer<Datum, Output>;
  signal?: AbortSignal;
}): Promise<BootstrapResult<Datum, Output>> {
  const {
    adapter,
    candidate,
    trainingSet,
    minScore,
    maxDemos = DEFAULT_MAX_DEMOS,
    batchSize = maxDemos,
    maxMetricCalls = trainingSet.length,
    maxCostUsd,
    rng,
    renderDemo,
    signal,
  } = args;

  if (trainingSet.length === 0) {
    throw new Error("harvestFewShotExamples requires a non-empty trainingSet");
  }

  const harvest = await harvestRollouts<Datum, Trajectory, Output, K>({
    adapter,
    candidate,
    data: trainingSet,
    maxRollouts: maxDemos,
    batchSize,
    maxMetricCalls,
    ...(maxCostUsd === undefined ? {} : { maxCostUsd }),
    ...(minScore === undefined ? {} : { minScore }),
    ...(rng === undefined ? {} : { rng }),
    ...(signal === undefined ? {} : { signal }),
  });

  return {
    demos: harvest.rollouts,
    block: formatDemos(
      harvest.rollouts,
      renderDemo === undefined ? {} : { render: renderDemo },
    ),
    metricCalls: harvest.metricCalls,
    usage: harvest.usage,
    attempted: harvest.attempted,
    unclassifiedFailures: harvest.unclassifiedFailures,
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
 * Rewrite the demos a component holds, leaving everything else it says intact.
 *
 * A component is not always only examples. SIMBA appends advice to the same
 * text it appends demonstrations to, so replacing the component wholesale with
 * a fresh block would delete the instructions the other mutation wrote. The
 * replacement lands where the first demo was, so a block a caller placed after
 * its preamble stays after it.
 */
export function replaceDemos<Datum, Output>(args: {
  text: string;
  demos: readonly Demo<Datum, Output>[];
  render?: DemoRenderer<Datum, Output>;
}): string {
  const { text, demos, render } = args;

  const block = formatDemos(demos, render === undefined ? {} : { render });
  const surrounding = text.replace(DEMO_BLOCK, "\u0000");
  const [before, ...rest] = surrounding.split("\u0000");

  if (rest.length === 0) {
    return join([text, block]);
  }
  return join([before ?? "", block, rest.join("")]);
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

/** Joins what survived a replacement, without leaving blank runs behind. */
function join(parts: readonly string[]): string {
  return parts
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .join("\n\n");
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
    return escapeDelimiters(value);
  }
  try {
    return escapeDelimiters(JSON.stringify(value, null, 2) ?? String(value));
  } catch {
    return escapeDelimiters(String(value));
  }
}

function parseValue(text: string): unknown {
  const unescaped = unescapeDelimiters(text);
  try {
    return JSON.parse(unescaped);
  } catch {
    return unescaped;
  }
}

/**
 * Neutralize the tags a demo block is delimited by, so a value carrying one
 * cannot end the block it sits in.
 *
 * A system that quotes its own prompt back produces exactly that: the output
 * worth keeping is a demonstration containing `</demo>`, and appending it raw
 * closes the outer block early. What comes back is not the demo that went in,
 * and SIMBA reparses and rewrites the block at every step, so the damage
 * compounds over a run rather than showing up once.
 *
 * The escape is escaped first, so a value that already reads `&lt;demo>`
 * survives the round trip as itself.
 */
function escapeDelimiters(text: string): string {
  return text
    .replaceAll("&lt;", "&amp;lt;")
    .replace(DELIMITER_TAG, "&lt;$1$2>");
}

function unescapeDelimiters(text: string): string {
  return text
    .replace(ESCAPED_DELIMITER_TAG, "<$1$2>")
    .replaceAll("&amp;lt;", "&lt;");
}
