import type { Candidate, ScoreResult, TextModel } from "textopt";
import type { GepaAdapter } from "textopt/gepa";

export interface BenchDatum {
  id: number;
  /** Terms an answer must contain to score, per component. */
  required: Record<string, string[]>;
}

export interface BenchTask {
  name: string;
  seedCandidate: Candidate;
  trainingSet: BenchDatum[];
  validationSet: BenchDatum[];
  testSet: BenchDatum[];
  adapter: GepaAdapter<BenchDatum, string, string>;
  /**
   * Components a demonstration search harvests into. Named per task because an
   * optimizer that writes examples rather than instructions still has to put
   * them somewhere, and here that is the same component the text-proposing
   * entrants rewrite — so every entrant is scored on one candidate shape.
   */
  demoComponents: string[];
  /** Rollouts each optimizer is given. The same for all of them, by design. */
  maxMetricCalls: number;
  /** Reflection calls, which no metric budget covers. */
  maxReflectionCalls: number;
}

/** Terms the metric rewards. Wide enough that no single run covers them all. */
const REQUIRED = [
  "hold",
  "ten seconds",
  "ticket",
  "portal",
  "thirty days",
  "billing",
  "prorated",
  "refund",
  "escalate",
  "timestamp",
  "celsius",
  "apology",
  "deadline",
  "receipt",
  "invoice",
  "sla",
  "utc",
  "warranty",
  "signature",
  "voicemail",
  "callback",
  "tracking number",
  "zip code",
  "business days",
];

/** Terms the metric penalizes, so blind accumulation is not free. */
const DISTRACTORS = [
  "sprocket",
  "lorem",
  "widget",
  "colour",
  "ipsum",
  "gizmo",
  "flange",
  "quux",
];

/** What a proposer draws from: neither list is labelled, so it must be told. */
const POOL = REQUIRED.flatMap((term, index) => {
  const distractor = DISTRACTORS[index % DISTRACTORS.length] as string;
  return index % 2 === 0 ? [term, distractor] : [term];
});

/** Score lost per irrelevant term, the cost of instruction bloat. */
const BLOAT_PENALTY = 0.03;

export function benchTasks(): BenchTask[] {
  return [clean(), noisy(), interacting(), demonstrated()];
}

/**
 * A stand-in for the model under optimization's proposer. It reads whichever
 * evidence the prompt carries — written feedback, a score history, or neither
 * — so the differences a run measures are differences between searches rather
 * than between the prompts they happen to send.
 *
 * `absorb` is why feedback is an advantage and not an oracle: a real model
 * acts on some of what it is told, not all of it, so a reflective run still
 * has to iterate.
 */
export function createBenchReflector(
  args: { absorb?: number; draftSize?: number } = {},
): TextModel {
  const { absorb = 2, draftSize = 4 } = args;
  let cursor = 0;

  return async ({ prompt }) => {
    const base = bestAttempt(prompt) ?? currentText(prompt);
    const missing = missingTerms(prompt).filter(
      (term) => !base.toLowerCase().includes(term.toLowerCase()),
    );

    if (missing.length > 0) {
      return fence([base, ...missing.slice(0, absorb)].join(" ").trim());
    }

    const drawn = draw({
      from: cursor,
      count: base.length === 0 ? draftSize : 1,
    });
    cursor += 1;
    return fence([base, ...drawn].join(" ").trim());
  };
}

/**
 * The same simulated proposer in the shape SIMBA asks for: per-component advice
 * rather than a rewritten instruction. It absorbs the same number of terms per
 * call as `createBenchReflector` and draws from the same half-useless pool when
 * there is nothing to fix, so the two searches are given equally capable models
 * and the comparison measures the search rather than the proposer.
 */
export function createBenchAdviser(args: { absorb?: number } = {}): TextModel {
  const { absorb = 2 } = args;
  let cursor = 0;

  return async ({ prompt }) => {
    const components = adviceComponents(prompt);
    const missing = missingTerms(prompt);

    const terms =
      missing.length > 0
        ? missing.slice(0, absorb)
        : draw({ from: cursor, count: 1 });
    cursor += 1;

    return components
      .map(
        (component) =>
          `<advice component="${component}">${terms.join(" ")}</advice>`,
      )
      .join("\n");
  };
}

/**
 * Renders a harvested rollout without its label.
 *
 * A bench datum is an id and the terms an answer needs, so the default JSON
 * renderer would print the answer key into the candidate — and the metric
 * scores the candidate for containing exactly those terms. A demonstration
 * search would then be reading the labels out of its own prompt and scoring
 * for it. Only the id goes in, which is what a real demo's input is: the
 * question, not the mark scheme.
 */
export function renderBenchDemo(args: {
  demo: { input: BenchDatum; output: string };
}): string {
  const { demo } = args;
  return `<input>\n${demo.input.id}\n</input>\n<output>\n${demo.output}\n</output>`;
}

/** A noiseless metric with a clean gradient: the reference case. */
function clean(): BenchTask {
  const data = singleComponentData();

  return {
    name: "clean",
    demoComponents: ["instruction"],
    seedCandidate: { instruction: "" },
    ...data,
    adapter: keywordAdapter({ noise: 0 }),
    maxMetricCalls: 200,
    maxReflectionCalls: 40,
  };
}

/**
 * The same task, measured with instance-level noise. Real metrics are noisy —
 * a judge model, a sampled generation — and noise is what separates an
 * acceptance rule that survives contact with a real system from one tuned on
 * a deterministic fixture.
 */
function noisy(): BenchTask {
  const data = singleComponentData();

  return {
    name: "noisy",
    demoComponents: ["instruction"],
    seedCandidate: { instruction: "" },
    ...data,
    adapter: keywordAdapter({ noise: 0.2 }),
    maxMetricCalls: 200,
    maxReflectionCalls: 40,
  };
}

/**
 * Two components that only pay off together: an instance needs a term from
 * each, so improving either alone leaves the score where it was. This is the
 * case joint search exists for and per-component search cannot see.
 */
function interacting(): BenchTask {
  return {
    name: "interacting",
    demoComponents: ["alpha", "beta"],
    seedCandidate: { alpha: "", beta: "" },
    trainingSet: pairedInstances({ count: 12, stride: 5, from: 0 }),
    validationSet: pairedInstances({ count: 12, stride: 5, from: 0 }),
    testSet: pairedInstances({ count: 12, stride: 7, from: 100 }),
    adapter: keywordAdapter({ noise: 0, allOrNothing: true }),
    maxMetricCalls: 250,
    maxReflectionCalls: 40,
  };
}

/**
 * The same metric again, over a system that is sometimes right on its own.
 *
 * Every task above scores text the candidate already holds, so a rollout's
 * output tells the search nothing its prompt did not, and a demonstration
 * search harvesting those rollouts can only hand a candidate its own words
 * back. Real systems are not like that: a model answers one instance correctly
 * and the next one of the same kind wrong, and the examples worth showing it
 * are the ones it already got right.
 *
 * `reliability` is that property and nothing else. It is what makes harvesting
 * a lever here and a no-op on the three tasks above — which is the comparison
 * worth running, because it is the question a caller actually has: are my
 * failures a matter of the instruction, or of consistency?
 */
function demonstrated(): BenchTask {
  const data = singleComponentData();

  return {
    name: "demonstrated",
    demoComponents: ["instruction"],
    seedCandidate: { instruction: "" },
    ...data,
    adapter: keywordAdapter({ noise: 0, reliability: 0.4 }),
    maxMetricCalls: 200,
    maxReflectionCalls: 40,
  };
}

function singleComponentData(): Pick<
  BenchTask,
  "trainingSet" | "validationSet" | "testSet"
> {
  return {
    trainingSet: instances({ count: 12, stride: 5, from: 0 }),
    validationSet: instances({ count: 12, stride: 5, from: 0 }),
    // Held out: the same vocabulary, different pairings, so a candidate that
    // covered the validation instances by memorizing pairs scores lower here
    // than one that covered the vocabulary.
    testSet: instances({ count: 12, stride: 7, from: 100 }),
  };
}

/** One term per component, scored only when both land. */
function pairedInstances(args: {
  count: number;
  stride: number;
  from: number;
}): BenchDatum[] {
  return instances(args).map((datum) => {
    const [first, second] = datum.required.instruction as string[];
    return {
      id: datum.id,
      required: { alpha: [first ?? ""], beta: [second ?? first ?? ""] },
    };
  });
}

/** Instances requiring two terms each, spread over the vocabulary. */
function instances(args: {
  count: number;
  stride: number;
  from: number;
}): BenchDatum[] {
  const { count, stride, from } = args;

  return Array.from({ length: count }, (_, index) => {
    const first = REQUIRED[index % REQUIRED.length] as string;
    const second = REQUIRED[(index * stride + 1) % REQUIRED.length] as string;

    return {
      id: from + index,
      required: { instruction: [...new Set([first, second])] },
    };
  });
}

/**
 * Scores a candidate by the required terms its text covers, and says which
 * ones are missing. The feedback is the whole point: it is what a reflective
 * optimizer reads and a score-only one cannot.
 */
function keywordAdapter(args: {
  noise: number;
  allOrNothing?: boolean;
  /**
   * Share of instances the system answers correctly with no help from its
   * prompt. Zero — the default, and the case the other tasks model — makes the
   * output a pure function of the candidate.
   */
  reliability?: number;
}): GepaAdapter<BenchDatum, string, string> {
  const { noise, allOrNothing = false, reliability = 0 } = args;

  return {
    evaluate: ({ batch, candidate }) => {
      const scored = batch.map((datum) => score({ datum, candidate }));

      return {
        outputs: scored.map((entry) => entry.answer),
        scores: scored.map((entry) => entry.score),
        feedback: scored.map((entry) => entry.feedback ?? ""),
        trajectories: scored.map((entry) => entry.answer),
      };
    },

    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) => {
      const records = batch.map((datum, index) => ({
        inputs: { instance: datum.id },
        generatedOutputs: evaluation.outputs[index] ?? "",
        feedback: evaluation.feedback?.[index] ?? "",
        score: evaluation.scores[index],
      }));

      return Object.fromEntries(
        componentsToUpdate.map((component) => [component, records]),
      );
    },
  };

  function score(args: {
    datum: BenchDatum;
    candidate: Candidate;
  }): ScoreResult & {
    answer: string;
  } {
    const { datum, candidate } = args;
    // The output, which is the candidate's text plus whatever the system got
    // right unaided. Scoring the output rather than the candidate is what
    // gives a harvest something to carry.
    const recalled = recalls({ id: datum.id, reliability })
      ? Object.values(datum.required).flat()
      : [];
    const answer = [...Object.values(candidate), ...recalled]
      .join(" ")
      .toLowerCase();

    const missing: string[] = [];
    let covered = 0;
    let total = 0;

    for (const terms of Object.values(datum.required)) {
      for (const term of terms) {
        total += 1;
        if (answer.includes(term.toLowerCase())) {
          covered += 1;
        } else {
          missing.push(term);
        }
      }
    }

    const bloat = DISTRACTORS.filter((term) => answer.includes(term));
    const base = allOrNothing
      ? missing.length === 0
        ? 1
        : 0
      : covered / total;
    const measured = clamp(
      base - bloat.length * BLOAT_PENALTY + jitter({ answer, datum, noise }),
    );

    return {
      score: measured,
      answer,
      feedback: [
        missing.length === 0
          ? "All required terms present."
          : `Missing required terms: ${missing.join(", ")}`,
        bloat.length === 0
          ? ""
          : `Irrelevant terms to remove: ${bloat.join(", ")}`,
      ]
        .filter(Boolean)
        .join(" "),
    };
  }
}

/**
 * Whether the system answers this instance right without being shown how.
 *
 * A function of the instance alone, so a run is reproducible and two instances
 * needing the same term can differ — which is the whole mechanism: the one it
 * got right is harvestable, and showing it makes the one it got wrong right
 * too.
 */
function recalls(args: { id: number; reliability: number }): boolean {
  const { id, reliability } = args;
  if (reliability === 0) {
    return false;
  }
  return hash32(`recall:${id}`) % 1000 < reliability * 1000;
}

/**
 * Measurement noise that is a function of what was measured, so a run is
 * reproducible and a re-measurement of the same text on the same instance
 * repeats — which is what a cached score assumes.
 */
function jitter(args: {
  answer: string;
  datum: BenchDatum;
  noise: number;
}): number {
  const { answer, datum, noise } = args;
  if (noise === 0) {
    return 0;
  }

  const hash = hash32(`${datum.id}:${answer}`);
  return ((hash % 2001) / 1000 - 1) * noise;
}

/** A window of the term pool, wrapping, so successive drafts overlap but differ. */
function draw(args: { from: number; count: number }): string[] {
  const { from, count } = args;
  return Array.from(
    { length: count },
    (_, offset) => POOL[(from + offset) % POOL.length] as string,
  );
}

function hash32(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function fence(text: string): string {
  return `\`\`\`\n${text}\n\`\`\``;
}

function currentText(prompt: string): string {
  const match = prompt.match(
    /<current_instruction>\n([\s\S]*?)\n<\/current_instruction>/,
  );
  return match?.[1]?.trim() ?? "";
}

/** The highest-scoring attempt a score-history prompt lists, when it lists any. */
function bestAttempt(prompt: string): string | undefined {
  let bestScore = Number.NEGATIVE_INFINITY;
  let best: string | undefined;

  for (const match of prompt.matchAll(
    /score:\s*([\d.]+)[\s\S]*?<instruction>\n?([\s\S]*?)\n?<\/instruction>/g,
  )) {
    const value = Number(match[1]);
    if (value > bestScore) {
      bestScore = value;
      best = (match[2] ?? "").trim();
    }
  }
  return best;
}

function adviceComponents(prompt: string): string[] {
  const match = prompt.match(/<components>\n([\s\S]*?)\n<\/components>/);
  return (match?.[1] ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function missingTerms(prompt: string): string[] {
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
