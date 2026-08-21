import type { ScoreResult, TextModel } from "./types.js";

/** One thing the judge grades, and what a perfect answer looks like for it. */
export interface JudgeCriterion {
  name: string;
  description: string;
  /**
   * Share of the instance score this criterion carries, relative to the other
   * criteria. Default 1, which is the unweighted mean.
   *
   * 0 removes a criterion from the aggregate, not from the search. It is still
   * graded and still recorded in `objectiveScores`, which
   * `paretoSelector({ frontier: "objective" })` and `"hybrid"` build their
   * selection fronts from — so a candidate leading a zero-weight criterion
   * still earns parent selection under those. Under the default instance
   * frontier, 0 is enough. For a number that can never steer the search,
   * compute it outside the judge.
   */
  weight?: number;
  /**
   * Grade, on the judge's own scale, that this criterion must reach for the
   * instance to score at all. Below it the instance scores 0 whatever the
   * other criteria said.
   *
   * A mean lets a search trade a hard requirement away: a candidate that tanks
   * one non-negotiable criterion and aces three cosmetic ones outranks the
   * incumbent that kept the rule. Anything a caller would not ship without is
   * a gate rather than a term in the average.
   *
   * Enforce it once. A gate already makes the requirement non-negotiable, so a
   * heavy `weight` on the same criterion redistributes score only among
   * candidates that all cleared it — and pins that share of the aggregate near
   * its ceiling, narrowing the range the search has left to move in. Gate it,
   * then weight it low.
   *
   * @see docs/metric-preflight.md
   */
  gate?: number;
}

export type JudgePromptBuilder = (args: {
  input: string;
  output: string;
  expected?: string;
  criteria: readonly JudgeCriterion[];
  scale: number;
}) => string;

export type Judge<Datum, Output> = (args: {
  input: Datum;
  output: Output;
  /** The gold answer, when the caller has one. */
  expected?: Output;
  signal?: AbortSignal;
}) => Promise<ScoreResult>;

const DEFAULT_SCALE = 5;
const SCORE = /<score\s+name="([^"]+)"\s*>\s*([\d.]+)\s*<\/score>/g;
const FEEDBACK = /<feedback>([\s\S]*?)<\/feedback>/;

/**
 * A model-graded metric that returns written feedback alongside the score.
 *
 * The feedback is the point. A judge that returns only a number reduces a whole
 * paragraph of diagnosis to one scalar, and reflective search — the thing most
 * likely to be pointed at a task too open-ended to score by string match — runs
 * on exactly that diagnosis. So the prompt demands feedback, and demands it be
 * addressed to the instructions rather than to the graded output: "this answer
 * should have mentioned the refund window" tells a rewriting model nothing that
 * "the instruction never says to state the refund window" does not say better.
 *
 * Grades on a small integer scale and normalizes afterwards. Models discriminate
 * between 2 and 4 far more reliably than between 0.4 and 0.8, and the scale is
 * the caller's to widen once they have seen the judge bunch its answers.
 */
export function createJudge<Datum = string, Output = string>(args: {
  model: TextModel;
  criteria: readonly JudgeCriterion[];
  /** Highest grade the judge may award per criterion. Default 5. */
  scale?: number;
  renderInput?: (input: Datum) => string;
  renderOutput?: (output: Output) => string;
  buildPrompt?: JudgePromptBuilder;
}): Judge<Datum, Output> {
  const {
    model,
    criteria,
    scale = DEFAULT_SCALE,
    renderInput = stringify,
    renderOutput = stringify,
    buildPrompt = buildJudgePrompt,
  } = args;

  if (criteria.length === 0) {
    throw new Error("createJudge requires at least one criterion");
  }
  // Every grade is normalized by dividing by this, so anything else turns a
  // verdict into a non-finite or negative score the search then ranks by.
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`scale must be a positive number, received ${scale}`);
  }
  assertCriteria({ criteria, scale });

  return async ({ input, output, expected, signal }) => {
    const response = await model({
      prompt: buildPrompt({
        input: renderInput(input),
        output: renderOutput(output),
        expected: expected === undefined ? undefined : renderOutput(expected),
        criteria,
        scale,
      }),
      signal,
    });

    return readVerdict({ response, criteria, scale });
  };
}

export function buildJudgePrompt(args: {
  input: string;
  output: string;
  expected?: string;
  criteria: readonly JudgeCriterion[];
  scale: number;
}): string {
  const { input, output, expected, criteria, scale } = args;

  return [
    "You are grading one output of an automated system against the criteria below.",
    "",
    "<input>",
    input,
    "</input>",
    "",
    "<output>",
    output,
    "</output>",
    ...(expected === undefined
      ? []
      : ["", "<expected_answer>", expected, "</expected_answer>"]),
    "",
    "<criteria>",
    criteria
      .map(
        (criterion) =>
          `<${criterion.name}>${criterion.description}</${criterion.name}>`,
      )
      .join("\n"),
    "</criteria>",
    "",
    `Grade each criterion from 0 to ${scale}, where ${scale} is a perfect answer.`,
    "",
    "Then write feedback. It is read by a program that rewrites the system's instructions, not by a person reviewing this output, so say what the instructions should tell the system to do differently. Feedback about this particular answer is of no use to it.",
    ...(expected === undefined
      ? []
      : [
          "",
          "Do not restate the expected answer, or any fact drawn from it, in the feedback. The instruction it is rewritten into is reused on inputs whose answers you have not seen: a fact copied out of the expected answer becomes an answer key memorised in the prompt, which raises the score on this input and teaches the system nothing. Name the kind of thing the answer was missing, not the thing itself.",
        ]),
    "",
    "Reply in exactly this format and nothing else:",
    ...criteria.map((criterion) => `<score name="${criterion.name}">…</score>`),
    "<feedback>…</feedback>",
  ].join("\n");
}

/**
 * A judge that answered off-format graded nothing, so the result is unknown
 * rather than bad. Reporting it as transient is what keeps a formatting failure
 * out of the cache and off the candidate's record — the same treatment a rate
 * limit gets, for the same reason.
 */
function readVerdict(args: {
  response: string;
  criteria: readonly JudgeCriterion[];
  scale: number;
}): ScoreResult {
  const { response, criteria, scale } = args;
  const feedback = response.match(FEEDBACK)?.[1]?.trim() ?? "";
  const graded = new Map<string, number>();

  for (const match of response.matchAll(SCORE)) {
    graded.set(match[1] as string, Number(match[2]));
  }

  const objectiveScores: Record<string, number> = {};
  let gated = false;
  let weighted = 0;
  let totalWeight = 0;

  for (const criterion of criteria) {
    const grade = graded.get(criterion.name);
    if (grade === undefined || Number.isNaN(grade)) {
      return {
        score: 0,
        feedback: `Judge did not grade "${criterion.name}".`,
        transient: true,
      };
    }

    const normalized = clamp(grade / scale);
    objectiveScores[criterion.name] = normalized;

    if (criterion.gate !== undefined && grade < criterion.gate) {
      gated = true;
    }

    const weight = criterion.weight ?? 1;
    weighted += normalized * weight;
    totalWeight += weight;
  }

  // The objectives are reported either way: the aggregate says the instance
  // failed, and only the per-criterion grades say which requirement failed it.
  return {
    score: gated ? 0 : weighted / totalWeight,
    feedback,
    objectiveScores,
  };
}

function assertCriteria(args: {
  criteria: readonly JudgeCriterion[];
  scale: number;
}): void {
  const { criteria, scale } = args;

  let totalWeight = 0;
  for (const { name, weight = 1, gate } of criteria) {
    if (!Number.isFinite(weight) || weight < 0) {
      throw new Error(
        `weight on criterion "${name}" must be a non-negative number, received ${weight}`,
      );
    }
    totalWeight += weight;

    // A gate at or below 0 can never fire and one above the scale can never be
    // cleared, so either one is a requirement the caller believes is enforced
    // and is not.
    if (gate !== undefined && (!Number.isFinite(gate) || gate <= 0)) {
      throw new Error(
        `gate on criterion "${name}" must be greater than 0, received ${gate}; no grade can fall below 0`,
      );
    }
    if (gate !== undefined && gate > scale) {
      throw new Error(
        `gate on criterion "${name}" is ${gate}, above the scale of ${scale}; no grade can reach it`,
      );
    }
  }

  if (totalWeight <= 0) {
    throw new Error(
      "criteria weights must sum to more than 0; at least one criterion has to count towards the score",
    );
  }
}

function stringify(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}
