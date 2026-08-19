import type { ScoreResult, TextModel } from "./types.js";

/** One thing the judge grades, and what a perfect answer looks like for it. */
export interface JudgeCriterion {
  name: string;
  description: string;
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
  for (const criterion of criteria) {
    const grade = graded.get(criterion.name);
    if (grade === undefined || Number.isNaN(grade)) {
      return {
        score: 0,
        feedback: `Judge did not grade "${criterion.name}".`,
        transient: true,
      };
    }
    objectiveScores[criterion.name] = clamp(grade / scale);
  }

  const grades = Object.values(objectiveScores);

  return {
    score: grades.reduce((total, grade) => total + grade, 0) / grades.length,
    feedback,
    objectiveScores,
  };
}

function stringify(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}
