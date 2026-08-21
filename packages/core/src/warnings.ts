/**
 * Something about a run that its own numbers cannot say.
 *
 * A search reports a score, a stop reason and what it spent, and every one of
 * those reads the same whether the run measured what the caller thinks it did
 * or not. These are the conditions under which a normal-looking result means
 * less than it appears to: they never stop a run, and they are carried on the
 * result and the `finish` event so a report can say so next to the number.
 */
export interface RunWarning {
  code: RunWarningCode;
  message: string;
}

export type RunWarningCode =
  "validationSetReusesTraining" | "seedScoreSaturated" | "seedScoreFloored";

/**
 * The validation set a run will actually select against, and whatever the
 * choice costs it.
 *
 * Defaulting to the training set is the right default for a first run and the
 * wrong number to report from one. It is worse than ordinary overfitting under
 * reflective search: the reflection prompt asks the model to mine domain facts
 * out of the traces it is shown, so those facts come out of the very instances
 * that then select the candidate carrying them. `"reuseTraining"` is the same
 * behaviour with the caller's name on it, and silences the warning.
 */
export function resolveValidationSet<Datum>(args: {
  validationSet: readonly Datum[] | "reuseTraining" | undefined;
  trainingSet: readonly Datum[];
}): { validationSet: readonly Datum[]; warnings: RunWarning[] } {
  const { validationSet, trainingSet } = args;

  if (validationSet === undefined) {
    return {
      validationSet: trainingSet,
      warnings: [
        {
          code: "validationSetReusesTraining",
          message:
            'No validationSet was given, so the search selected candidates on the training instances reflection read. bestScore is fitted to them; pass a validationSet, or a testSet to measure the gap, or validationSet: "reuseTraining" to accept it.',
        },
      ],
    };
  }

  return {
    validationSet:
      validationSet === "reuseTraining" ? trainingSet : validationSet,
    warnings: [],
  };
}

/**
 * What the seed's own validation row says about whether the run could have
 * learned anything.
 *
 * A search ranks candidates by how they differ across instances, so a seed row
 * with no spread leaves nothing to rank: at the ceiling every proposal is a tie
 * the acceptance test resolves by noise, and at the floor no proposal has a
 * partial improvement to build on. Both produce a run that spends its whole
 * budget and reports a stop reason that looks like any other.
 */
export function seedScoreWarnings(args: {
  scores: readonly (number | undefined)[];
  perfectScore: number;
}): RunWarning[] {
  const { scores, perfectScore } = args;

  const measured = scores.filter(
    (score): score is number => score !== undefined,
  );
  if (measured.length === 0) {
    return [];
  }

  if (measured.every((score) => score >= perfectScore)) {
    return [
      {
        code: "seedScoreSaturated",
        message: `The seed candidate already scores ${perfectScore} on every validation instance, so no proposal has anything to improve on and the search ranks ties. Use a harder validation set or a metric that separates these instances.`,
      },
    ];
  }

  if (measured.every((score) => score <= 0)) {
    return [
      {
        code: "seedScoreFloored",
        message:
          "The seed candidate scores 0 on every validation instance. That is a seed with everything to gain when the feedback says what is missing, and a metric that scores nothing when it does not — the score alone cannot tell the two apart. Check the metric against a candidate you know is good before reading much into this run.",
      },
    ];
  }

  return [];
}
