export interface AdviceTrajectory<Output = unknown> {
  output: Output;
  score: number;
  feedback?: string;
}

export interface AdvicePromptArgs<Datum = unknown, Output = unknown> {
  /** Components the advice is wanted for, named so each can be addressed. */
  components: readonly string[];
  /**
   * What each of those components currently says. Advice is appended to this
   * text rather than replacing it, so a proposer that cannot see it writes
   * blind: it restates what is already there, and cannot correct it.
   */
  current: Record<string, string>;
  input: Datum;
  /** The higher scoring run of this instance, when there is one to contrast. */
  better?: AdviceTrajectory<Output>;
  /** The lower scoring run of this instance, when there is one to contrast. */
  worse?: AdviceTrajectory<Output>;
}

export type AdvicePromptBuilder<Datum = unknown, Output = unknown> = (
  args: AdvicePromptArgs<Datum, Output>,
) => string;

const ADVICE_BLOCK = /<advice\s+component="([^"]+)"\s*>([\s\S]*?)<\/advice>/g;

/**
 * Adapted from SIMBA's `OfferFeedback` signature (Opsahl-Ong et al.'s DSPy).
 *
 * The framing that matters is "build up experience": the component will not
 * see this instance again, so advice that only fixes this one input is wasted.
 * Contrasting two runs of the *same* input is what makes that possible — the
 * inputs are held constant, so the difference in reward is attributable to the
 * behaviour rather than to the instance being easier.
 */
export function buildAdvicePrompt<Datum, Output>(
  args: AdvicePromptArgs<Datum, Output>,
): string {
  const { components, current, input, better, worse } = args;

  return [
    "Two runs of the same system on the same input are shown below, with the reward each earned.",
    "Your job is to write advice that would make the system behave like the higher scoring run the next time it sees a similar input.",
    "",
    "<input>",
    serialize(input),
    "</input>",
    ...(worse === undefined ? [] : trajectoryBlock("worse", worse)),
    ...(better === undefined ? [] : trajectoryBlock("better", better)),
    "",
    "Write advice for each of these components, shown with what it says now:",
    "",
    "<components>",
    components
      .map((component) =>
        [
          `<component name="${component}">`,
          current[component] ?? "",
          "</component>",
        ].join("\n"),
      )
      .join("\n"),
    "</components>",
    "",
    "Your advice is appended to what the component already says. Do not restate advice it already carries; add what is missing or correct what is wrong.",
    "",
    "The component will not have access to this example, so advice that only covers this input is wasted. State the general behaviour it should adopt, and be concrete about when it applies.",
    "Address each component's own sub-task rather than the system as a whole.",
    "Base the advice on what actually differed between the two runs. If nothing useful can be said for a component, leave it out.",
    "",
    'Return one block per component, in the form <advice component="name">…</advice>, and nothing else.',
  ].join("\n");
}

/**
 * Read the per-component advice out of the model's response, ignoring anything
 * written around it. A component the model had nothing to say about is absent
 * rather than empty, so the caller appends nothing instead of appending noise.
 */
export function parseAdvice(response: string): Record<string, string> {
  const advice: Record<string, string> = {};

  for (const match of response.matchAll(ADVICE_BLOCK)) {
    const component = match[1] as string;
    const text = (match[2] ?? "").trim();
    if (text.length > 0) {
      advice[component] = text;
    }
  }
  return advice;
}

function trajectoryBlock<Output>(
  label: "better" | "worse",
  trajectory: AdviceTrajectory<Output>,
): string[] {
  return [
    "",
    `<${label}_trajectory>`,
    `reward: ${trajectory.score}`,
    "<output>",
    serialize(trajectory.output),
    "</output>",
    ...(trajectory.feedback === undefined
      ? []
      : ["<feedback>", trajectory.feedback, "</feedback>"]),
    `</${label}_trajectory>`,
  ];
}

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
