import type {
  ComponentPatch,
  ProposeArgs,
  ReflectiveRecord,
  RejectedProposal,
} from "./types.js";

export interface ReflectionPromptArgs {
  componentName: string;
  currentText: string;
  records: readonly ReflectiveRecord[];
  /** Texts already tried for this component that lost to their parent. */
  rejected?: readonly RejectedProposal[];
}

const LANGUAGE_TAG = /^[a-zA-Z0-9_+.-]*\n/;
const DANGLING_OPEN_FENCE = /^\s*```\S*\n?/;
const DANGLING_CLOSE_FENCE = /\n?```\s*$/;

/**
 * Adapted from the reflection prompt in the GEPA paper (Agrawal et al., 2025).
 * The instruction to mine domain facts out of the traces matters as much as the
 * instruction to fix failures — most of the lift comes from the model writing
 * down knowledge the traces revealed.
 */
export function buildReflectionPrompt(args: ReflectionPromptArgs): string {
  const { componentName, currentText, records, rejected = [] } = args;

  return [
    `I gave an assistant the following instruction for the "${componentName}" component of a larger system:`,
    "",
    "<current_instruction>",
    currentText,
    "</current_instruction>",
    "",
    "Below are task inputs the assistant received, the outputs it produced, and feedback on how each output could be better:",
    "",
    "<examples>",
    JSON.stringify(records, jsonSafeReplacer, 2),
    "</examples>",
    ...(rejected.length === 0
      ? []
      : [
          "",
          "These instructions have already been tried for this component and scored worse than the one they replaced. Do not propose them again, and do not propose a variation that repeats the idea that made them fail:",
          "",
          "<rejected_instructions>",
          JSON.stringify(rejected, jsonSafeReplacer, 2),
          "</rejected_instructions>",
        ]),
    "",
    "Write a new instruction for this component.",
    "Read the inputs carefully and infer a detailed description of the task the component is solving, including its input format.",
    "Read every output and its feedback. Identify all niche or domain-specific factual information the task depends on and state it explicitly in the instruction — the assistant will not have access to these examples in future.",
    "If the assistant used a generalizable strategy that worked, describe that strategy.",
    "If the feedback shows a recurring failure, add a precise rule that prevents it.",
    "",
    "Return only the new instruction, inside a ``` block.",
  ].join("\n");
}

/**
 * Pull the proposed text out of the reflection model's response.
 *
 * Spans the *first* fence to the *last* one rather than matching blocks
 * individually: proposed instructions routinely contain their own fenced
 * examples, and per-block matching would silently return only the trailing
 * fragment. A response with a single fence was truncated mid-generation, so the
 * stray fence is stripped and the partial text kept.
 */
export function parseProposedText(response: string): string {
  const start = response.indexOf("```");
  const end = response.lastIndexOf("```");

  if (start !== -1 && start !== end) {
    return response
      .slice(start + 3, end)
      .replace(LANGUAGE_TAG, "")
      .trim();
  }

  return response
    .replace(DANGLING_OPEN_FENCE, "")
    .replace(DANGLING_CLOSE_FENCE, "")
    .trim();
}

/**
 * Default instruction proposer: one reflection call per component being
 * updated. Adapters override this via `proposeNewTexts` when components need
 * coupled updates or a structured proposal format.
 */
export function createDefaultProposer(): (
  args: ProposeArgs,
) => Promise<ComponentPatch> {
  return async (args) => {
    const {
      candidate,
      reflectiveDataset,
      componentsToUpdate,
      rejectedProposals,
      reflect,
      signal,
    } = args;

    const proposed: ComponentPatch = {};

    for (const componentName of componentsToUpdate) {
      const records = reflectiveDataset[componentName];
      if (records === undefined || records.length === 0) {
        continue;
      }

      const currentText = candidate[componentName] ?? "";
      const response = await reflect({
        prompt: buildReflectionPrompt({
          componentName,
          currentText,
          records,
          rejected: rejectedProposals?.[componentName],
        }),
        signal,
      });
      const newText = parseProposedText(response);

      if (newText.length > 0 && newText !== currentText) {
        proposed[componentName] = newText;
      }
    }

    return proposed;
  };
}

function jsonSafeReplacer(_key: string, value: unknown): unknown {
  if (value instanceof Error) {
    return { name: value.name, message: value.message };
  }
  if (value instanceof Map) {
    return Object.fromEntries(value);
  }
  if (value instanceof Set) {
    return [...value];
  }
  return value;
}
