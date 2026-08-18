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

export type ReflectionPromptBuilder = (args: ReflectionPromptArgs) => string;

/**
 * Ceilings on what one reflection call is allowed to carry. Traces are the one
 * input the optimizer cannot bound in advance — a single failing rollout can
 * serialize to hundreds of kilobytes — and an over-long prompt fails the whole
 * call rather than degrading.
 */
export interface ReflectionLimits {
  /** Records shown per component. The worst scoring ones are kept. */
  maxRecords?: number;
  /** Rough ceiling on the characters the records serialize to. */
  maxCharacters?: number;
}

const LANGUAGE_TAG = /^[a-zA-Z0-9_+.-]*\n/;
const TRUNCATION_MARKER = "… [truncated]";
const MIN_STRING_BUDGET = 40;
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
    serializeRecords(records),
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
 * Trims a reflective dataset down to what one prompt should carry: the worst
 * scoring records first, since reflection is about diagnosing failures, and
 * long strings cut to a share of the character budget.
 */
export function limitReflectiveRecords(args: {
  records: readonly ReflectiveRecord[];
  maxRecords?: number;
  maxCharacters?: number;
}): ReflectiveRecord[] {
  const { records, maxRecords, maxCharacters } = args;

  let kept = [...records];

  if (maxRecords !== undefined && kept.length > maxRecords) {
    // Rank by score, then restore the original order: the model reads the
    // records as a sequence, and reordering them by score would imply one.
    const ranked = kept
      .map((record, position) => ({ record, position }))
      .sort(
        (a, b) =>
          (a.record.score ?? Number.POSITIVE_INFINITY) -
          (b.record.score ?? Number.POSITIVE_INFINITY),
      )
      .slice(0, maxRecords)
      .sort((a, b) => a.position - b.position);
    kept = ranked.map((entry) => entry.record);
  }

  if (maxCharacters === undefined || kept.length === 0) {
    return kept;
  }

  const perRecord = Math.max(
    MIN_STRING_BUDGET,
    Math.floor(maxCharacters / kept.length),
  );
  kept = kept.map(
    (record) => truncateStrings(record, perRecord) as ReflectiveRecord,
  );

  // Truncating strings bounds one huge trace; dropping records bounds a long
  // tail of small ones. Both are needed, and at least one record survives —
  // an empty dataset would silently skip the component entirely.
  while (kept.length > 1 && serializeRecords(kept).length > maxCharacters) {
    kept.pop();
  }
  return kept;
}

/**
 * Default instruction proposer: one reflection call per component being
 * updated. Adapters override this via `proposeNewTexts` when components need
 * coupled updates or a structured proposal format; `buildPrompt` is the
 * lighter seam for changing only the wording.
 */
export function createDefaultProposer<K extends string = string>(
  options: {
    buildPrompt?: ReflectionPromptBuilder;
    limits?: ReflectionLimits;
  } = {},
): (args: ProposeArgs<K>) => Promise<ComponentPatch<K>> {
  const { buildPrompt = buildReflectionPrompt, limits = {} } = options;

  return async (args) => {
    const {
      candidate,
      reflectiveDataset,
      componentsToUpdate,
      rejectedProposals,
      reflect,
      signal,
    } = args;

    const proposed: ComponentPatch<K> = {};

    for (const componentName of componentsToUpdate) {
      const records = reflectiveDataset[componentName];
      if (records === undefined || records.length === 0) {
        continue;
      }

      const currentText = candidate[componentName] ?? "";
      const response = await reflect({
        prompt: buildPrompt({
          componentName,
          currentText,
          records: limitReflectiveRecords({ records, ...limits }),
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

function serializeRecords(records: readonly ReflectiveRecord[]): string {
  return JSON.stringify(records, jsonSafeReplacer, 2);
}

/** Recursive so a long trace buried in a nested output is cut too. */
function truncateStrings(value: unknown, budget: number): unknown {
  if (typeof value === "string") {
    return value.length <= budget
      ? value
      : `${value.slice(0, budget)}${TRUNCATION_MARKER}`;
  }
  if (Array.isArray(value)) {
    return value.map((item) => truncateStrings(item, budget));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [
        key,
        truncateStrings(item, budget),
      ]),
    );
  }
  return value;
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
