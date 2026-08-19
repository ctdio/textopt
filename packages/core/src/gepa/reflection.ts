import { parseProposedText } from "../text.js";
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

const TRUNCATION_MARKER = "… [truncated]";
const MIN_STRING_BUDGET = 40;

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
 * Cut rather than add. Reflective evolution only ever appends — every
 * iteration diagnoses a failure and writes a rule preventing it — so an
 * instruction grows monotonically until it is mostly edge cases that no longer
 * fire. Nothing else in the loop ever removes one.
 */
export function buildSimplifyPrompt(args: ReflectionPromptArgs): string {
  const { componentName, currentText, records } = args;

  return [
    `The instruction below drives the "${componentName}" component of a larger system. It has been edited many times and has accumulated rules.`,
    "",
    "<current_instruction>",
    currentText,
    "</current_instruction>",
    "",
    "Here is how it behaved on recent inputs, with feedback on each output:",
    "",
    "<examples>",
    serializeRecords(records),
    "</examples>",
    "",
    "Write a shorter instruction.",
    "Remove any rule that is redundant, that restates something already said, that contradicts another rule, or that no longer earns the space it takes.",
    "Keep every rule the examples show is load-bearing. The component must still behave the same way on the inputs above.",
    "Do not add new rules. If nothing can be removed, say the same thing in fewer words.",
    "",
    "Return only the new instruction, inside a ``` block.",
  ].join("\n");
}

/**
 * Replace a rule that fits the instances it was written from with the
 * principle behind it. Feedback is drawn from minibatches, so a rule written
 * to fix three examples routinely encodes those three examples.
 */
export function buildGeneralizePrompt(args: ReflectionPromptArgs): string {
  const { componentName, currentText, records } = args;

  return [
    `The instruction below drives the "${componentName}" component of a larger system.`,
    "",
    "<current_instruction>",
    currentText,
    "</current_instruction>",
    "",
    "It was written from a small sample of inputs, so parts of it may describe those specific inputs rather than the task. Here is how it behaved on recent inputs, with feedback on each output:",
    "",
    "<examples>",
    serializeRecords(records),
    "</examples>",
    "",
    "Write a new instruction that states the underlying principle instead of the special cases.",
    "Where a rule names a specific input, value or phrasing, ask what general property that case is an instance of, and write that property.",
    "Keep concrete domain facts that are genuinely fixed — names, thresholds, formats. Those are knowledge, not overfitting.",
    "",
    "Return only the new instruction, inside a ``` block.",
  ].join("\n");
}

/**
 * Start from the evidence rather than from the incumbent. Every other strategy
 * edits the current text, which anchors each proposal to whatever the search
 * happened to reach first; this one is the only escape from a bad opening.
 */
export function buildRewritePrompt(args: ReflectionPromptArgs): string {
  const { componentName, records } = args;

  return [
    `Write the instruction for the "${componentName}" component of a larger system, from scratch.`,
    "",
    "Below are task inputs the component received, the outputs it produced, and feedback on how each output could be better:",
    "",
    "<examples>",
    serializeRecords(records),
    "</examples>",
    "",
    "You are deliberately not being shown the instruction currently in use. Work out what the component is for from the inputs and outputs alone.",
    "Infer the task, its input format, and what a correct output looks like.",
    "State explicitly any domain-specific facts the task depends on — the component will not have access to these examples in future.",
    "",
    "Return only the new instruction, inside a ``` block.",
  ].join("\n");
}

/**
 * A rotation covering the four directions a proposal can move in: fix what is
 * broken, cut what is dead, widen what is too narrow, and start over.
 *
 * Drawing a proposal k times from one template samples one direction k times.
 * Rotating costs nothing extra — same call count, same rollouts — and is the
 * cheapest diversity available. Opt in via `reflection.strategies`; the
 * default stays the published single prompt.
 */
export function diverseReflectionStrategies(): ReflectionPromptBuilder[] {
  return [
    buildReflectionPrompt,
    buildSimplifyPrompt,
    buildGeneralizePrompt,
    buildRewritePrompt,
  ];
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
    /**
     * Rotated by `attempt`, one direction per proposal. Mutually exclusive
     * with `buildPrompt`: passing both leaves it ambiguous which one a given
     * proposal used, which is exactly the thing a rotation has to be legible
     * about.
     */
    strategies?: readonly ReflectionPromptBuilder[];
    limits?: ReflectionLimits;
  } = {},
): (args: ProposeArgs<K>) => Promise<ComponentPatch<K>> {
  const { buildPrompt, strategies, limits = {} } = options;

  if (buildPrompt !== undefined && strategies !== undefined) {
    throw new Error(
      "createDefaultProposer takes buildPrompt or strategies, not both",
    );
  }
  if (strategies !== undefined && strategies.length === 0) {
    throw new Error(
      "createDefaultProposer requires a non-empty strategies list",
    );
  }

  const rotation = strategies ?? [buildPrompt ?? buildReflectionPrompt];

  return async (args) => {
    const {
      candidate,
      reflectiveDataset,
      componentsToUpdate,
      rejectedProposals,
      attempt = 0,
      reflect,
      signal,
    } = args;

    const proposed: ComponentPatch<K> = {};
    const strategy = rotation[
      attempt % rotation.length
    ] as ReflectionPromptBuilder;

    for (const componentName of componentsToUpdate) {
      const records = reflectiveDataset[componentName];
      if (records === undefined || records.length === 0) {
        continue;
      }

      const currentText = candidate[componentName] ?? "";
      const response = await reflect({
        prompt: strategy({
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
