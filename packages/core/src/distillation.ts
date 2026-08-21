import type { Rollout } from "./harvest.js";

/** A turn in a training example, in the shape trainers read. */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface TrainingExample {
  messages: ChatMessage[];
}

/**
 * Turns one harvested rollout into the example to train on, or `null` to skip
 * it. The callback exists because only the caller knows how to render a
 * `Datum` — the library never sees inside one.
 *
 * It is also where the consequential decision lives: how much of the optimized
 * candidate to leave in the input. Dropping it entirely moves the whole prompt
 * into weights and gives up the ability to steer the model with text
 * afterwards; keeping a short task statement distills away only the tokens the
 * search accreted. See `docs/distillation.md` in this package.
 */
export type TrainingExampleRenderer<Datum, Output> = (args: {
  rollout: Rollout<Datum, Output>;
  index: number;
}) => TrainingExample | null;

const ROLES = new Set(["system", "user", "assistant"]);

/**
 * Serialize harvested rollouts as JSONL, one training example per line.
 *
 * The chat-messages shape rather than any one vendor's: it is what Axolotl,
 * Together, Fireworks and the Hugging Face trainers all ingest, and the
 * providers that read it outlive the ones that do not.
 *
 * Returns the text rather than writing it. Only `file-cache` touches the
 * filesystem, and a caller uploading this straight to a provider should not
 * have to round-trip it through a file to do so.
 */
export function toTrainingJsonl<Datum, Output>(args: {
  rollouts: readonly Rollout<Datum, Output>[];
  render: TrainingExampleRenderer<Datum, Output>;
}): string {
  const { rollouts, render } = args;

  const lines: string[] = [];

  for (const [index, rollout] of rollouts.entries()) {
    const example = render({ rollout, index });
    if (example === null) {
      continue;
    }
    assertUsable(example, index);
    lines.push(JSON.stringify(example));
  }

  return lines.join("\n");
}

/**
 * Checked here rather than left to the provider. A malformed line surfaces as
 * a rejected upload hours later, naming a line number in a file the caller
 * never wrote by hand; naming the rollout at the point it was rendered is the
 * same error while it is still fixable.
 */
function assertUsable(example: TrainingExample, index: number): void {
  if (!Array.isArray(example.messages) || example.messages.length === 0) {
    throw new Error(`rollout ${index} rendered no messages`);
  }

  for (const message of example.messages) {
    if (!ROLES.has(message.role)) {
      throw new Error(
        `rollout ${index} rendered the unknown role ${message.role}`,
      );
    }
    if (typeof message.content !== "string") {
      throw new Error(`rollout ${index} rendered non-text content`);
    }
  }
}
