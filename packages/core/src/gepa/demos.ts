import { formatDemos, parseDemos } from "../demos.js";
import type { Demo, DemoRenderer } from "../demos.js";
import { createDefaultProposer } from "./reflection.js";
import type { ComponentPatch, ProposeArgs, ReflectiveRecord } from "./types.js";

const DEFAULT_MAX_DEMOS = 4;
const DEFAULT_MIN_SCORE = 1;

/**
 * A proposer that fills demonstration components from rollouts the run has
 * already paid for.
 *
 * The reflective dataset carries every minibatch rollout's input, output and
 * score, so the successful ones are a few-shot block sitting in memory. Taking
 * them costs no rollout and no reflection call — the search buys demos as a
 * side effect of the evaluations it was making anyway.
 *
 * A proposal appends to the block its parent already holds rather than
 * replacing it: a block built from one minibatch alone would hold three or
 * four examples and forget every earlier one. Accumulation therefore follows
 * the accepted lineage — a demo only persists if the candidate carrying it
 * beat its parent, which is the same bar every other component is held to.
 */
export function createDemoProposer<K extends string = string>(args: {
  /** Components that hold demonstrations rather than instructions. */
  components: readonly K[];
  /** Score a rollout must reach to be kept as a demo. Default 1. */
  minScore?: number;
  /** Demos a block holds before the oldest are dropped. Default 4. */
  maxDemos?: number;
  render?: DemoRenderer;
  /**
   * Handles the components not named above. Defaults to ordinary reflection,
   * so a candidate mixing instructions and demos needs nothing else wired up.
   */
  fallback?: (args: ProposeArgs<K>) => Promise<ComponentPatch<K>>;
}): (args: ProposeArgs<K>) => Promise<ComponentPatch<K>> {
  const {
    components,
    minScore = DEFAULT_MIN_SCORE,
    maxDemos = DEFAULT_MAX_DEMOS,
    render,
    fallback = createDefaultProposer<K>(),
  } = args;

  if (components.length === 0) {
    throw new Error("createDemoProposer requires at least one component");
  }
  const demoComponents = new Set<K>(components);

  return async (proposeArgs) => {
    const { candidate, reflectiveDataset, componentsToUpdate } = proposeArgs;

    const demoTargets = componentsToUpdate.filter((name) =>
      demoComponents.has(name),
    );
    const others = componentsToUpdate.filter(
      (name) => !demoComponents.has(name),
    );

    const patch: ComponentPatch<K> =
      others.length === 0
        ? {}
        : await fallback({ ...proposeArgs, componentsToUpdate: others });

    for (const name of demoTargets) {
      const harvested = harvestDemos({
        records: reflectiveDataset[name] ?? [],
        minScore,
      });
      if (harvested.length === 0) {
        continue;
      }

      const kept = mergeDemos({
        existing: parseDemos(candidate[name] ?? ""),
        harvested,
        maxDemos,
      });
      const block = formatDemos(kept, render === undefined ? {} : { render });

      if (block !== candidate[name]) {
        patch[name] = block;
      }
    }

    return patch;
  };
}

function harvestDemos(args: {
  records: readonly ReflectiveRecord[];
  minScore: number;
}): Demo[] {
  const { records, minScore } = args;

  return records
    .filter((record) => (record.score ?? Number.NEGATIVE_INFINITY) >= minScore)
    .map((record) => ({
      input: record.inputs,
      output: record.generatedOutputs,
      score: record.score,
    }));
}

/**
 * Newest wins on overflow. A demo harvested later came from a stronger
 * candidate, since a weaker one would not have scored highly enough to be
 * harvested at all — so the tail of the block is the better end of it.
 */
function mergeDemos(args: {
  existing: readonly Demo[];
  harvested: readonly Demo[];
  maxDemos: number;
}): Demo[] {
  const { existing, harvested, maxDemos } = args;

  const merged: Demo[] = [...existing];
  const seen = new Set(existing.map((demo) => keyOf(demo.input)));

  for (const demo of harvested) {
    const key = keyOf(demo.input);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    merged.push(demo);
  }

  return merged.slice(Math.max(0, merged.length - maxDemos));
}

function keyOf(input: unknown): string {
  try {
    return JSON.stringify(input) ?? String(input);
  } catch {
    return String(input);
  }
}
