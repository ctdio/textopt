import { describe, expect, test } from "vitest";
import { parseDemos } from "../demos.js";
import { KEYWORD_EXAMPLES, createKeywordAdapter } from "../testing.js";
import type { TextModel } from "../types.js";
import { createDemoProposer } from "./demos.js";
import { GepaOptimizer } from "./optimize.js";
import type { ProposeArgs, ReflectiveRecord } from "./types.js";

type Component = "instruction" | "examples";

const NEVER_CALLED: TextModel = async () => {
  throw new Error("the reflection model should not have been called");
};

function record(args: {
  question: string;
  answer: string;
  score: number;
}): ReflectiveRecord {
  return {
    inputs: { question: args.question },
    generatedOutputs: args.answer,
    feedback: args.score === 1 ? "Correct." : "Wrong.",
    score: args.score,
  };
}

function proposeArgs(
  overrides: Partial<ProposeArgs<Component>> = {},
): ProposeArgs<Component> {
  return {
    candidate: { instruction: "Answer the question.", examples: "" },
    reflectiveDataset: {
      examples: [
        record({ question: "reset?", answer: "Hold ten seconds.", score: 1 }),
        record({ question: "refund?", answer: "No idea.", score: 0 }),
        record({
          question: "upgrade?",
          answer: "Billing, prorated.",
          score: 1,
        }),
      ],
    },
    componentsToUpdate: ["examples"],
    reflect: NEVER_CALLED,
    ...overrides,
  };
}

describe("createDemoProposer", () => {
  test("writes the rollouts that scored well into the demo component", async () => {
    const propose = createDemoProposer<Component>({ components: ["examples"] });

    const patch = await propose(proposeArgs());
    const demos = parseDemos(patch.examples ?? "");

    expect(demos).toHaveLength(2);
    expect(JSON.stringify(demos)).toContain("Hold ten seconds.");
    expect(JSON.stringify(demos)).not.toContain("No idea.");
  });

  test("makes no reflection call for a demo component", async () => {
    const propose = createDemoProposer<Component>({ components: ["examples"] });

    // NEVER_CALLED throws; reaching a patch at all is the assertion.
    await expect(propose(proposeArgs())).resolves.toHaveProperty("examples");
  });

  test("keeps the demos already in the component and appends to them", async () => {
    const propose = createDemoProposer<Component>({ components: ["examples"] });

    const first = await propose(proposeArgs());
    const second = await propose(
      proposeArgs({
        candidate: {
          instruction: "Answer the question.",
          examples: first.examples ?? "",
        },
        reflectiveDataset: {
          examples: [
            record({
              question: "contact?",
              answer: "Ticket, portal.",
              score: 1,
            }),
          ],
        },
      }),
    );

    const demos = parseDemos(second.examples ?? "");
    expect(demos).toHaveLength(3);
    expect(JSON.stringify(demos)).toContain("Hold ten seconds.");
    expect(JSON.stringify(demos)).toContain("Ticket, portal.");
  });

  test("never keeps the same input twice", async () => {
    const propose = createDemoProposer<Component>({ components: ["examples"] });

    const first = await propose(proposeArgs());
    const second = await propose(
      proposeArgs({
        candidate: {
          instruction: "Answer the question.",
          examples: first.examples ?? "",
        },
        reflectiveDataset: {
          examples: [
            // One the block already holds, one it does not.
            record({
              question: "reset?",
              answer: "Hold ten seconds.",
              score: 1,
            }),
            record({
              question: "contact?",
              answer: "Ticket, portal.",
              score: 1,
            }),
          ],
        },
      }),
    );

    expect(parseDemos(second.examples ?? "")).toHaveLength(3);
  });

  test("proposes nothing when every harvested demo is already held", async () => {
    const propose = createDemoProposer<Component>({ components: ["examples"] });

    const first = await propose(proposeArgs());
    const second = await propose(
      proposeArgs({
        candidate: {
          instruction: "Answer the question.",
          examples: first.examples ?? "",
        },
      }),
    );

    expect(second.examples).toBeUndefined();
  });

  test("drops the oldest demos once the block is full", async () => {
    const propose = createDemoProposer<Component>({
      components: ["examples"],
      maxDemos: 2,
    });

    const first = await propose(proposeArgs());
    const second = await propose(
      proposeArgs({
        candidate: {
          instruction: "Answer the question.",
          examples: first.examples ?? "",
        },
        reflectiveDataset: {
          examples: [
            record({
              question: "contact?",
              answer: "Ticket, portal.",
              score: 1,
            }),
          ],
        },
      }),
    );

    const demos = parseDemos(second.examples ?? "");
    expect(demos).toHaveLength(2);
    expect(JSON.stringify(demos)).not.toContain("Hold ten seconds.");
    expect(JSON.stringify(demos)).toContain("Ticket, portal.");
  });

  test("proposes nothing when no rollout cleared the threshold", async () => {
    const propose = createDemoProposer<Component>({ components: ["examples"] });

    const patch = await propose(
      proposeArgs({
        reflectiveDataset: {
          examples: [
            record({ question: "refund?", answer: "No idea.", score: 0 }),
          ],
        },
      }),
    );

    expect(patch.examples).toBeUndefined();
  });

  test("hands every other component to the fallback proposer", async () => {
    const propose = createDemoProposer<Component>({
      components: ["examples"],
      fallback: async ({ componentsToUpdate }) =>
        Object.fromEntries(
          componentsToUpdate.map((name) => [name, `rewritten ${name}`]),
        ),
    });

    const patch = await propose(
      proposeArgs({
        componentsToUpdate: ["instruction", "examples"],
        reflectiveDataset: {
          instruction: [
            record({
              question: "reset?",
              answer: "Hold ten seconds.",
              score: 1,
            }),
          ],
          examples: [
            record({
              question: "reset?",
              answer: "Hold ten seconds.",
              score: 1,
            }),
          ],
        },
      }),
    );

    expect(patch.instruction).toBe("rewritten instruction");
    expect(parseDemos(patch.examples ?? "")).toHaveLength(1);
  });

  test("refuses an empty component list", () => {
    expect(() => createDemoProposer<Component>({ components: [] })).toThrow(
      /at least one component/,
    );
  });
});

describe("createDemoProposer inside a run", () => {
  test("harvests the loop's own rollouts without a reflection call", async () => {
    const adapter = createKeywordAdapter();
    const proposer = createDemoProposer<"instruction" | "examples">({
      components: ["examples"],
      minScore: 0,
    });
    const proposedBlocks: string[] = [];
    let reflectionCalls = 0;

    await new GepaOptimizer({
      minibatchSize: 2,
      seed: 1,
      maxIterations: 4,
    }).optimize({
      componentSelector: () => ["examples"],
      seedCandidate: { instruction: "hold ten seconds", examples: "" },
      trainingSet: KEYWORD_EXAMPLES,
      adapter: {
        ...adapter,
        proposeNewTexts: async (args) => {
          const patch = await proposer(args);
          if (patch.examples !== undefined) {
            proposedBlocks.push(patch.examples);
          }
          return patch;
        },
      },
      reflect: async () => {
        reflectionCalls += 1;
        return "```\nunused\n```";
      },
      maxMetricCalls: 400,
    });

    expect(reflectionCalls).toBe(0);
    expect(proposedBlocks.length).toBeGreaterThan(0);

    // Every demo came out of a minibatch the run was evaluating anyway.
    const questions = KEYWORD_EXAMPLES.map((example) => example.question);
    for (const demo of parseDemos(proposedBlocks[0] as string)) {
      expect(questions).toContain(
        (demo.input as { question: string }).question,
      );
    }
  });
});
