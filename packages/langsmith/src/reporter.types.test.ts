import { Client } from "langsmith";
import { describe, expect, test } from "vitest";
import type { LangSmithClientLike } from "./reporter.js";
import { createLangSmithReporter } from "./reporter.js";

/**
 * `LangSmithClientLike` is declared structurally so this package carries no
 * runtime dependency on the SDK. This is what keeps that declaration honest:
 * if LangSmith changes one of the signatures the reporter calls, the
 * assignment below stops compiling instead of the reporter failing at runtime
 * against a client it claimed to accept.
 */
export function realClientSatisfiesTheStructuralType(client: Client): unknown {
  const accepted: LangSmithClientLike = client;

  return createLangSmithReporter({
    client: accepted,
    dataset: "val",
    experimentPrefix: "run",
    validationSet: [{ question: "why" }],
    toInput: (datum) => ({ question: datum.question }),
  });
}

describe("LangSmithClientLike", () => {
  test("accepts a client that reports no datasets yet", async () => {
    const reporter = createLangSmithReporter({
      client: {
        hasDataset: async () => false,
        readDataset: async () => ({ id: "unused" }),
        createDataset: async () => ({ id: "dataset-0" }),
        createExamples: async (uploads) => uploads,
        createProject: async ({ projectName }) => ({
          id: "project-0",
          name: projectName,
        }),
        createRun: async () => undefined,
        createFeedback: async () => undefined,
      },
      dataset: "val",
      experimentPrefix: "run",
      validationSet: [{ question: "why" }],
    });

    await expect(reporter.flush?.()).resolves.toBeUndefined();
  });
});
