import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

function sourceOf(relativePath: string): string {
  return fileURLToPath(new URL(relativePath, import.meta.url));
}

export default defineConfig({
  resolve: {
    // Ordered: the more specific subpath must be matched before the bare name.
    alias: [
      {
        find: "textopt/testing",
        replacement: sourceOf("./packages/core/src/testing.ts"),
      },
      {
        find: "textopt/mipro",
        replacement: sourceOf("./packages/core/src/mipro/index.ts"),
      },
      {
        find: "textopt/opro",
        replacement: sourceOf("./packages/core/src/opro/index.ts"),
      },
      {
        find: "textopt/random-search",
        replacement: sourceOf("./packages/core/src/random-search/index.ts"),
      },
      {
        find: "textopt/gepa",
        replacement: sourceOf("./packages/core/src/gepa/index.ts"),
      },
      {
        find: "textopt",
        replacement: sourceOf("./packages/core/src/index.ts"),
      },
    ],
  },
  test: {
    include: ["packages/*/src/**/*.test.ts"],
    environment: "node",
  },
});
