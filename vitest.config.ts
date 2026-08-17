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
        find: "@ctdio/gepa/testing",
        replacement: sourceOf("./packages/core/src/testing.ts"),
      },
      {
        find: "@ctdio/gepa",
        replacement: sourceOf("./packages/core/src/index.ts"),
      },
    ],
  },
  test: {
    include: ["packages/*/src/**/*.test.ts"],
    environment: "node",
  },
});
