import { defineConfig } from "tsdown";

export default defineConfig({
  entry: [
    "src/index.ts",
    "src/bootstrap-search/index.ts",
    "src/gepa/index.ts",
    "src/mipro/index.ts",
    "src/opro/index.ts",
    "src/simba/index.ts",
    "src/random-search/index.ts",
    "src/file-cache.ts",
    "src/file-reporter.ts",
    "src/testing.ts",
  ],
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
});
