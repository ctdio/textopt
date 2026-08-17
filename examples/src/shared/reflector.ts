import type { Reflector } from "@ctdio/gepa";
import { generateText, type LanguageModel } from "ai";

/** Vendor-specific knobs, typed off `generateText` rather than re-declared. */
type ProviderOptions = NonNullable<
  Parameters<typeof generateText>[0]["providerOptions"]
>;

export interface ReflectorSettings {
  model: LanguageModel;
  providerOptions?: ProviderOptions;
  maxOutputTokens?: number;
}

/**
 * A `Reflector` is `text in, text out` — the whole provider seam is these eight
 * lines. This one goes through the AI SDK, so it works with any provider the
 * SDK supports; a LangChain chat model, a raw SDK call, a local model, or a
 * hand-written rule would each be an equally valid `Reflector`.
 *
 * Which model to use is a real decision, so examples make it explicitly rather
 * than sniffing the environment: the system under optimization should be the
 * cheap model — it is the thing you are making better — while reflection wants
 * a frontier model, because it is the component doing the reasoning about
 * failure.
 */
export function createReflector(settings: ReflectorSettings): Reflector {
  const { model, providerOptions = {}, maxOutputTokens = 8192 } = settings;

  return async ({ prompt, signal }) => {
    const result = await generateText({
      model,
      prompt,
      maxOutputTokens,
      abortSignal: signal,
      providerOptions,
    });

    return result.text;
  };
}

export function requireApiKey(envVar: string): void {
  if (process.env[envVar] !== undefined) {
    return;
  }

  console.error(
    `${envVar} is not set.\n` +
      "This example names its models explicitly at the top of the file — edit them to use another provider.\n" +
      "The offline examples need no key at all:\n" +
      "  pnpm --filter @ctdio/gepa-examples keyword\n" +
      "  pnpm --filter @ctdio/gepa-examples pareto",
  );
  process.exit(1);
}
