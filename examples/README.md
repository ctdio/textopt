# Examples

Runnable scripts. Keys are read from a root `.env` if present; the offline ones
need no keys and make no network calls.

| Command                                     | Needs               | Demonstrates                                                                                                 |
| ------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `pnpm --filter textopt-examples keyword`    | none                | An offline optimization run.                                                                                 |
| `pnpm --filter textopt-examples pareto`     | none                | The instance-level Pareto frontier.                                                                          |
| `pnpm --filter textopt-examples ai-sdk`     | `ANTHROPIC_API_KEY` | Optimization of one `generateText` call.                                                                     |
| `pnpm --filter textopt-examples langchain`  | `OPENAI_API_KEY`    | Optimization of a LangChain chain.                                                                           |
| `pnpm --filter textopt-examples braintrust` | `OPENAI_API_KEY`    | autoevals scoring and Braintrust logging. Without `BRAINTRUST_API_KEY`, the script prints events locally.    |
| `pnpm --filter textopt-examples merge`      | `ANTHROPIC_API_KEY` | System-aware merging of two components from separate lineages.                                               |
| `pnpm --filter textopt-examples custom`     | `ANTHROPIC_API_KEY` | A custom adapter over a vendor SDK. Set `VENDOR` to use OpenAI instead; this also requires `OPENAI_API_KEY`. |
| `pnpm --filter textopt-examples simba`      | none                | Mini-batch ascent against a metric that returns a different number every time.                               |
| `pnpm --filter textopt-examples bootstrap`  | none                | Few-shot search with no proposal model, and why more demos is not better.                                    |
| `pnpm --filter textopt-examples compare`    | none                | Three optimizers over the same seeds, with p-values.                                                         |
| `pnpm --filter textopt-examples judge`      | `ANTHROPIC_API_KEY` | A model-graded metric built from named criteria.                                                             |
| `pnpm --filter textopt-examples pipeline`   | `ANTHROPIC_API_KEY` | Two modules in sequence, each with its own instruction.                                                      |
