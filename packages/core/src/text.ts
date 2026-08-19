const LANGUAGE_TAG = /^[a-zA-Z0-9_+.-]*\n/;
const DANGLING_OPEN_FENCE = /^\s*```\S*\n?/;
const DANGLING_CLOSE_FENCE = /\n?```\s*$/;

/**
 * Pull the proposed text out of the reflection model's response.
 *
 * Spans the *first* fence to the *last* one rather than matching blocks
 * individually: proposed instructions routinely contain their own fenced
 * examples, and per-block matching would silently return only the trailing
 * fragment. A response with a single fence was truncated mid-generation, so the
 * stray fence is stripped and the partial text kept.
 */
export function parseProposedText(response: string): string {
  const start = response.indexOf("```");
  const end = response.lastIndexOf("```");

  if (start !== -1 && start !== end) {
    return response
      .slice(start + 3, end)
      .replace(LANGUAGE_TAG, "")
      .trim();
  }

  return response
    .replace(DANGLING_OPEN_FENCE, "")
    .replace(DANGLING_CLOSE_FENCE, "")
    .trim();
}
