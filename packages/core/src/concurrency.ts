/**
 * Order-preserving concurrency-limited map. GEPA's workload is entirely IO
 * bound, so adapters use this to fan out a batch across a bounded number of
 * in-flight model calls.
 */
export async function mapWithConcurrency<Item, Result>(args: {
  items: readonly Item[];
  limit: number;
  task: (item: Item, index: number) => Promise<Result>;
  /**
   * Checked before each dispatch. Aborting stops the fan-out rather than
   * letting the remaining batch items spend rollouts on a cancelled run.
   */
  signal?: AbortSignal;
}): Promise<Result[]> {
  const { items, limit, task, signal } = args;

  if (items.length === 0) {
    return [];
  }

  signal?.throwIfAborted();

  const workerCount = Math.max(1, Math.min(Math.floor(limit), items.length));
  const results = new Array<Result>(items.length);
  let cursor = 0;
  /**
   * The first failure, kept rather than thrown so the workers that are already
   * running settle before the caller gets control back. A rejection that left
   * tasks running in the background would go on spending an optimizer's budget
   * and writing its caches after the run it belonged to had ended.
   */
  let failure: { err: unknown } | undefined;

  async function worker(): Promise<void> {
    while (cursor < items.length && failure === undefined) {
      const index = cursor;
      cursor += 1;
      try {
        signal?.throwIfAborted();
        results[index] = await task(items[index] as Item, index);
      } catch (err) {
        failure ??= { err };
        return;
      }
    }
  }

  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  if (failure !== undefined) {
    throw failure.err;
  }
  return results;
}
