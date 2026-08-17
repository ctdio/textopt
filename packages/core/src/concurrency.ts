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

  async function worker(): Promise<void> {
    while (cursor < items.length) {
      signal?.throwIfAborted();
      const index = cursor;
      cursor += 1;
      results[index] = await task(items[index] as Item, index);
    }
  }

  await Promise.all(Array.from({ length: workerCount }, () => worker()));

  return results;
}
