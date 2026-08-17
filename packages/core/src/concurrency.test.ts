import { describe, expect, test } from "vitest";
import { mapWithConcurrency } from "./concurrency.js";

describe("mapWithConcurrency", () => {
  test("preserves input order in the results", async () => {
    const results = await mapWithConcurrency({
      items: [30, 10, 20],
      limit: 3,
      task: async (value) => {
        await delay(value);
        return value * 2;
      },
    });

    expect(results).toEqual([60, 20, 40]);
  });

  test("never exceeds the concurrency limit", async () => {
    let active = 0;
    let peak = 0;

    await mapWithConcurrency({
      items: Array.from({ length: 20 }, (_, index) => index),
      limit: 4,
      task: async () => {
        active += 1;
        peak = Math.max(peak, active);
        await delay(5);
        active -= 1;
        return null;
      },
    });

    expect(peak).toBeLessThanOrEqual(4);
    expect(peak).toBeGreaterThan(1);
  });

  test("passes the item index to the task", async () => {
    const results = await mapWithConcurrency({
      items: ["a", "b", "c"],
      limit: 2,
      task: async (item, index) => `${index}:${item}`,
    });

    expect(results).toEqual(["0:a", "1:b", "2:c"]);
  });

  test("rejects when a task throws", async () => {
    await expect(
      mapWithConcurrency({
        items: [1, 2, 3],
        limit: 2,
        task: async (value) => {
          if (value === 2) {
            throw new Error("boom");
          }
          return value;
        },
      }),
    ).rejects.toThrow("boom");
  });

  test("stops dispatching new items once the signal aborts", async () => {
    const controller = new AbortController();
    let dispatched = 0;

    await expect(
      mapWithConcurrency({
        items: Array.from({ length: 20 }, (_, index) => index),
        limit: 2,
        signal: controller.signal,
        task: async () => {
          dispatched += 1;
          controller.abort();
          await delay(1);
          return null;
        },
      }),
    ).rejects.toThrow(/abort/i);

    // The two in-flight workers finish their current item; nothing after that
    // is dispatched.
    expect(dispatched).toBeLessThanOrEqual(2);
  });

  test("rejects immediately when the signal is already aborted", async () => {
    let dispatched = 0;

    await expect(
      mapWithConcurrency({
        items: [1, 2, 3],
        limit: 2,
        signal: AbortSignal.abort(),
        task: async () => {
          dispatched += 1;
          return null;
        },
      }),
    ).rejects.toThrow(/abort/i);

    expect(dispatched).toBe(0);
  });

  test("returns an empty array for empty input", async () => {
    const results = await mapWithConcurrency({
      items: [],
      limit: 4,
      task: async () => 1,
    });

    expect(results).toEqual([]);
  });
});

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
