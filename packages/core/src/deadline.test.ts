import { describe, expect, test } from "vitest";
import { createDeadline } from "./deadline.js";

describe("createDeadline", () => {
  test("never expires when no limit was set", () => {
    let clock = 0;
    const deadline = createDeadline({ now: () => clock });

    clock = 1_000_000;

    expect(deadline.exceeded()).toBe(false);
  });

  test("has not expired before the limit is reached", () => {
    let clock = 0;
    const deadline = createDeadline({ maxWallClockMs: 100, now: () => clock });

    clock = 99;

    expect(deadline.exceeded()).toBe(false);
  });

  test("expires once the elapsed time reaches the limit", () => {
    let clock = 0;
    const deadline = createDeadline({ maxWallClockMs: 100, now: () => clock });

    clock = 100;

    expect(deadline.exceeded()).toBe(true);
  });

  test("measures from when it was created, not from the epoch", () => {
    let clock = 5_000;
    const deadline = createDeadline({ maxWallClockMs: 100, now: () => clock });

    clock = 5_050;

    expect(deadline.exceeded()).toBe(false);
  });

  test("expires immediately when the limit is zero", () => {
    const deadline = createDeadline({ maxWallClockMs: 0, now: () => 0 });

    expect(deadline.exceeded()).toBe(true);
  });

  test("reports the time left, for a caller deciding whether to start work", () => {
    let clock = 0;
    const deadline = createDeadline({ maxWallClockMs: 100, now: () => clock });

    clock = 40;

    expect(deadline.remainingMs()).toBe(60);
  });

  test("reports unlimited time left when no limit was set", () => {
    const deadline = createDeadline({ now: () => 0 });

    expect(deadline.remainingMs()).toBe(Number.POSITIVE_INFINITY);
  });
});
