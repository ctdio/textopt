import { expect, test } from "vitest";
import { consoleReporter } from "./reporting.js";
import type { Reporter } from "./reporting.js";
import { BOOTSTRAP_SEARCH_EVENT_TYPES } from "./bootstrap-search/optimize.js";
import type { BootstrapSearchEvent } from "./bootstrap-search/optimize.js";
import { GEPA_EVENT_TYPES } from "./gepa/types.js";
import type { GepaEvent } from "./gepa/types.js";
import { MIPRO_EVENT_TYPES } from "./mipro/optimize.js";
import type { MiproEvent } from "./mipro/optimize.js";
import { OPRO_EVENT_TYPES } from "./opro/optimize.js";
import type { OproEvent } from "./opro/optimize.js";
import { RANDOM_SEARCH_EVENT_TYPES } from "./random-search/optimize.js";
import type { RandomSearchEvent } from "./random-search/optimize.js";
import { SIMBA_EVENT_TYPES } from "./simba/optimize.js";
import type { SimbaEvent } from "./simba/optimize.js";

/**
 * Each assignment below fails to compile if an event is added to that union
 * without being listed, and the `satisfies` on the list itself fails if a name
 * is listed that no event carries. Together they are what makes the warning a
 * run gives a reporter trustworthy: an unlisted event would make a correct
 * handler look like a typo.
 *
 * Nothing here is called — the compiler is the assertion.
 */
export function everyEventNameIsListed(): unknown[] {
  const gepa: (typeof GEPA_EVENT_TYPES)[number] = "" as GepaEvent["type"];
  const simba: (typeof SIMBA_EVENT_TYPES)[number] = "" as SimbaEvent["type"];
  const opro: (typeof OPRO_EVENT_TYPES)[number] = "" as OproEvent["type"];
  const mipro: (typeof MIPRO_EVENT_TYPES)[number] = "" as MiproEvent["type"];
  const bootstrap: (typeof BOOTSTRAP_SEARCH_EVENT_TYPES)[number] =
    "" as BootstrapSearchEvent["type"];
  const random: (typeof RANDOM_SEARCH_EVENT_TYPES)[number] =
    "" as RandomSearchEvent["type"];

  return [gepa, simba, opro, mipro, bootstrap, random];
}

test("every optimizer names its events once", () => {
  const lists = {
    gepa: GEPA_EVENT_TYPES,
    simba: SIMBA_EVENT_TYPES,
    opro: OPRO_EVENT_TYPES,
    mipro: MIPRO_EVENT_TYPES,
    bootstrapSearch: BOOTSTRAP_SEARCH_EVENT_TYPES,
    randomSearch: RANDOM_SEARCH_EVENT_TYPES,
  };

  for (const [optimizer, names] of Object.entries(lists)) {
    expect([optimizer, new Set(names).size]).toEqual([optimizer, names.length]);
  }
});

test("every optimizer emits the events a cross-optimizer reporter reads", () => {
  // `ReportableEvent` promises these five from any search. A list missing one
  // is an optimizer that quietly reports less than the type says it does.
  const shared = [
    "start",
    "evaluation",
    "rollout",
    "candidateAccepted",
    "finish",
  ];

  for (const names of [
    GEPA_EVENT_TYPES,
    SIMBA_EVENT_TYPES,
    OPRO_EVENT_TYPES,
    MIPRO_EVENT_TYPES,
    BOOTSTRAP_SEARCH_EVENT_TYPES,
    RANDOM_SEARCH_EVENT_TYPES,
  ]) {
    expect(names.filter((name) => shared.includes(name)).toSorted()).toEqual(
      shared.toSorted(),
    );
  }
});

/**
 * `consoleReporter` is typed against the tag alone, which is what lets one
 * reporter drop into every optimizer's `reporters` array. Each position below
 * stops compiling if a search's union stops being assignable to it.
 */
export function oneReporterFitsEveryOptimizer(): {
  gepa: Reporter<GepaEvent>[];
  simba: Reporter<SimbaEvent>[];
  opro: Reporter<OproEvent>[];
  mipro: Reporter<MiproEvent>[];
  bootstrapSearch: Reporter<BootstrapSearchEvent>[];
  randomSearch: Reporter<RandomSearchEvent>[];
} {
  return {
    gepa: [consoleReporter()],
    simba: [consoleReporter()],
    opro: [consoleReporter()],
    mipro: [consoleReporter()],
    bootstrapSearch: [consoleReporter()],
    randomSearch: [consoleReporter()],
  };
}
