/**
 * Optimizing few-shot examples with no proposal model at all.
 *
 * Every other optimizer here needs a second model to write the next candidate.
 * `BootstrapSearchOptimizer` needs none: it runs the system you already have
 * over shuffled training data, keeps the rollouts the metric rewarded, and
 * searches over which of them to show as demonstrations. That makes it the one
 * to reach for when a proposal model is not available — no budget for it, no
 * approved provider — or when the demonstrations are the whole instruction.
 *
 * The system below is deterministic and offline, and it depends on its demos
 * the way a real prompted model does. It can read every date format it is given
 * and has no idea which one to write back: with no demos it echoes the input,
 * which happens to be right for the rows that already arrived in ISO. Those are
 * the rollouts a harvest keeps — and a demo block full of ISO outputs is what
 * tells it to write ISO for everything else too.
 *
 * That is the shape of what bootstrapping can do in general. It amplifies a
 * behaviour the system already produces somewhere; it cannot teach one the
 * system never produces at all.
 *
 * Watch the labels-only candidate lose. Sixteen gold examples is more block
 * than this system can use, and it ends up copying them instead of learning
 * from them — which is why the search is over *which* demos to show, and why
 * zero-shot stays in the running as the baseline to beat.
 *
 * The system reads its own demo block with `parseDemos`, which is why no
 * `renderDemo` is passed: the default rendering is the one that round-trips
 * back into the inputs and outputs it was built from.
 *
 *   pnpm --filter textopt-examples bootstrap
 */
import type { Adapter } from "textopt";
import { parseDemos } from "textopt";
import { BootstrapSearchOptimizer } from "textopt/bootstrap-search";

interface DateRow {
  raw: string;
  iso: string;
}

/** Every input shape the system can read, and how to read one. */
const FORMATS = [
  {
    match: /^(\d{4})-(\d{2})-(\d{2})$/,
    read: (parts: string[]) => `${parts[1]}-${parts[2]}-${parts[3]}`,
  },
  {
    match: /^(\d{2})\/(\d{2})\/(\d{4})$/,
    read: (parts: string[]) => `${parts[3]}-${parts[1]}-${parts[2]}`,
  },
  {
    match: /^(\d{2})\.(\d{2})\.(\d{4})$/,
    read: (parts: string[]) => `${parts[3]}-${parts[2]}-${parts[1]}`,
  },
  {
    match: /^([A-Z][a-z]+) (\d{1,2}), (\d{4})$/,
    read: (parts: string[]) =>
      `${parts[3]}-${monthNumber(parts[1] ?? "")}-${(parts[2] ?? "").padStart(2, "0")}`,
  },
] as const;

const ISO = /^\d{4}-\d{2}-\d{2}$/;
const ANCHOR_LIMIT = 3;

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

// A quarter of the rows already arrive in ISO. Those are the only ones the
// system gets right unprompted, so they are the only ones a harvest can keep —
// and they are enough to teach it what to write for the rest.
const TRAIN: DateRow[] = [
  { raw: "2024-03-05", iso: "2024-03-05" },
  { raw: "03/05/2024", iso: "2024-03-05" },
  { raw: "05.03.2024", iso: "2024-03-05" },
  { raw: "March 5, 2024", iso: "2024-03-05" },
  { raw: "2023-11-30", iso: "2023-11-30" },
  { raw: "11/30/2023", iso: "2023-11-30" },
  { raw: "30.11.2023", iso: "2023-11-30" },
  { raw: "November 30, 2023", iso: "2023-11-30" },
];

const VALIDATION: DateRow[] = [
  { raw: "2022-07-04", iso: "2022-07-04" },
  { raw: "09/16/2021", iso: "2021-09-16" },
  { raw: "22.01.2020", iso: "2020-01-22" },
  { raw: "July 4, 2022", iso: "2022-07-04" },
];

/**
 * The system under optimization. It reads the demos out of its own candidate —
 * exactly as a prompted model would read the few-shot block it was given — and
 * takes its output convention from them. Shown nothing, it echoes the input.
 */
const adapter: Adapter<DateRow, unknown, string> = {
  evaluate: ({ batch, candidate }) => {
    const demos = parseDemos(candidate.demos ?? "");
    const writesIso = demos.some((demo) => ISO.test(String(demo.output)));
    // Past a few examples it stops generalizing and starts copying the block it
    // was given — the reason more demos is not monotonically better, and the
    // reason this search is over which demos rather than how many it can fit.
    const anchored = demos.length > ANCHOR_LIMIT;

    const outputs = batch.map((row) => {
      const format = FORMATS.find((format) => format.match.test(row.raw));
      const parts = format?.match.exec(row.raw);

      if (anchored) {
        return String(demos[0]?.output ?? row.raw);
      }
      return !writesIso || format === undefined || parts == null
        ? row.raw
        : format.read(parts);
    });

    return {
      outputs,
      scores: outputs.map((output, index) =>
        output === batch[index]?.iso ? 1 : 0,
      ),
      feedback: outputs.map((output, index) =>
        output === batch[index]?.iso
          ? "Normalized."
          : `Left "${batch[index]?.raw}" as "${output}"; expected "${batch[index]?.iso}".`,
      ),
    };
  },
};

const result = await new BootstrapSearchOptimizer({
  // Shuffled harvests beyond the fixed candidates every run tries. The default
  // is DSPy's 16; this task is small enough that 8 covers it.
  candidates: 8,
  maxDemos: 3,
  // A demo is an assertion that this is what good looks like, so only rollouts
  // the metric actually rewarded are kept.
  demoMinScore: 1,
  seed: 3,
}).optimize({
  seedCandidate: { demos: "" },
  trainingSet: TRAIN,
  validationSet: VALIDATION,
  adapter,
  demoComponents: ["demos"],
  // With labels, the run also tries a candidate built from gold outputs alone.
  // It costs no rollouts to build, and it is the only candidate available at
  // all to a system too weak to bootstrap anything.
  goldOutput: (row) => row.iso,
  maxMetricCalls: 300,
  reporters: [
    {
      onEvent: (event) => {
        if (event.type === "candidate") {
          console.log(
            `  ${event.source.padEnd(12)} ${event.demos} demos` +
              ` scored ${event.score.toFixed(3)}${event.accepted ? "  <- new best" : ""}`,
          );
        }
      },
    },
  ],
});

console.log(
  `\nseed ${result.seedScore.toFixed(3)} -> best ${result.bestScore.toFixed(3)}` +
    ` over ${result.candidates.length} candidates` +
    ` (${result.metricCalls} metric calls, stopped because ${result.stopReason})`,
);
console.log(`\nwinning demo block:\n${result.bestCandidate.demos}`);

function monthNumber(name: string): string {
  const index = MONTHS.indexOf(name);
  return index < 0 ? "??" : String(index + 1).padStart(2, "0");
}
