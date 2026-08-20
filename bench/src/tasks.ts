import type { Candidate, Demo, ScoreResult, TextModel } from "textopt";
import type { GepaAdapter } from "textopt/gepa";

export type Tier = "free" | "pro" | "enterprise";
export type Channel = "email" | "chat" | "phone";
export type Issue = "billing" | "outage" | "howto" | "bug";
export type Region = "us" | "eu";
export type Feature = "tier" | "channel" | "issue" | "region";

/**
 * One support ticket. The features are the whole input: a candidate never sees
 * an id, so there is nothing instance-shaped to memorise and the only thing
 * that transfers to a held-out ticket is a rule about the features.
 */
export interface BenchDatum {
  id: number;
  tier: Tier;
  channel: Channel;
  issue: Issue;
  region: Region;
}

export interface BenchTask {
  name: string;
  seedCandidate: Candidate;
  trainingSet: BenchDatum[];
  validationSet: BenchDatum[];
  testSet: BenchDatum[];
  adapter: GepaAdapter<BenchDatum, string, string>;
  /**
   * Which features each component is allowed to write rules about. A rule in
   * the wrong component does not fire, which is what makes a two-component
   * task need both components right rather than one component twice.
   */
  componentScope: Record<string, Feature[]>;
  /** Components a demonstration search harvests into. */
  demoComponents: string[];
  /** Rollouts each optimizer is given. The same for all of them, by design. */
  maxMetricCalls: number;
  /** Reflection calls, which no metric budget covers. */
  maxReflectionCalls: number;
}

interface PolicyRule {
  feature: Feature;
  value: string;
  action: string;
}

interface Observation {
  tier: string;
  channel: string;
  issue: string;
  region: string;
  missing: string[];
  unnecessary: string[];
}

const TIERS: Tier[] = ["free", "pro", "enterprise"];
const CHANNELS: Channel[] = ["email", "chat", "phone"];
const ISSUES: Issue[] = ["billing", "outage", "howto", "bug"];

/**
 * A feature of every ticket that no policy rule ever keys on.
 *
 * Induction here is counting: a proposer believes a rule once it has seen a
 * feature value and a missing action together often enough. On data balanced
 * enough for that count to be a good estimator, every rule the policy holds is
 * recovered on sight and the search degenerates into transcription — which
 * measures a budget, not a search. A feature carried on the ticket, named in
 * the feedback and expressible in the grammar, but absent from the policy, is
 * what a coincidence looks like: it competes for the same evidence, wins
 * whenever it happens to co-occur as often as the real condition, and costs a
 * search the rollouts to find out. Real inputs are full of these.
 */
const REGIONS: Region[] = ["us", "eu"];

/**
 * What a correct answer has to do, as a function of the ticket. This is the
 * ground truth a run is searching for and no optimizer is ever shown it — the
 * metric reports which actions an answer missed on one ticket, and which
 * feature explains a miss is what has to be induced across tickets.
 *
 * `free` and `email` carry no obligation of their own, so instances differ in
 * how many actions they need rather than only in which.
 */
const POLICY: PolicyRule[] = [
  { feature: "issue", value: "outage", action: "status page" },
  { feature: "issue", value: "outage", action: "escalate" },
  { feature: "issue", value: "billing", action: "prorated refund" },
  { feature: "issue", value: "howto", action: "docs link" },
  { feature: "issue", value: "bug", action: "repro steps" },
  { feature: "tier", value: "enterprise", action: "sla" },
  { feature: "tier", value: "pro", action: "priority queue" },
  { feature: "channel", value: "phone", action: "callback" },
  { feature: "channel", value: "chat", action: "transcript" },
];

/** Every action the policy can ask for, and the widest an answer can be. */
export const ACTIONS: readonly string[] = [
  ...new Set(POLICY.map((rule) => rule.action)),
];

/**
 * What a proposer draws from when it has nothing to induce from: every rule the
 * language can express, ordered by a hash so the pool is walked in no
 * particular relation to which of them are right.
 *
 * Every expressible rule, and not a shortlist built from `POLICY`, because a
 * shortlist containing the answers is not a blind draw — it is the answer key
 * with extra steps. A pool half-made of correct rules turns speculation into
 * enumeration, and any search that can keep an improvement and discard a
 * regression then climbs it without learning anything. Nine of these ninety are
 * right, so drawing is a real gamble and evidence is worth having.
 */
const DRAW_POOL: PolicyRule[] = (
  [
    ["tier", TIERS],
    ["channel", CHANNELS],
    ["issue", ISSUES],
    ["region", REGIONS],
  ] as [Feature, readonly string[]][]
)
  .flatMap(([feature, values]) =>
    values.flatMap((value) =>
      ACTIONS.map((action) => ({ feature, value, action })),
    ),
  )
  .sort((left, right) => hash32(renderRule(left)) - hash32(renderRule(right)));

/**
 * Score lost per action an answer offers that the ticket did not call for.
 *
 * Set so that answering every ticket with every action — the cheapest thing a
 * candidate can do that is not nothing — scores well below a candidate that
 * found the rules. A penalty light enough to leave that strategy competitive
 * would make the benchmark a measure of how fast a search can bloat.
 */
const BLOAT_PENALTY = 0.1;

/**
 * Instances a condition must be seen across before a proposer will believe it.
 * One ticket is consistent with three explanations — its tier, its channel and
 * its issue — so induction from a single observation is guessing, and a
 * proposer that guessed would hand the search an answer it had not earned.
 */
const MIN_EVIDENCE = 2;

/**
 * The largest reflection batch the benchmark sweeps.
 *
 * A ceiling on the stand-in proposer, not on any optimizer. Induction here is
 * exact counting, so its accuracy is a function of how many observations one
 * call is shown: past roughly a dozen the count stops being an estimate and
 * every rule is recovered on sight, at which point tuning selects whichever
 * entrant asks for the biggest batch and the table ranks reflection-prompt size
 * rather than search. A real proposal model does not sharpen like that, so
 * letting the grid run past the point where this one does would measure an
 * artefact of the stand-in. Below it the proposer is wrong often enough that
 * proposals have to be evaluated and rejected, which is the thing being
 * compared.
 */
export const MAX_MINIBATCH = 9;

const WILDCARD = "any";
const RULE_PATTERN = /when (tier|channel|issue|region) is ([a-z ]+):([^\n]*)/g;
const DEMO_PATTERN =
  /<input>\s*([\s\S]*?)\s*<\/input>\s*<output>\s*([\s\S]*?)\s*<\/output>/g;
const OBSERVATION_PATTERN =
  /tier=(\w+) channel=(\w+) issue=(\w+) region=(\w+) \| missing: ([^|\n]*)(?:\| unnecessary: ([^|\n]*))?/g;

export function benchTasks(): BenchTask[] {
  return [clean(), noisy(), interacting(), demonstrated()];
}

/**
 * The candidate the search is looking for: every policy rule, written into the
 * component allowed to hold it. Published as the benchmark's ceiling, because a
 * table of scores with no ceiling row leaves a reader no way to tell a search
 * that nearly solved the task from one that barely started.
 */
export function policyCandidate(task: BenchTask): Candidate {
  const components = Object.keys(task.seedCandidate);

  return Object.fromEntries(
    components.map((component) => [
      component,
      POLICY.filter((rule) =>
        (task.componentScope[component] ?? []).includes(rule.feature),
      )
        .map(renderRule)
        .join("\n"),
    ]),
  );
}

/**
 * Every action on every ticket, written into each component against a feature
 * that component actually governs. Published as the benchmark's zero-search
 * floor: a search that scores under it found nothing a candidate written
 * without one would not have found too.
 *
 * The feature has to be chosen per component rather than fixed. A rule naming a
 * feature its component does not govern is dropped, so one text written
 * everywhere scores zero on a task whose components govern different features —
 * not because spraying earns nothing there, but because it never ran, and a
 * floor that low flatters every entrant measured against it.
 */
export function shotgunCandidate(task: BenchTask): Candidate {
  return spray({ task, actions: ACTIONS });
}

/**
 * The strongest answer that ignores the ticket: the subset of actions that
 * scores best when emitted on every one of them, chosen over the training and
 * validation sets and never over the held-out set it is reported on.
 *
 * A floor selected on the number it is published at is an oracle, not a floor.
 * Chosen honestly it is the harder question the shotgun row only gestures at —
 * the shotgun sprays all nine actions and eats the bloat penalty for the ones
 * that miss, while this one knows the distribution well enough to stop. It is
 * the score to beat before a search can be said to have learned a rule, as
 * distinct from having learned which actions are common.
 */
export function bestUnconditionalCandidate(task: BenchTask): Candidate {
  const selectOn = [...task.trainingSet, ...task.validationSet];

  let best: { candidate: Candidate; score: number } | undefined;
  for (let subset = 1; subset < 2 ** ACTIONS.length; subset += 1) {
    const actions = ACTIONS.filter((_, index) => (subset >> index) & 1);
    const candidate = spray({ task, actions });
    const { scores } = task.adapter.evaluate({
      batch: selectOn,
      candidate,
      captureTraces: false,
      run: {
        iteration: 0,
        phase: "validation",
        split: "val",
        candidateId: null,
      },
    }) as { scores: number[] };
    const score =
      scores.reduce((total, value) => total + value, 0) / scores.length;

    if (best === undefined || score > best.score) {
      best = { candidate, score };
    }
  }

  return (best as { candidate: Candidate }).candidate;
}

/**
 * The prompt with the metric's per-instance diagnosis removed, leaving a
 * proposer the blind draw a score-only search already gets.
 *
 * Built from `OBSERVATION_PATTERN` rather than spelled out a second time, so it
 * cannot fall behind the grammar it strips. A redaction that quietly stops
 * matching does not fail loudly — it publishes the unredacted search under the
 * redacted row's name, which is the one thing this row exists to rule out.
 * What a component governs is left in place: that is not evidence about any
 * ticket, and taking it too would handicap the row twice over.
 */
export function redactObservations(prompt: string): string {
  return prompt.replace(new RegExp(OBSERVATION_PATTERN.source, "g"), "");
}

/** What the policy asks of one ticket. Ground truth, for scoring and floors. */
export function requiredActions(datum: BenchDatum): string[] {
  return [
    ...new Set(
      POLICY.filter((rule) => datum[rule.feature] === rule.value).map(
        (rule) => rule.action,
      ),
    ),
  ];
}

/**
 * A stand-in for the proposal model, which reads whichever evidence the prompt
 * carries — per-instance feedback, a score history, or neither.
 *
 * What it will not do is copy an answer out of the prompt. Feedback names the
 * actions one ticket missed; turning that into a rule means finding the feature
 * those tickets share, which takes several of them agreeing. Under `MIN_EVIDENCE`
 * observations it declines and speculates instead, so an informative prompt is
 * an advantage that still has to be worked for rather than an oracle.
 */
export function createBenchReflector(
  args: { absorb?: number; seed?: number } = {},
): TextModel {
  const { absorb = 2, seed = 0 } = args;
  let cursor = 0;

  return async ({ prompt }) => {
    const base = bestAttempt(prompt) ?? currentText(prompt);
    const rules = induce({ prompt, absorb, base });

    if (rules.length > 0) {
      return fence([base, ...rules.map(renderRule)].join("\n").trim());
    }

    const narrowed = narrow({ prompt, base });
    if (narrowed !== undefined) {
      return fence(narrowed);
    }

    const drawn = draw({ seed, cursor });
    cursor += 1;
    return fence([base, renderRule(drawn)].join("\n").trim());
  };
}

/**
 * The same proposer in the shape SIMBA asks for: per-component advice rather
 * than a rewritten instruction. It induces from the same evidence under the
 * same threshold and speculates from the same pool.
 *
 * One thing the reflector does is missing here on purpose, because SIMBA cannot
 * use it: there is no narrowing path. `appendRule` appends advice to a
 * component and never rewrites it, so a rule this proposer regrets is one the
 * search is stuck with. That follows the algorithm rather than handicapping it,
 * and it is why advice accumulates here — the cost of an append-only mutation
 * is that every wrong rule is permanent and goes on paying the bloat penalty.
 */
export function createBenchAdviser(
  args: { absorb?: number; seed?: number } = {},
): TextModel {
  const { absorb = 2, seed = 0 } = args;
  let cursor = 0;

  return async ({ prompt }) => {
    const components = adviceComponents(prompt);
    const induced = induce({ prompt, absorb, base: adviceBase(prompt) });

    const rules = induced.length > 0 ? induced : [draw({ seed, cursor })];
    cursor += 1;

    return components
      .map(
        (component) =>
          `<advice component="${component}">${rules.map(renderRule).join(" ")}</advice>`,
      )
      .join("\n");
  };
}

/**
 * Renders a harvested rollout as the ticket it ran on and the answer it
 * produced. The ticket's features are its input; there is no label to leak,
 * because what made the answer worth keeping is that the metric rewarded it.
 */
export function renderBenchDemo(args: {
  demo: Demo<BenchDatum, string>;
  index: number;
}): string {
  const { demo } = args;
  return `<input>\n${describe(demo.input)}\n</input>\n<output>\n${demo.output}\n</output>`;
}

/** A noiseless metric with a clean gradient: the reference case. */
function clean(): BenchTask {
  return {
    name: "clean",
    ...singleComponent(),
    adapter: policyAdapter({
      noise: 0,
      scope: singleComponent().componentScope,
    }),
    maxMetricCalls: 400,
    maxReflectionCalls: 40,
  };
}

/**
 * The same task, measured with instance-level noise. Real metrics are noisy —
 * a judge model, a sampled generation — and noise is what separates an
 * acceptance rule that survives contact with a real system from one tuned on
 * a deterministic fixture.
 */
function noisy(): BenchTask {
  return {
    name: "noisy",
    ...singleComponent(),
    adapter: policyAdapter({
      noise: 0.2,
      scope: singleComponent().componentScope,
    }),
    maxMetricCalls: 400,
    maxReflectionCalls: 40,
  };
}

/**
 * Two components in a pipeline, where the second is worthless until the first
 * is right. `triage` holds the rules about the issue, `response` those about
 * the tier and channel, and the system applies `response` only to tickets
 * `triage` already handled correctly — the way a real second stage never sees a
 * ticket the first stage misrouted.
 *
 * That is the case joint search exists for and per-component search cannot see:
 * every improvement to `response` scores exactly zero until `triage` is
 * complete, so a search crediting components independently spends its budget on
 * a component that cannot pay yet. Unlike scoring the whole ticket all or
 * nothing, it leaves a gradient — `triage` pays on its own — so the task
 * measures where a search spends rather than whether it got lucky.
 */
function interacting(): BenchTask {
  const componentScope: Record<string, Feature[]> = {
    triage: ["issue", "region"],
    response: ["tier", "channel", "region"],
  };

  return {
    name: "interacting",
    seedCandidate: { triage: "", response: "" },
    ...splits(),
    componentScope,
    demoComponents: ["triage", "response"],
    adapter: policyAdapter({
      noise: 0,
      scope: componentScope,
      gate: { gated: "response", requires: "triage" },
    }),
    maxMetricCalls: 500,
    maxReflectionCalls: 40,
  };
}

/**
 * The same metric again, over a system that is sometimes right on its own.
 *
 * Every task above answers a ticket only from the rules the candidate holds, so
 * a rollout tells the search nothing its prompt did not, and a demonstration
 * search harvesting those rollouts can only hand a candidate its own words
 * back. Real systems are not like that: a model answers one ticket correctly
 * and the next one wrong, and the examples worth showing it are the ones it
 * already got right.
 */
function demonstrated(): BenchTask {
  return {
    name: "demonstrated",
    ...singleComponent(),
    adapter: policyAdapter({
      noise: 0,
      reliability: 0.4,
      scope: singleComponent().componentScope,
    }),
    maxMetricCalls: 400,
    maxReflectionCalls: 40,
  };
}

function singleComponent(): Pick<
  BenchTask,
  | "seedCandidate"
  | "componentScope"
  | "demoComponents"
  | "trainingSet"
  | "validationSet"
  | "testSet"
> {
  return {
    seedCandidate: { instruction: "" },
    componentScope: { instruction: ["tier", "channel", "issue", "region"] },
    demoComponents: ["instruction"],
    ...splits(),
  };
}

/**
 * Every combination of the three features, dealt into three balanced splits.
 *
 * The deal is what makes the held-out score mean something. Each split holds
 * twelve combinations none of the others contain, while every individual
 * feature value appears in all three — so every rule is learnable from the
 * training set and no ticket in the test set has been seen. A candidate that
 * fitted the combinations it was shown has nothing to fall back on; one that
 * found the rules scores exactly as well on tickets it never saw.
 *
 * The deal is `(tier + channel + issue) % 3`, which buys two things a simpler
 * one does not. Dealing every third combination of the enumeration left each
 * split double-weighting a different channel — training on email, selecting on
 * chat, scoring on phone — and every search here selects on the validation set
 * and is reported on the test set, so a held-out score became partly a lottery
 * on which rule the split it was measured against happened to favour.
 *
 * All three indices have to appear in the sum. Dealing on `(tier + channel)`
 * balances the marginals but pairs each tier with exactly one channel inside a
 * split, and two features in bijection make their rules indistinguishable: a
 * search cannot tell `tier is pro` from `channel is phone` when the training
 * data never separates them, so it learns the wrong one, converges there from
 * every seed, and the ceiling stops being reachable by any method. Adding the
 * issue index breaks the pairing — every tier meets every channel within a
 * split — while leaving the marginals even in each of the three.
 *
 * Twenty-four tickets per split, rather than the twelve the three real features
 * alone would give, is also what keeps a single reflection call from seeing the
 * whole training set: shown all of it at once the proposer recovers the policy
 * outright, and a benchmark whose winner saturates the ceiling from every seed
 * has stopped measuring anything.
 */
function splits(): Pick<
  BenchTask,
  "trainingSet" | "validationSet" | "testSet"
> {
  const combinations = TIERS.flatMap((tier, tierIndex) =>
    CHANNELS.flatMap((channel, channelIndex) =>
      ISSUES.flatMap((issue, issueIndex) =>
        REGIONS.map((region, regionIndex) => ({
          tier,
          channel,
          issue,
          region,
          split: (tierIndex + channelIndex + issueIndex + regionIndex) % 3,
        })),
      ),
    ),
  );
  const deal = (remainder: number): BenchDatum[] =>
    combinations
      .map((features, index) => ({ id: index, ...features }))
      .filter((entry) => entry.split === remainder)
      .map(({ split, ...datum }) => datum);

  return {
    trainingSet: deal(0),
    validationSet: deal(1),
    testSet: deal(2),
  };
}

/**
 * Runs a candidate against a ticket and scores what came back.
 *
 * The answer is what the system emitted, not what the candidate said: rules
 * fire only on the tickets their condition matches, demonstrations carry to
 * tickets they resemble, and an unreliable system sometimes answers correctly
 * with no help at all. That indirection is the point — it is what a rollout can
 * tell a search that reading the candidate could not.
 */
function policyAdapter(args: {
  noise: number;
  scope: Record<string, Feature[]>;
  /**
   * A component the system applies only to tickets another component already
   * handled correctly. Absent on the single-component tasks, where there is
   * nothing upstream to depend on.
   */
  gate?: { gated: string; requires: string };
  /**
   * Share of tickets the system answers correctly with no help from its
   * prompt. Zero — the default, and the case the other tasks model — makes the
   * answer a pure function of the candidate.
   */
  reliability?: number;
}): GepaAdapter<BenchDatum, string, string> {
  const { noise, scope, gate, reliability = 0 } = args;

  return {
    evaluate: ({ batch, candidate }) => {
      const scored = batch.map((datum) => score({ datum, candidate }));

      return {
        outputs: scored.map((entry) => entry.answer),
        scores: scored.map((entry) => entry.score),
        feedback: scored.map((entry) => entry.feedback ?? ""),
        trajectories: scored.map((entry) => entry.answer),
      };
    },

    makeReflectiveDataset: ({ batch, evaluation, componentsToUpdate }) => {
      return Object.fromEntries(
        componentsToUpdate.map((component) => [
          component,
          batch.map((datum, index) => ({
            inputs: describe(datum),
            generatedOutputs: evaluation.outputs[index] ?? "",
            // The component's remit rides along with the diagnosis: a proposer
            // that cannot see which features this component governs writes
            // rules into the one that will not fire them.
            feedback: `${evaluation.feedback?.[index] ?? ""} | governs: ${(scope[component] ?? []).join(", ")}`,
            score: evaluation.scores[index],
          })),
        ]),
      );
    },
  };

  function score(args: {
    datum: BenchDatum;
    candidate: Candidate;
  }): ScoreResult & {
    answer: string;
  } {
    const { datum, candidate } = args;
    const emitted = answer({ datum, candidate });
    const required = requiredActions(datum);

    const covered = required.filter((action) => emitted.has(action));
    const missing = required.filter((action) => !emitted.has(action));
    const unnecessary = [...emitted].filter(
      (action) => !required.includes(action),
    );

    const base = covered.length / required.length;
    const text = [...emitted].sort().join(" ");

    return {
      score: clamp(
        base -
          unnecessary.length * BLOAT_PENALTY +
          jitter({ answer: text, datum, noise }),
      ),
      answer: text,
      feedback: [
        `${describe(datum)} | missing: ${missing.join(", ")}`,
        unnecessary.length === 0
          ? ""
          : ` | unnecessary: ${unnecessary.join(", ")}`,
      ].join(""),
    };
  }

  /** What the system emits for one ticket: rules, demonstrations, recall. */
  function answer(args: {
    datum: BenchDatum;
    candidate: Candidate;
  }): Set<string> {
    const { datum, candidate } = args;
    const byComponent = new Map<string, Set<string>>();

    for (const [component, text] of Object.entries(candidate)) {
      const produced = new Set<string>();
      for (const rule of parseRules({
        text,
        allowed: scope[component] ?? [],
      })) {
        if (matches({ rule, datum })) {
          produced.add(rule.action);
        }
      }
      for (const action of demonstratedActions({ text, datum })) {
        produced.add(action);
      }
      byComponent.set(component, produced);
    }

    const emitted = new Set<string>();
    for (const [component, produced] of byComponent) {
      if (
        component === gate?.gated &&
        !upstreamHandled({ datum, byComponent })
      ) {
        continue;
      }
      for (const action of produced) {
        emitted.add(action);
      }
    }

    if (recalls({ id: datum.id, reliability })) {
      for (const action of requiredActions(datum)) {
        emitted.add(action);
      }
    }
    return emitted;
  }

  /**
   * Whether the component a gated one depends on got this ticket right. Right
   * means every action the upstream component was responsible for, since a
   * ticket half-triaged is one the next stage would have handled as the wrong
   * kind of ticket.
   */
  function upstreamHandled(args: {
    datum: BenchDatum;
    byComponent: Map<string, Set<string>>;
  }): boolean {
    const { datum, byComponent } = args;
    const upstream = byComponent.get(gate?.requires ?? "") ?? new Set<string>();
    const owed = requiredActions(datum).filter((action) =>
      POLICY.some(
        (rule) =>
          rule.action === action &&
          (scope[gate?.requires ?? ""] ?? []).includes(rule.feature),
      ),
    );

    return owed.every((action) => upstream.has(action));
  }
}

/**
 * Actions a ticket's neighbours in the candidate's demonstrations suggest.
 *
 * A demonstration is read by analogy rather than looked up: it carries to a
 * ticket agreeing with it on two of three features. That is why a demo is
 * worth something on a split whose combinations it never contains, and why
 * showing too many costs precision — an analogy drawn from a ticket that
 * differed in the feature that mattered offers an action this one did not want.
 */
function demonstratedActions(args: {
  text: string;
  datum: BenchDatum;
}): string[] {
  const { text, datum } = args;
  const actions: string[] = [];

  for (const [, input, output] of text.matchAll(DEMO_PATTERN)) {
    const shared = (["tier", "channel", "issue"] as Feature[]).filter(
      (feature) => (input ?? "").includes(`${feature}=${datum[feature]}`),
    );
    if (shared.length >= 2) {
      actions.push(...mentionedActions(output ?? ""));
    }
  }
  return actions;
}

/**
 * Rules a component holds, keeping only the ones it is allowed to hold. An
 * optimizer is free to write any rule anywhere; a rule outside the component's
 * remit is simply text the system does not act on.
 */
function parseRules(args: { text: string; allowed: Feature[] }): PolicyRule[] {
  const { text, allowed } = args;
  const rules: PolicyRule[] = [];

  for (const [, feature, value, actions] of text.matchAll(RULE_PATTERN)) {
    const named = feature as Feature;
    if (!allowed.includes(named)) {
      continue;
    }
    for (const action of mentionedActions(actions ?? "")) {
      rules.push({ feature: named, value: (value ?? "").trim(), action });
    }
  }
  return rules;
}

/** `actions` on every ticket, in whichever component is allowed to fire them. */
function spray(args: {
  task: BenchTask;
  actions: readonly string[];
}): Candidate {
  const { task, actions } = args;

  return Object.fromEntries(
    Object.keys(task.seedCandidate).map((component) => {
      const [governed] = task.componentScope[component] ?? [];
      return [
        component,
        `when ${governed} is ${WILDCARD}: ${actions.join(", ")}`,
      ];
    }),
  );
}

/**
 * The rule a proposer speculates with on its nth call of a run.
 *
 * Keyed on the run's seed as well as the call, so two runs walk different parts
 * of the pool. Walking it in one fixed order instead made a blind score a
 * function of nothing but how many proposals an entrant's budget bought: the
 * first eleven entries of the shared order hold none of the nine correct rules
 * and the first forty hold five, so an entrant that proposes often out-scored
 * one that proposes rarely without either of them searching any better. Which
 * stretch of the pool a run happens to walk is luck, and luck is what a blind
 * draw is supposed to be measuring.
 */
function draw(args: { seed: number; cursor: number }): PolicyRule {
  const { seed, cursor } = args;
  const index = hash32(`draw:${seed}:${cursor}`) % DRAW_POOL.length;

  return DRAW_POOL[index] as PolicyRule;
}

function matches(args: { rule: PolicyRule; datum: BenchDatum }): boolean {
  const { rule, datum } = args;
  return rule.value === WILDCARD || datum[rule.feature] === rule.value;
}

/** Actions named in a fragment, matched by name so punctuation cannot hide one. */
function mentionedActions(text: string): string[] {
  const lowered = text.toLowerCase();
  return ACTIONS.filter((action) => lowered.includes(action));
}

/**
 * The rule the prompt's observations support, if they support one.
 *
 * Counts how often each (feature value, missing action) pair co-occurs and
 * keeps the pairs seen at least `MIN_EVIDENCE` times. A missing action explained
 * by a feature the component does not govern is dropped, and a rule the
 * candidate already holds is not proposed twice.
 */
function induce(args: {
  prompt: string;
  absorb: number;
  base: string;
}): PolicyRule[] {
  const { prompt, absorb, base } = args;
  const allowed = governedFeatures(prompt);
  const counts = new Map<string, number>();

  for (const observation of readObservations(prompt)) {
    for (const action of observation.missing) {
      for (const feature of allowed) {
        const value = observation[feature];
        counts.set(
          `${feature}|${value}|${action}`,
          (counts.get(`${feature}|${value}|${action}`) ?? 0) + 1,
        );
      }
    }
  }

  return [...counts.entries()]
    .filter(([, count]) => count >= MIN_EVIDENCE)
    .sort(([left, leftCount], [right, rightCount]) =>
      leftCount === rightCount
        ? left.localeCompare(right)
        : rightCount - leftCount,
    )
    .map(([key]) => {
      const [feature, value, action] = key.split("|");
      return {
        feature: feature as Feature,
        value: value as string,
        action: action as string,
      };
    })
    .filter((rule) => !base.includes(renderRule(rule)))
    .slice(0, absorb);
}

/**
 * The candidate with its worst rule removed, when the prompt says an answer
 * offered actions the ticket did not want and names nothing missing. Without
 * this a speculative rule is permanent, and a search that could only add would
 * ratchet itself down.
 */
function narrow(args: { prompt: string; base: string }): string | undefined {
  const { prompt, base } = args;
  const observations = readObservations(prompt);
  const offending = new Set(
    observations.flatMap((observation) => observation.unnecessary),
  );

  if (offending.size === 0 || observations.some((o) => o.missing.length > 0)) {
    return undefined;
  }

  const lines = base.split("\n");
  const index = lines.findIndex((line) =>
    [...offending].some((action) => line.includes(action)),
  );
  return index === -1
    ? undefined
    : lines.filter((_, position) => position !== index).join("\n");
}

/** The metric's per-instance diagnoses, as the prompt happens to carry them. */
function readObservations(prompt: string): Observation[] {
  return [...prompt.matchAll(OBSERVATION_PATTERN)].map(
    ([, tier, channel, issue, region, missing, unnecessary]) => ({
      tier: tier as string,
      channel: channel as string,
      issue: issue as string,
      region: region as string,
      missing: mentionedActions(missing ?? ""),
      unnecessary: mentionedActions(unnecessary ?? ""),
    }),
  );
}

/**
 * What the components already say, as the advice prompt reports it. Induction
 * is filtered against this so a proposer does not spend a call re-proposing a
 * rule the candidate is already carrying — which, under an append-only
 * mutation, would be a duplicate that costs bloat and buys nothing.
 */
function adviceBase(prompt: string): string {
  return [
    ...prompt.matchAll(/<component name="[^"]+">([\s\S]*?)<\/component>/g),
  ]
    .map((match) => match[1] ?? "")
    .join("\n");
}

/** Features the component under revision may write rules about. */
function governedFeatures(prompt: string): Feature[] {
  const match = prompt.match(/governs: ([a-z, ]+)/);
  const named = (match?.[1] ?? "")
    .split(",")
    .map((feature) => feature.trim())
    .filter(Boolean) as Feature[];

  return named.length > 0 ? named : ["tier", "channel", "issue", "region"];
}

function renderRule(rule: PolicyRule): string {
  return `when ${rule.feature} is ${rule.value}: ${rule.action}`;
}

function describe(datum: BenchDatum): string {
  return `tier=${datum.tier} channel=${datum.channel} issue=${datum.issue} region=${datum.region}`;
}

/**
 * Whether the system answers this ticket right without being shown how.
 *
 * A function of the ticket alone, so a run is reproducible and two tickets
 * needing the same action can differ — which is the whole mechanism: the one it
 * got right is harvestable, and showing it makes the one it got wrong right too.
 */
function recalls(args: { id: number; reliability: number }): boolean {
  const { id, reliability } = args;
  if (reliability === 0) {
    return false;
  }
  return hash32(`recall:${id}`) % 1000 < reliability * 1000;
}

/**
 * Measurement noise that is a function of what was measured, so a run is
 * reproducible and a re-measurement of the same answer on the same ticket
 * repeats — which is what a cached score assumes.
 */
function jitter(args: {
  answer: string;
  datum: BenchDatum;
  noise: number;
}): number {
  const { answer, datum, noise } = args;
  if (noise === 0) {
    return 0;
  }

  const hash = hash32(`${datum.id}:${answer}`);
  return ((hash % 2001) / 1000 - 1) * noise;
}

function hash32(value: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function clamp(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function fence(text: string): string {
  return `\`\`\`\n${text}\n\`\`\``;
}

function currentText(prompt: string): string {
  const match = prompt.match(
    /<current_instruction>\n([\s\S]*?)\n<\/current_instruction>/,
  );
  return match?.[1]?.trim() ?? "";
}

/** The highest-scoring attempt a score-history prompt lists, when it lists any. */
function bestAttempt(prompt: string): string | undefined {
  let bestScore = Number.NEGATIVE_INFINITY;
  let best: string | undefined;

  for (const match of prompt.matchAll(
    /score:\s*([\d.]+)[\s\S]*?<instruction>\n?([\s\S]*?)\n?<\/instruction>/g,
  )) {
    const value = Number(match[1]);
    if (value > bestScore) {
      bestScore = value;
      best = (match[2] ?? "").trim();
    }
  }
  return best;
}

function adviceComponents(prompt: string): string[] {
  return [...prompt.matchAll(/<component name="([^"]+)">/g)].map(
    (match) => match[1] as string,
  );
}
