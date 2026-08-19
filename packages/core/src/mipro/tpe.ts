import type { Rng } from "../rng.js";

/** One configuration that was scored: an index into each component's menu. */
export interface Observation {
  choices: readonly number[];
  score: number;
}

const DEFAULT_SAMPLES = 24;
const GOOD_FRACTION = 0.1;
const MAX_GOOD = 25;
const DEFAULT_STARTUP = 10;
const DEFAULT_PRIOR_WEIGHT = 1;

/**
 * Tree-structured Parzen Estimator over a categorical space.
 *
 * The search space here is a menu index per component. A TPE splits the
 * observations into the good ones and the rest, models the density of each,
 * and proposes the configuration that maximizes their ratio: sample where
 * good configurations live, prefer where bad ones do not.
 *
 * The densities are mixtures with one kernel centred on each observation,
 * rather than one histogram per component. That distinction is the whole
 * reason this optimizer exists. Per-component histograms only record how often
 * an option appears among good trials, so a space where every option is
 * equally common and only the *pairing* matters looks completely flat to them.
 * A kernel centred on an observed configuration keeps its components together,
 * so "B works, but only alongside A" survives into the proposal. It is what
 * Optuna calls a multivariate sampler, and what MIPROv2 turns on.
 *
 * What this buys over sampling every combination: a candidate with five
 * components and four options each has 1024 configurations and a budget for
 * perhaps thirty evaluations. Per-component hill climbing cannot see
 * interactions at all; enumerating cannot afford to. This can do both, at the
 * cost of being an estimate.
 */
export function proposeConfiguration(args: {
  observations: readonly Observation[];
  /** Menu length per component, positionally. */
  menuSizes: readonly number[];
  /**
   * Fraction of observations treated as good. Defaults to Optuna's rule — a
   * tenth of them, never more than 25 — which narrows the good set as the run
   * goes on. A fixed fraction keeps admitting weaker observations as evidence
   * accumulates, which is backwards: the more you have measured, the less a
   * middling result should count as something to aim at.
   */
  gamma?: number;
  /** Configurations drawn before the best is chosen. Default 24. */
  samples?: number;
  /** Observations required before the model is trusted at all. Default 10. */
  startupTrials?: number;
  /**
   * Weight of the uniform prior inside every kernel. Optuna's `prior_weight`;
   * 1 leaves an observed option twice as likely as one beside it.
   */
  priorWeight?: number;
  /**
   * Model components jointly rather than one at a time. Default true, as
   * MIPROv2 sets on Optuna.
   *
   * Joint modelling is the only way to represent "this option works, but only
   * beside that one", and it costs something where no such dependency exists.
   * Kernels sit on observed *combinations*, so evidence does not carry across
   * to combinations nobody has drawn; independent histograms generalize from
   * far fewer trials but cannot express a dependency at all.
   *
   * Which way that trades depends on how much of the space a run can cover.
   * Measured here on five components of five options — 3125 configurations
   * against 60 trials — joint reaches a mean best of 0.87 and solves 8 runs in
   * 15, where independent reaches 0.78 and solves 5. On a 16-configuration
   * space that 30 trials nearly enumerate, the ordering reverses and
   * independent lands on the best configuration in 16 runs of 20 against 14,
   * because there the surrogate is barely steering anything. Joint is the
   * default because the case it loses is the case that needed a surrogate
   * least.
   */
  multivariate?: boolean;
  rng: Rng;
}): number[] {
  const {
    observations,
    menuSizes,
    gamma,
    samples = DEFAULT_SAMPLES,
    startupTrials = DEFAULT_STARTUP,
    priorWeight = DEFAULT_PRIOR_WEIGHT,
    multivariate = true,
    rng,
  } = args;

  for (const size of menuSizes) {
    if (!Number.isInteger(size) || size < 1) {
      throw new Error(
        `every component needs a menu of at least one option, received ${size}`,
      );
    }
  }

  // Below the startup count the densities are noise fitted to a handful of
  // points, and trusting them collapses the search onto whatever the first
  // lucky draw contained.
  if (observations.length < startupTrials) {
    return menuSizes.map((size) => rng.nextInt(size));
  }

  // Splitting into good and bad presumes the scores rank the observations. If
  // they are all equal that presumption is false, and the split falls to
  // whichever configurations happen to sort first — which the batch of draws
  // below then locks onto, re-proposing measured configurations for the rest
  // of the run while leaving others untouched. A flat surface is exactly the
  // state a hard task starts in, so this is not a rare corner.
  //
  // Published TPE has no such guard, and neither does Optuna: both split at the
  // gamma quantile regardless and go on sampling from whatever density the tie
  // produced. This is a deliberate departure, not an oversight — leave it in.
  const scores = observations.map((observation) => observation.score);
  if (Math.max(...scores) === Math.min(...scores)) {
    return menuSizes.map((size) => rng.nextInt(size));
  }

  const ranked = [...observations].sort((a, b) => b.score - a.score);
  const requested =
    gamma === undefined
      ? Math.min(Math.ceil(GOOD_FRACTION * ranked.length), MAX_GOOD)
      : Math.floor(gamma * ranked.length);
  const goodCount = Math.min(ranked.length - 1, Math.max(1, requested));
  const good = ranked.slice(0, goodCount);
  const bad = ranked.slice(goodCount);

  let best: number[] | undefined;
  let bestScore = Number.NEGATIVE_INFINITY;

  const goodModel = fit({
    observations: good,
    menuSizes,
    priorWeight,
    multivariate,
  });
  const badModel = fit({
    observations: bad,
    menuSizes,
    priorWeight,
    multivariate,
  });

  for (let sample = 0; sample < samples; sample += 1) {
    const choices = goodModel.sample(rng);
    const score =
      Math.log(goodModel.density(choices)) -
      Math.log(badModel.density(choices));

    if (score > bestScore) {
      bestScore = score;
      best = choices;
    }
  }

  // A configuration already tried is allowed back. Each trial scores it on a
  // fresh minibatch, so a repeat is a second reading of a noisy measurement
  // rather than a wasted one — and the instances it shares with the first
  // reading come out of the cache free.
  return best as number[];
}

interface Density {
  density(choices: readonly number[]): number;
  sample(rng: Rng): number[];
}

/**
 * Builds the density over a set of observations.
 *
 * Joint: a mixture with one kernel centred on each observation, plus a uniform
 * kernel. Because a kernel keeps an observation's components together, the
 * mixture assigns high density to combinations that were seen together — which
 * is what carries a dependency between components.
 *
 * Independent: one smoothed histogram per component, multiplied. Evidence
 * about a component generalizes across every combination it appears in, which
 * converges far faster when the components really are independent, and cannot
 * express a dependency at all.
 *
 * Both keep a uniform component — Optuna's `consider_prior` — so an option
 * nobody has drawn never falls to zero probability and stays reachable.
 */
function fit(args: {
  observations: readonly Observation[];
  menuSizes: readonly number[];
  priorWeight: number;
  multivariate: boolean;
}): Density {
  const { observations, menuSizes, priorWeight, multivariate } = args;

  if (multivariate) {
    return {
      density: (choices) => {
        let total = uniformKernel(menuSizes);
        for (const observation of observations) {
          total += kernel({
            choices,
            center: observation.choices,
            menuSizes,
            priorWeight,
          });
        }
        return total / (observations.length + 1);
      },
      sample: (rng) => {
        const picked = rng.nextInt(observations.length + 1);
        if (picked === observations.length) {
          return menuSizes.map((size) => rng.nextInt(size));
        }
        const center = (observations[picked] as Observation).choices;
        return menuSizes.map((size, component) =>
          drawWeighted(
            Array.from({ length: size }, (_, option) =>
              option === center[component] ? priorWeight + 1 : priorWeight,
            ),
            rng,
          ),
        );
      },
    };
  }

  const histograms = menuSizes.map((size, component) => {
    const counts = new Array<number>(size).fill(priorWeight);
    for (const observation of observations) {
      const choice = observation.choices[component];
      if (choice !== undefined && choice < size) {
        counts[choice] = (counts[choice] as number) + 1;
      }
    }
    const total = observations.length + priorWeight * size;
    return counts.map((count) => count / total);
  });

  return {
    density: (choices) => {
      let product = 1;
      for (let component = 0; component < histograms.length; component += 1) {
        const histogram = histograms[component] as number[];
        product *= histogram[choices[component] as number] as number;
      }
      return product;
    },
    sample: (rng) =>
      histograms.map((histogram) => drawWeighted(histogram, rng)),
  };
}

function kernel(args: {
  choices: readonly number[];
  center: readonly number[];
  menuSizes: readonly number[];
  priorWeight: number;
}): number {
  const { choices, center, menuSizes, priorWeight } = args;

  let product = 1;
  for (let component = 0; component < menuSizes.length; component += 1) {
    const size = menuSizes[component] as number;
    const matched = choices[component] === center[component];
    product *= (priorWeight + (matched ? 1 : 0)) / (priorWeight * size + 1);
  }
  return product;
}

function uniformKernel(menuSizes: readonly number[]): number {
  let product = 1;
  for (const size of menuSizes) {
    product /= size;
  }
  return product;
}

function drawWeighted(weights: readonly number[], rng: Rng): number {
  return rng.weighted(
    weights.map((_, option) => option),
    weights,
  );
}
