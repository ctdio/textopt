import type { Rng } from "./rng.js";

export type BatchSampler<Datum> = ((args: {
  trainset: readonly Datum[];
  /**
   * Position in the sequence of minibatches drawn so far, not the loop
   * iteration. Draws made concurrently within one iteration arrive as
   * consecutive positions, so siblings diagnose different failures instead of
   * sharing a batch.
   */
  iteration: number;
  rng: Rng;
}) => number[]) & {
  /**
   * Position within the sampler's own schedule, checkpointed alongside the
   * random stream. Without it a resumed run restarts its epoch and re-walks
   * minibatches the interrupted run had already spent.
   */
  state?: () => unknown;
  restore?: (state: unknown) => void;
};

/**
 * Shuffles the trainset once per epoch and walks it in fixed-size chunks, so
 * every training example is seen once before any is seen twice.
 */
export function createEpochShuffledSampler<Datum>(args: {
  minibatchSize: number;
}): BatchSampler<Datum> {
  const { minibatchSize } = args;

  let shuffled: number[] = [];
  let epoch = -1;
  let lastTrainsetSize = -1;

  const sampler: BatchSampler<Datum> = ({ trainset, iteration, rng }) => {
    if (trainset.length === 0) {
      throw new Error("Cannot sample a minibatch from an empty trainset");
    }

    const baseIndex = iteration * minibatchSize;
    const currentEpoch =
      shuffled.length === 0 ? 0 : Math.floor(baseIndex / shuffled.length);

    if (
      shuffled.length === 0 ||
      trainset.length !== lastTrainsetSize ||
      currentEpoch > epoch
    ) {
      epoch = currentEpoch;
      lastTrainsetSize = trainset.length;
      shuffled = buildPaddedShuffle({
        size: trainset.length,
        minibatchSize,
        rng,
      });
    }

    const start = baseIndex % shuffled.length;
    return shuffled.slice(start, start + minibatchSize);
  };

  // The shuffle is drawn once per epoch, so it cannot be replayed from the
  // random stream alone: a resumed run that reshuffled would walk a different
  // epoch and re-spend minibatches the interrupted run had already seen.
  sampler.state = () => ({ shuffled: [...shuffled], epoch, lastTrainsetSize });
  sampler.restore = (state: unknown) => {
    if (!isSamplerState(state)) {
      return;
    }
    shuffled = [...state.shuffled];
    epoch = state.epoch;
    lastTrainsetSize = state.lastTrainsetSize;
  };

  return sampler;
}

function isSamplerState(state: unknown): state is {
  shuffled: number[];
  epoch: number;
  lastTrainsetSize: number;
} {
  if (state === null || typeof state !== "object") {
    return false;
  }
  const candidate = state as Record<string, unknown>;
  return (
    Array.isArray(candidate.shuffled) &&
    typeof candidate.epoch === "number" &&
    typeof candidate.lastTrainsetSize === "number"
  );
}

function buildPaddedShuffle(args: {
  size: number;
  minibatchSize: number;
  rng: Rng;
}): number[] {
  const { size, minibatchSize, rng } = args;

  const indices = rng.shuffle(Array.from({ length: size }, (_, i) => i));
  const remainder = indices.length % minibatchSize;
  const padding = remainder === 0 ? 0 : minibatchSize - remainder;

  const frequencies = new Map<number, number>(
    indices.map((index) => [index, 1]),
  );
  for (let i = 0; i < padding; i += 1) {
    const leastUsed = indices.reduce((best, index) =>
      (frequencies.get(index) as number) < (frequencies.get(best) as number)
        ? index
        : best,
    );
    indices.push(leastUsed);
    frequencies.set(leastUsed, (frequencies.get(leastUsed) as number) + 1);
  }

  return indices;
}
