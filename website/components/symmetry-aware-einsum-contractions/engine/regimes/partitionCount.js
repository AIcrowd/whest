import { restrictStabilizerToPositions } from '../outputOrbit.js';
import {
  countMapOrbitsUnderH,
  generateTypedSetPartitions,
  inducedBlockActionSize,
  inducedPrefixMaps,
  numBlocks,
  partitionKey,
  partitionOrbitReps,
  typedLabelingCount,
} from '../partition/typedPartitions.js';

const DEFAULT_PARTITION_BUDGET = 20000;

export const partitionCountRegime = {
  id: 'partitionCount',

  recognize(ctx) {
    const partitions = generateTypedSetPartitions(ctx.sizes);
    if (partitions.length > DEFAULT_PARTITION_BUDGET) {
      return {
        fired: false,
        reason: `typed partition count refused: ${partitions.length} partitions exceed budget ${DEFAULT_PARTITION_BUDGET}`,
      };
    }
    return {
      fired: true,
      reason: `typed partition count over ${partitions.length} equality patterns`,
    };
  },

  compute(ctx) {
    const { elements, sizes, visiblePositions } = ctx;
    const hElements = restrictStabilizerToPositions(elements, visiblePositions);
    const partitions = generateTypedSetPartitions(sizes);
    const reps = partitionOrbitReps(partitions, elements);

    let total = 0;
    const subTrace = [];

    for (const partition of reps) {
      const labelings = typedLabelingCount(partition, sizes);
      const blockActionSize = inducedBlockActionSize(partition, elements);
      if (labelings % blockActionSize !== 0) {
        throw new Error(
          `partition ${partitionKey(partition)} has labeling count ${labelings} not divisible by block action size ${blockActionSize}`,
        );
      }

      const inputOrbitCount = labelings / blockActionSize;
      const maps = inducedPrefixMaps(partition, elements, visiblePositions);
      const outputOrbitCount = countMapOrbitsUnderH(maps, hElements);
      const term = inputOrbitCount * outputOrbitCount;
      total += term;

      subTrace.push({
        partition: partitionKey(partition),
        blocks: numBlocks(partition),
        typedLabelings: labelings,
        blockActionSize,
        inputOrbitCount,
        inducedMapCount: maps.size,
        outputOrbitCount,
        contribution: term,
      });
    }

    return {
      count: total,
      latex: String.raw`A = \sum_{\tilde{x}\in P_{\mathrm{typed}}(L)/G} \frac{\prod_s (n_s)_{b_s(\tilde{x})}}{|\overline{G}_{\tilde{x}}|}\,|A_{\tilde{x}}/H|`,
      latexSymbolic: String.raw`A = \#\{(O,Q): \pi_V(O)\cap Q\ne\varnothing\}`,
      subTrace,
    };
  },
};
