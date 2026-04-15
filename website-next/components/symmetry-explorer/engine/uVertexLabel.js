/**
 * Build human-readable labels for U-vertices in the bipartite graph.
 *
 * Format: "A0.i" where A is the operand name, 0 is the operand occurrence
 * index (only shown when the operand appears more than once), and i is the
 * subscript label at that axis position.
 *
 * @param {Array} uVertices - array of { opIdx, labels }
 * @param {Object} example - example with operandNames and subscripts
 * @returns {string[]} - label for each U-vertex
 */
export function buildUVertexLabels(uVertices, example) {
  const opNames = example?.operandNames || [];

  // Count occurrences of each operand name to decide whether to show index
  const nameCounts = {};
  for (const name of opNames) {
    nameCounts[name] = (nameCounts[name] || 0) + 1;
  }

  // Track which occurrence of each name this operand is
  const nameOccurrence = {};
  const opOccurrenceIdx = [];
  for (const name of opNames) {
    nameOccurrence[name] = (nameOccurrence[name] || 0);
    opOccurrenceIdx.push(nameOccurrence[name]);
    nameOccurrence[name]++;
  }

  return uVertices.map(u => {
    const opIdx = u.opIdx;
    const name = opNames[opIdx] || `Op${opIdx}`;
    const lbl = [...(u.labels || [])].sort().join(',');

    // Find which axis position this label is at
    // (for single-label U-vertices, find the char in the subscript)
    const showIdx = nameCounts[name] > 1;
    const prefix = showIdx ? `${name}${opOccurrenceIdx[opIdx]}` : name;
    return `${prefix}.${lbl}`;
  });
}
