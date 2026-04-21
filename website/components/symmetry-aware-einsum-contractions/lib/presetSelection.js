export const CUSTOM_IDX = -1;

const PRESET_GLYPHS_BY_CLASSIFICATION = {
  trivial: '·',
  allVisible: '◌',
  allSummed: '∑',
  mixed: '⟡',
  singleton: '①',
  directProduct: '⊗',
  young: 'Y',
  bruteForceOrbit: '◎',
};

function presetGlyphForClassification(leafId) {
  return PRESET_GLYPHS_BY_CLASSIFICATION[leafId] ?? '◆';
}

export function getPresetControlSelection(exampleIdx, isDirty) {
  return isDirty ? CUSTOM_IDX : exampleIdx;
}

export function presetToState(ex) {
  return {
    variables: ex.variables.map((v) => ({
      name: v.name,
      rank: v.rank,
      symmetry: v.symmetry || 'none',
      symAxes: v.symAxes ? [...v.symAxes] : null,
      generators: v.generators || '',
    })),
    subscriptsStr: ex.expression.subscripts,
    outputStr: ex.expression.output,
    operandNamesStr: ex.expression.operandNames,
  };
}

export function getPresetSummary(ex) {
  const leafId = ex.regimeId ?? ex.shapeId ?? null;
  const caseIds = leafId == null
    ? []
    : ['singleton', 'directProduct', 'young', 'bruteForceOrbit'].includes(leafId)
      ? ['mixed', leafId]
      : [leafId];

  return {
    id: ex.id,
    name: ex.name,
    formula: ex.formula,
    description: ex.description ?? '',
    glyph: presetGlyphForClassification(leafId),
    caseIds,
    regimeId: leafId,
    expectedGroup: ex.expectedGroup ?? '',
    color: ex.color ?? '#7C3AED',
  };
}

export function resolvePresetSelection(examples, selectedPresetIdx) {
  if (selectedPresetIdx === CUSTOM_IDX) {
    return {
      kind: 'custom',
      activePresetIdx: CUSTOM_IDX,
      dirtyState: 'preserve',
      presetState: null,
      presetSummary: null,
    };
  }

  const example = examples[selectedPresetIdx];
  if (!example) {
    return {
      kind: 'invalid',
      activePresetIdx: CUSTOM_IDX,
      dirtyState: 'preserve',
      presetState: null,
      presetSummary: null,
    };
  }

  return {
    kind: 'preset',
    activePresetIdx: selectedPresetIdx,
    dirtyState: 'clear',
    presetState: presetToState(example),
    presetSummary: getPresetSummary(example),
  };
}
