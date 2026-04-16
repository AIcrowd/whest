export const CUSTOM_IDX = -1;

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
  return {
    id: ex.id,
    name: ex.name,
    formula: ex.formula,
    description: ex.description ?? '',
    caseType: ex.caseType ?? null,
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
