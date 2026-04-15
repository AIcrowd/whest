import { useState, useCallback, useMemo, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { validateAll } from '../engine/validation.js';
import { generatePython } from '../engine/pythonCodegen.js';
import { buildVariableColors, SYMMETRY_ICONS, contrastText } from '../engine/colorPalette.js';
import { parseCycleNotation } from '../engine/cycleParser.js';
import { cn } from '../lib/utils.js';
import { CUSTOM_IDX, getPresetSummary, presetToState, resolvePresetSelection } from '../lib/presetSelection.js';
import CaseBadge from './CaseBadge.jsx';
import ExplorerField from './ExplorerField.jsx';
import ExplorerSectionCard from './ExplorerSectionCard.jsx';
import PythonCodeBlock from './PythonCodeBlock.jsx';

const SYM_TYPES = ['none', 'symmetric', 'cyclic', 'dihedral', 'custom'];
const SYM_LABELS = {
  none: 'dense',
  symmetric: 'S_k',
  cyclic: 'C_k',
  dihedral: 'D_k',
  custom: 'custom',
};

function groupOrder(symmetry, k) {
  switch (symmetry) {
    case 'symmetric': {
      let factorial = 1;
      for (let i = 2; i <= k; i += 1) factorial *= i;
      return factorial;
    }
    case 'cyclic':
      return k;
    case 'dihedral':
      return 2 * k;
    default:
      return 1;
  }
}

function badgeLabel(variable) {
  const { symmetry, rank, symAxes, generators } = variable;
  if (symmetry === 'none') return 'dense';
  if (symmetry === 'custom') {
    if (!generators || !generators.trim()) return 'custom';
    const parsed = parseCycleNotation(generators);
    if (parsed.error || !parsed.generators) return 'custom';
    return `custom (${parsed.generators.length} gen${parsed.generators.length !== 1 ? 's' : ''})`;
  }
  const k = (symAxes && symAxes.length) || rank;
  const prefix = symmetry === 'symmetric' ? 'S' : symmetry === 'cyclic' ? 'C' : 'D';
  return `${prefix}${k}`;
}

function badgeOrder(variable) {
  const { symmetry, rank, symAxes } = variable;
  if (symmetry === 'none' || symmetry === 'custom') return null;
  const k = (symAxes && symAxes.length) || rank;
  return groupOrder(symmetry, k);
}

function buildPerOpSymmetry(variables, operandNamesStr, subscriptsStr) {
  const varMap = new Map();
  for (const variable of variables) {
    varMap.set(variable.name.trim(), variable);
  }

  const opNames = operandNamesStr.split(',').map((part) => part.trim()).filter(Boolean);
  const subs = subscriptsStr.split(',').map((part) => part.trim());

  return opNames.map((name, idx) => {
    const variable = varMap.get(name);
    if (!variable || variable.symmetry === 'none') return null;

    const sub = subs[idx] || '';
    const allAxes = sub.length;

    if (variable.symmetry === 'symmetric') {
      if (!variable.symAxes || variable.symAxes.length === allAxes) return 'symmetric';
      return { type: 'symmetric', axes: [...variable.symAxes] };
    }
    if (variable.symmetry === 'cyclic') {
      if (!variable.symAxes || variable.symAxes.length === allAxes) {
        return { type: 'cyclic', axes: [...Array(allAxes).keys()] };
      }
      return { type: 'cyclic', axes: [...variable.symAxes] };
    }
    if (variable.symmetry === 'dihedral') {
      if (!variable.symAxes || variable.symAxes.length === allAxes) {
        return { type: 'dihedral', axes: [...Array(allAxes).keys()] };
      }
      return { type: 'dihedral', axes: [...variable.symAxes] };
    }
    if (variable.symmetry === 'custom') {
      const parsed = parseCycleNotation(variable.generators || '');
      const axes = variable.symAxes || [...Array(allAxes).keys()];
      return {
        type: 'custom',
        axes: [...axes],
        generators: parsed.generators || [],
      };
    }

    return null;
  });
}

export default function ExampleChooser({
  examples,
  onSelect,
  selectedPresetIdx = 0,
  dimensionN,
  onCustom,
  onCustomExample,
  onDirtyChange,
  act,
  checkpointItems = [],
}) {
  const initialSelection = resolvePresetSelection(examples, selectedPresetIdx);
  const initialPresetIdx = initialSelection.activePresetIdx >= 0 ? initialSelection.activePresetIdx : 0;
  const initial = initialSelection.presetState ?? presetToState(examples[initialPresetIdx]);

  const [variables, setVariables] = useState(initial.variables);
  const [subscriptsStr, setSubscriptsStr] = useState(initial.subscriptsStr);
  const [outputStr, setOutputStr] = useState(initial.outputStr);
  const [operandNamesStr, setOperandNamesStr] = useState(initial.operandNamesStr);
  const [activePresetIdx, setActivePresetIdx] = useState(initialPresetIdx);

  const presetSummaries = useMemo(() => examples.map(getPresetSummary), [examples]);

  const loadPreset = useCallback((idx) => {
    const selection = resolvePresetSelection(examples, idx);
    if (selection.kind !== 'preset') return;
    const { presetState } = selection;
    setVariables(presetState.variables);
    setSubscriptsStr(presetState.subscriptsStr);
    setOutputStr(presetState.outputStr);
    setOperandNamesStr(presetState.operandNamesStr);
    setActivePresetIdx(selection.activePresetIdx);
    onSelect(idx);
    onDirtyChange?.(false);
  }, [examples, onDirtyChange, onSelect]);

  useEffect(() => {
    const selection = resolvePresetSelection(examples, selectedPresetIdx);

    if (selection.kind === 'custom' || selection.kind === 'invalid') {
      setActivePresetIdx(CUSTOM_IDX);
      return;
    }

    const { presetState } = selection;
    setVariables(presetState.variables);
    setSubscriptsStr(presetState.subscriptsStr);
    setOutputStr(presetState.outputStr);
    setOperandNamesStr(presetState.operandNamesStr);
    setActivePresetIdx(selection.activePresetIdx);
    if (selection.dirtyState === 'clear') {
      onDirtyChange?.(false);
    }
  }, [examples, onDirtyChange, selectedPresetIdx]);

  const markCustom = useCallback(() => {
    setActivePresetIdx(CUSTOM_IDX);
    onDirtyChange?.(true);
  }, [onDirtyChange]);

  const updateVar = useCallback((idx, field, value) => {
    setVariables((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], [field]: value };

      if (field === 'symmetry') {
        if (value === 'none') {
          next[idx].symAxes = null;
          next[idx].generators = '';
        } else if (value === 'custom') {
          if (!next[idx].symAxes) {
            next[idx].symAxes = [...Array(next[idx].rank).keys()];
          }
        } else {
          next[idx].symAxes = [...Array(next[idx].rank).keys()];
          next[idx].generators = '';
        }
      }

      if (field === 'rank') {
        const newRank = value;
        if (next[idx].symAxes) {
          next[idx].symAxes = next[idx].symAxes.filter((axis) => axis < newRank);
        }
        if (next[idx].symmetry !== 'none' && next[idx].symmetry !== 'custom') {
          next[idx].symAxes = [...Array(newRank).keys()];
        }
      }

      return next;
    });
    markCustom();
  }, [markCustom]);

  const toggleAxis = useCallback((varIdx, axisIdx) => {
    setVariables((prev) => {
      const next = [...prev];
      const variable = { ...next[varIdx] };
      const axes = new Set(variable.symAxes || []);
      if (axes.has(axisIdx)) axes.delete(axisIdx);
      else axes.add(axisIdx);
      variable.symAxes = [...axes].sort((a, b) => a - b);
      next[varIdx] = variable;
      return next;
    });
    markCustom();
  }, [markCustom]);

  const addVar = useCallback(() => {
    setVariables((prev) => [
      ...prev,
      { name: 'A', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ]);
    markCustom();
  }, [markCustom]);

  const removeVar = useCallback((idx) => {
    setVariables((prev) => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== idx)));
    markCustom();
  }, [markCustom]);

  const handleSubscriptsChange = useCallback((value) => {
    setSubscriptsStr(value);
    markCustom();
  }, [markCustom]);

  const handleOutputChange = useCallback((value) => {
    setOutputStr(value);
    markCustom();
  }, [markCustom]);

  const handleOperandNamesChange = useCallback((value) => {
    setOperandNamesStr(value);
    markCustom();
  }, [markCustom]);

  const varColors = useMemo(() => buildVariableColors(variables), [variables]);
  const validation = useMemo(
    () => validateAll(variables, subscriptsStr, outputStr, operandNamesStr),
    [variables, subscriptsStr, outputStr, operandNamesStr],
  );
  const pythonCode = useMemo(
    () => generatePython(variables, subscriptsStr, outputStr, operandNamesStr, dimensionN),
    [variables, subscriptsStr, outputStr, operandNamesStr, dimensionN],
  );

  const handleAnalyze = useCallback(() => {
    if (!validation.valid) return;

    const subs = subscriptsStr.split(',').map((part) => part.trim());
    const out = outputStr.trim();
    const opsArr = operandNamesStr.split(',').map((part) => part.trim()).filter(Boolean);
    const perOpSymArr = buildPerOpSymmetry(variables, operandNamesStr, subscriptsStr);
    const hasAnySym = perOpSymArr.some((symmetry) => symmetry !== null);

    const formula = `einsum('${subscriptsStr}->${out}', ${opsArr.join(', ')})`;

    const customExample = {
      id: activePresetIdx >= 0 ? examples[activePresetIdx].id : 'custom',
      name: activePresetIdx >= 0 ? examples[activePresetIdx].name : 'Custom',
      formula,
      subscripts: subs,
      output: out,
      operandNames: opsArr,
      perOpSymmetry: hasAnySym ? perOpSymArr : null,
      description: activePresetIdx >= 0 ? examples[activePresetIdx].description : 'User-defined expression',
      expectedGroup: activePresetIdx >= 0 ? examples[activePresetIdx].expectedGroup : '',
      color: activePresetIdx >= 0 ? examples[activePresetIdx].color : '#7C3AED',
    };

    onCustomExample(customExample);
    onDirtyChange?.(false);
  }, [activePresetIdx, examples, onCustomExample, onDirtyChange, operandNamesStr, outputStr, subscriptsStr, validation.valid, variables]);

  const builderContent = (
    <ExplorerSectionCard
      eyebrow="Act 1"
      title={act?.heading}
      description={act?.why}
      className="border-gray-200 bg-white"
      contentClassName="pt-5"
    >
      {checkpointItems.length > 0 && (
        <div className="mt-4 grid gap-3 rounded-xl border border-gray-200 bg-gray-50 p-4 sm:grid-cols-2">
          {checkpointItems.map((item) => (
            <div key={item.label}>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-400">{item.label}</div>
              <div className="mt-1 text-sm text-gray-700">{item.value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-6 space-y-4">
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">Variables</div>
          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {variables.map((variable, idx) => {
              const variableColor = varColors[variable.name] || {};
              const color = variableColor.color || '#888';

              return (
                <div
                  key={`${variable.name}-${idx}`}
                  className="rounded-lg border p-2"
                  style={{ borderColor: color }}
                >
                  <div className="mb-1 flex items-center gap-1.5">
                    <Input
                      className="h-auto w-16 rounded border border-gray-200 px-1.5 py-0.5 text-xs font-mono focus:border-coral focus:ring-coral/30"
                      value={variable.name}
                      onChange={(event) => updateVar(idx, 'name', event.target.value)}
                      placeholder="X"
                      maxLength={8}
                    />

                    <div className="flex items-center gap-1">
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-xs"
                        className="h-6 w-6 rounded border-gray-200 text-xs hover:bg-gray-50 disabled:opacity-30"
                        onClick={() => updateVar(idx, 'rank', Math.max(1, variable.rank - 1))}
                        disabled={variable.rank <= 1}
                      >
                        -
                      </Button>
                      <span className="w-4 text-center text-sm font-mono">{variable.rank}</span>
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-xs"
                        className="h-6 w-6 rounded border-gray-200 text-xs hover:bg-gray-50 disabled:opacity-30"
                        onClick={() => updateVar(idx, 'rank', Math.min(8, variable.rank + 1))}
                        disabled={variable.rank >= 8}
                      >
                        +
                      </Button>
                    </div>

                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-xs"
                      className="ml-auto flex h-5 w-5 rounded-full p-0 text-gray-400 transition-colors hover:bg-red-50 hover:text-red-500 disabled:opacity-20"
                      onClick={() => removeVar(idx)}
                      disabled={variables.length <= 1}
                      title="Remove variable"
                    >
                      ×
                    </Button>
                  </div>

                  <div className="mb-1 flex flex-wrap gap-1">
                    {SYM_TYPES.map((symType) => (
                      <Button
                        key={symType}
                        type="button"
                        variant="outline"
                        size="xs"
                        className={cn(
                          'cursor-pointer rounded-full border px-2 py-0.5 text-[10px] font-medium transition-colors',
                          variable.symmetry === symType
                            ? 'border-gray-900 bg-gray-900 text-white'
                            : 'border-gray-200 bg-white text-gray-600 hover:border-gray-400',
                        )}
                        onClick={() => updateVar(idx, 'symmetry', symType)}
                      >
                        {SYMMETRY_ICONS[symType] ? `${SYMMETRY_ICONS[symType]} ` : ''}
                        {SYM_LABELS[symType]}
                      </Button>
                    ))}
                  </div>

                  {variable.symmetry !== 'none' && variable.rank > 0 && (
                    <div className="mb-1 flex flex-wrap gap-1">
                      {Array.from({ length: variable.rank }, (_, axisIdx) => {
                        const isSelected = variable.symAxes && variable.symAxes.includes(axisIdx);
                        return (
                          <Button
                            key={axisIdx}
                            type="button"
                            variant="outline"
                            size="icon-xs"
                            className={cn(
                              'h-6 w-6 rounded-full border text-[10px] font-mono transition-colors',
                              isSelected
                                ? 'border-gray-900 bg-gray-900 text-white'
                                : 'border-gray-200 bg-white text-gray-500 hover:border-gray-400',
                            )}
                            onClick={() => toggleAxis(idx, axisIdx)}
                            title={`Axis ${axisIdx}`}
                          >
                            {axisIdx}
                          </Button>
                        );
                      })}
                    </div>
                  )}

                  {variable.symmetry === 'custom' && (
                    <ExplorerField
                      label="Generators"
                      className="mb-1"
                      labelClassName="text-[10px] tracking-[0.18em] text-gray-400"
                      inputClassName="h-auto border-gray-200 px-2 py-1 text-xs font-mono focus:border-coral focus:ring-coral/30"
                      value={variable.generators}
                      onChange={(event) => updateVar(idx, 'generators', event.target.value)}
                      placeholder="(0 1)(2 3), (0 2)(1 3)"
                      hint={(
                        <>
                          Cycle notation, comma-separated generators. E.g. <code className="rounded bg-gray-100 px-1">(0 1)</code> swaps axes 0 and 1.
                        </>
                      )}
                    />
                  )}

                  <div
                    className="mt-2 inline-block rounded-full px-2.5 py-1 text-xs font-semibold"
                    style={{ backgroundColor: color, color: contrastText(color) }}
                  >
                    {badgeLabel(variable)}
                    {badgeOrder(variable) != null && (
                      <span className="opacity-70"> order {badgeOrder(variable)}</span>
                    )}
                  </div>
                </div>
              );
            })}

            <Button
              type="button"
              variant="outline"
              className="self-center border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 transition-colors hover:border-gray-400 hover:bg-gray-50"
              onClick={addVar}
            >
              + Add Variable
            </Button>
          </div>
        </div>

        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">Expression</div>
          <div className="my-4">
            <div className="flex flex-wrap items-start gap-2">
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm font-semibold text-coral">einsum(&#39;</span>
              <div className="flex min-w-[60px] flex-1 flex-col items-center">
                <Input
                  className={cn(
                    'h-auto w-full rounded-lg border border-gray-200 px-3 py-1.5 font-mono text-sm focus:border-coral focus:ring-coral/30',
                    validation.errors.some((error) => error.includes('subscript') || error.includes('Subscript') || error.includes('operand')) && 'border-red-300',
                  )}
                  value={subscriptsStr}
                  onChange={(event) => handleSubscriptsChange(event.target.value.toLowerCase())}
                  placeholder="ia,ib"
                />
                <span className="mt-1 text-[9px] font-semibold uppercase tracking-wider text-gray-400">subscripts</span>
              </div>
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm text-gray-400">&rarr;</span>
              <div className="flex min-w-[60px] flex-1 flex-col items-center">
                <Input
                  className={cn(
                    'h-auto w-full rounded-lg border border-gray-200 px-3 py-1.5 font-mono text-sm focus:border-coral focus:ring-coral/30',
                    validation.errors.some((error) => error.includes('utput')) && 'border-red-300',
                  )}
                  value={outputStr}
                  onChange={(event) => handleOutputChange(event.target.value.toLowerCase())}
                  placeholder="ab"
                />
                <span className="mt-1 text-[9px] font-semibold uppercase tracking-wider text-gray-400">output</span>
              </div>
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm text-gray-400">&#39;,</span>
              <div className="flex min-w-[60px] flex-1 flex-col items-center">
                <Input
                  className={cn(
                    'h-auto w-full rounded-lg border border-gray-200 px-3 py-1.5 font-mono text-sm focus:border-coral focus:ring-coral/30',
                    validation.errors.some((error) => error.includes('operand') || error.includes('Operand')) && 'border-red-300',
                  )}
                  value={operandNamesStr}
                  onChange={(event) => handleOperandNamesChange(event.target.value)}
                  placeholder="X, X"
                />
                <span className="mt-1 text-[9px] font-semibold uppercase tracking-wider text-gray-400">operands</span>
              </div>
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm text-gray-400">)</span>
              <Button
                type="button"
                className="shrink-0 whitespace-nowrap bg-coral px-5 py-2 text-sm font-semibold text-white shadow-md transition-all hover:bg-coral-hover hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-40"
                onClick={handleAnalyze}
                disabled={!validation.valid}
              >
                &#x25B6; Analyze
              </Button>
            </div>
          </div>
        </div>

        {validation.errors.length > 0 && (
          <div className="space-y-0.5 rounded-lg border border-red-200 bg-red-50 px-3 py-2">
            {validation.errors.map((error, idx) => (
              <div key={idx} className="flex items-center gap-1.5 text-xs text-red-600">
                <span className="shrink-0 text-red-400">&#x26A0;</span>
                {error}
              </div>
            ))}
          </div>
        )}
      </div>
    </ExplorerSectionCard>
  );

  return (
    <div className="space-y-6">
      <div className="md:hidden" aria-label="Mobile preset examples">
        <div className="grid gap-2">
          <Button
            type="button"
            variant="outline"
            className={cn(
              'h-auto flex-col items-start justify-start rounded-xl border px-4 py-3 text-left transition-colors',
              activePresetIdx === CUSTOM_IDX
                ? 'border-coral bg-coral-light/50'
                : 'border-gray-200 hover:border-gray-300',
            )}
            onClick={() => {
              setActivePresetIdx(CUSTOM_IDX);
              onCustom?.();
            }}
          >
            <span className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-900">Custom</span>
              <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-500">
                Freeform
              </span>
            </span>
            <code className="mt-1 block text-[11px] text-gray-500">Define your own contraction</code>
            <span className="mt-1 text-xs text-gray-600">Keep the current builder state and switch into custom mode.</span>
          </Button>

          {presetSummaries.map((summary, idx) => (
            <Button
              key={summary.id}
              type="button"
              variant="outline"
              onClick={() => loadPreset(idx)}
              className="flex h-auto w-full items-start justify-start gap-3 border-gray-200 px-3 py-2.5 text-left"
            >
              <span className="mt-0.5 h-full min-h-10 w-1 shrink-0 rounded-full" style={{ backgroundColor: summary.color }} />
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium text-gray-900">{summary.name}</span>
                  {summary.caseType && <CaseBadge caseType={summary.caseType} size="xs" variant="compact" interactive={false} />}
                </span>
                <code className="mt-1 block truncate text-[11px] text-gray-500">{summary.formula}</code>
                <span className="mt-1 block text-[11px] text-gray-400">{summary.expectedGroup}</span>
              </span>
            </Button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(360px,3fr)]">
        <div className="min-w-0">{builderContent}</div>
        <div className="min-w-0">
          <PythonCodeBlock
            code={pythonCode}
            title="Reference Code"
            description="This is a generated Python sketch of the contraction you are about to analyze."
          />
        </div>
      </div>
    </div>
  );
}
