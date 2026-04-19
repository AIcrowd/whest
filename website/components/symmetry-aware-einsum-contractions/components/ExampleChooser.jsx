import { useState, useCallback, useMemo, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { validateAll } from '../engine/validation.js';
import { varField } from '../engine/validationMessages.js';
import { generatePython } from '../engine/pythonCodegen.js';
import { buildVariableColors, SYMMETRY_ICONS, contrastText } from '../engine/colorPalette.js';
import { parseCycleNotation } from '../engine/cycleParser.js';
import { cn } from '../lib/utils.js';
import { CUSTOM_IDX, getPresetSummary, presetToState, resolvePresetSelection } from '../lib/presetSelection.js';
import { variableSymmetryLabel } from '../lib/symmetryLabel.js';
import CaseBadge from './CaseBadge.jsx';
import ExplorerField from './ExplorerField.jsx';
import PythonCodeBlock from './PythonCodeBlock.jsx';
import SymmetryBadge from './SymmetryBadge.jsx';

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

// Thin wrapper: the shared util lives in lib/symmetryLabel.js so the
// appendix modal's savings table can reuse the exact same vocabulary.
const badgeLabel = variableSymmetryLabel;

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
  onDimensionChange,
  onCustom,
  onCustomExample,
  onPreviewChange,
  onDirtyChange,
}) {
  const initialSelection = resolvePresetSelection(examples, selectedPresetIdx);
  const initialPresetIdx = initialSelection.activePresetIdx >= 0 ? initialSelection.activePresetIdx : 0;
  const initial = initialSelection.presetState ?? presetToState(examples[initialPresetIdx]);

  const [variables, setVariables] = useState(initial.variables);
  const [subscriptsStr, setSubscriptsStr] = useState(initial.subscriptsStr);
  const [outputStr, setOutputStr] = useState(initial.outputStr);
  const [operandNamesStr, setOperandNamesStr] = useState(initial.operandNamesStr);
  const [activePresetIdx, setActivePresetIdx] = useState(initialPresetIdx);

  // Fields the user has "finished" interacting with (blur or button click).
  // Validation errors are suppressed until their field lands here — prevents
  // half-typed-state noise. Clicking Analyze promotes every offending field.
  const [touched, setTouched] = useState(() => new Set());
  const touch = useCallback((field) => {
    if (!field) return;
    setTouched((prev) => (prev.has(field) ? prev : new Set(prev).add(field)));
  }, []);

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
    setTouched(new Set());
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
      setTouched(new Set());
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

  const visibleErrors = useMemo(
    () => validation.errors.filter((error) => touched.has(error.field)),
    [validation.errors, touched],
  );
  const errorFieldSet = useMemo(
    () => new Set(visibleErrors.map((error) => error.field)),
    [visibleErrors],
  );

  const applyFix = useCallback((fix) => {
    if (!fix || typeof fix.apply !== 'function') return;
    const next = fix.apply({
      variables,
      subscriptsStr,
      outputStr,
      operandNamesStr,
    });
    if (next.variables !== variables) setVariables(next.variables);
    if (next.subscriptsStr !== subscriptsStr) setSubscriptsStr(next.subscriptsStr);
    if (next.outputStr !== outputStr) setOutputStr(next.outputStr);
    if (next.operandNamesStr !== operandNamesStr) setOperandNamesStr(next.operandNamesStr);
    markCustom();
  }, [markCustom, operandNamesStr, outputStr, subscriptsStr, variables]);
  const pythonCode = useMemo(
    () => generatePython(variables, subscriptsStr, outputStr, operandNamesStr, dimensionN),
    [variables, subscriptsStr, outputStr, operandNamesStr, dimensionN],
  );

  useEffect(() => {
    const opsArr = operandNamesStr.split(',').map((part) => part.trim()).filter(Boolean);
    onPreviewChange?.({
      id: activePresetIdx >= 0 ? examples[activePresetIdx].id : 'custom',
      name: activePresetIdx >= 0 ? examples[activePresetIdx].name : 'Custom',
      formula: `einsum('${subscriptsStr}->${outputStr.trim()}', ${opsArr.join(', ')})`,
      variables,
      expression: {
        subscripts: subscriptsStr,
        output: outputStr.trim(),
        operandNames: operandNamesStr,
      },
    });
  }, [activePresetIdx, examples, onPreviewChange, operandNamesStr, outputStr, subscriptsStr, variables]);

  const handleAnalyze = useCallback(() => {
    if (!validation.valid) {
      // Reveal every offending field so the user can see what's blocking them.
      setTouched((prev) => {
        const next = new Set(prev);
        for (const error of validation.errors) next.add(error.field);
        return next;
      });
      return;
    }

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
      variables,
      expression: {
        subscripts: subscriptsStr,
        output: out,
        operandNames: operandNamesStr,
      },
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
    <>
      <div className="space-y-4">
        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">Variables</div>
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
                      className={cn(
                        'h-auto w-20 rounded border px-2.5 py-1.5 text-sm font-mono focus:border-coral focus:ring-coral/30',
                        errorFieldSet.has(varField(idx, 'name')) ? 'border-red-300' : 'border-gray-200',
                      )}
                      value={variable.name}
                      onChange={(event) => updateVar(idx, 'name', event.target.value)}
                      onBlur={() => touch(varField(idx, 'name'))}
                      placeholder="X"
                      maxLength={8}
                    />

                    <div className="flex items-center gap-1">
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-xs"
                        className="h-8 w-8 rounded border-gray-200 text-sm hover:bg-gray-50 disabled:opacity-30"
                        onClick={() => {
                          updateVar(idx, 'rank', Math.max(1, variable.rank - 1));
                          touch(varField(idx, 'rank'));
                        }}
                        disabled={variable.rank <= 1}
                      >
                        -
                      </Button>
                      <span className="w-4 text-center text-sm font-mono">{variable.rank}</span>
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-xs"
                        className="h-8 w-8 rounded border-gray-200 text-sm hover:bg-gray-50 disabled:opacity-30"
                        onClick={() => {
                          updateVar(idx, 'rank', Math.min(8, variable.rank + 1));
                          touch(varField(idx, 'rank'));
                        }}
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
                          'cursor-pointer rounded-full border px-2.5 py-1 text-xs font-medium transition-colors',
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
                              'h-8 w-8 rounded-full border text-sm font-mono transition-colors',
                              isSelected
                                ? 'border-gray-900 bg-gray-900 text-white'
                                : 'border-gray-200 bg-white text-gray-500 hover:border-gray-400',
                            )}
                            onClick={() => {
                              toggleAxis(idx, axisIdx);
                              touch(varField(idx, 'axes'));
                            }}
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
                      labelClassName="text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400"
                      inputClassName={cn(
                        'h-auto px-3 py-1.5 text-sm font-mono focus:border-coral focus:ring-coral/30',
                        errorFieldSet.has(varField(idx, 'generators')) ? 'border-red-300' : 'border-gray-200',
                      )}
                      value={variable.generators}
                      onChange={(event) => updateVar(idx, 'generators', event.target.value)}
                      onBlur={() => touch(varField(idx, 'generators'))}
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

            <button
              type="button"
              onClick={addVar}
              aria-label="Add variable"
              className="group flex min-h-[140px] flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-gray-300 p-4 text-gray-500 transition-colors hover:border-gray-400 hover:bg-gray-50 hover:text-gray-700 focus:outline-none focus-visible:border-gray-500 focus-visible:ring-2 focus-visible:ring-coral/30"
            >
              <span
                aria-hidden="true"
                className="flex h-8 w-8 items-center justify-center rounded-full border border-current text-xl leading-none"
              >
                +
              </span>
              <span className="text-sm font-medium">Add variable</span>
            </button>
          </div>
        </div>

        <div>
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">Expression</div>
          <div className="my-4">
            <div className="flex flex-wrap items-start gap-2">
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm font-semibold text-coral">einsum(&#39;</span>
              <div className="flex min-w-[60px] flex-1 flex-col items-center">
                <Input
                  className={cn(
                    'h-auto w-full rounded-lg border px-3 py-1.5 font-mono text-sm focus:border-coral focus:ring-coral/30',
                    errorFieldSet.has('subscripts') ? 'border-red-300' : 'border-gray-200',
                  )}
                  value={subscriptsStr}
                  onChange={(event) => handleSubscriptsChange(event.target.value.toLowerCase())}
                  onBlur={() => touch('subscripts')}
                  placeholder="ia,ib"
                />
                <span className="mt-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">subscripts</span>
              </div>
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm text-gray-400">&rarr;</span>
              <div className="flex min-w-[60px] flex-1 flex-col items-center">
                <Input
                  className={cn(
                    'h-auto w-full rounded-lg border px-3 py-1.5 font-mono text-sm focus:border-coral focus:ring-coral/30',
                    errorFieldSet.has('output') ? 'border-red-300' : 'border-gray-200',
                  )}
                  value={outputStr}
                  onChange={(event) => handleOutputChange(event.target.value.toLowerCase())}
                  onBlur={() => touch('output')}
                  placeholder="ab"
                />
                <span className="mt-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">output</span>
              </div>
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm text-gray-400">&#39;,</span>
              <div className="flex min-w-[60px] flex-1 flex-col items-center">
                <Input
                  className={cn(
                    'h-auto w-full rounded-lg border px-3 py-1.5 font-mono text-sm focus:border-coral focus:ring-coral/30',
                    errorFieldSet.has('operands') ? 'border-red-300' : 'border-gray-200',
                  )}
                  value={operandNamesStr}
                  onChange={(event) => handleOperandNamesChange(event.target.value)}
                  onBlur={() => touch('operands')}
                  placeholder="X, X"
                />
                <span className="mt-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-gray-400">operands</span>
              </div>
              <span className="whitespace-nowrap pt-1.5 font-mono text-sm text-gray-400">)</span>
              {/* Primary CTA — mirrors the home-page Install button
                  (app/(home)/page.tsx:423–428): rounded-lg 8px, coral
                  ground, coral-hover on hover, coral-at-20% focus ring,
                  text-white / sm / medium. No shadow elevation (the docs
                  register is shadowless). Native <button> instead of the
                  shadcn Button variant so chrome matches the home CTA
                  exactly without override dance. */}
              <button
                type="button"
                className={cn(
                  'inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-lg bg-[var(--coral)] px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[var(--coral-hover)] focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--coral)]/20',
                  !validation.valid && 'opacity-60',
                )}
                onClick={handleAnalyze}
                title={validation.valid ? undefined : 'Click to see what needs fixing'}
              >
                <span aria-hidden>&#x25B6;</span> Analyze
              </button>
            </div>
          </div>
          <div className="mt-3 flex justify-end">
            {/* Dimension knob — styled as a neutral form-field chip per the
                design-system input spec (gray-200 border, white bg, mono
                13px). The coral sits on the slider accent only, so the
                chip reads as a utility control rather than a brand CTA. */}
            <label
              className="flex cursor-pointer items-center gap-2.5 rounded-full border border-gray-200 bg-white px-3.5 py-1.5 transition-colors hover:border-gray-300"
              title="Per-label dimension — a demo knob for visualising the contraction at different scales. It does not change the einsum's structural cost (|L|, |G|); it only scales |X| = nᴸ, which is how the brute-force estimate |X|·|G| (counted in (tuple, g) pair-touches, cap 1,500,000) moves with it."
            >
              <span className="font-mono text-xs font-semibold uppercase tracking-[0.04em] text-gray-400">n</span>
              <input
                type="range"
                min={2}
                max={25}
                value={dimensionN}
                onChange={(event) => onDimensionChange?.(Number(event.target.value))}
                className="h-1.5 w-40 cursor-pointer accent-[var(--coral)]"
              />
              <span className="w-6 text-center font-mono text-sm font-semibold text-gray-900">
                {dimensionN}
              </span>
            </label>
          </div>
        </div>

        {visibleErrors.length > 0 && (
          <div className="space-y-1 rounded-lg border border-red-200 bg-red-50 px-3 py-2">
            {visibleErrors.map((error, idx) => (
              <div
                key={`${error.code}-${error.field}-${idx}`}
                className="flex items-start gap-1.5 text-xs text-red-600"
              >
                <span className="mt-0.5 shrink-0 text-red-400">&#x26A0;</span>
                <span className="flex-1 leading-snug">{error.message}</span>
                {error.fix && (
                  <button
                    type="button"
                    onClick={() => applyFix(error.fix)}
                    className="shrink-0 rounded-md border border-red-300 bg-white px-2 py-0.5 text-xs font-semibold text-red-700 transition-colors hover:bg-red-100"
                  >
                    {error.fix.label}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
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
              <span className="rounded-full bg-gray-100 px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-gray-500">
                Freeform
              </span>
            </span>
            <code className="mt-1 block text-sm text-gray-500">Define your own contraction</code>
            <span className="mt-1 text-sm text-gray-600">Keep the current builder state and switch into custom mode.</span>
          </Button>

          {presetSummaries.map((summary, idx) => (
            <Button
              key={summary.id}
              type="button"
              variant="outline"
              onClick={() => loadPreset(idx)}
              className={cn(
                'flex h-auto w-full items-start justify-start gap-3 px-4 py-3 text-left transition-colors',
                activePresetIdx === idx
                  ? 'border-coral bg-coral-light/50 ring-2 ring-coral/30'
                  : 'border-gray-200 hover:border-gray-300',
              )}
            >
              <span className="mt-0.5 h-full min-h-10 w-1 shrink-0 rounded-full" style={{ backgroundColor: summary.color }} />
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-2">
                  <span className="truncate text-sm font-medium text-gray-900">{summary.name}</span>
                  {summary.regimeId && <CaseBadge regimeId={summary.regimeId} size="sm" variant="pill" />}
                  <SymmetryBadge value={summary.expectedGroup} className="shrink-0" />
                </span>
                <code className="mt-1 block truncate text-sm text-gray-500">{summary.formula}</code>
                <span className="mt-1 block text-sm text-gray-400">{summary.description}</span>
              </span>
            </Button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 items-stretch lg:grid-cols-[minmax(0,60%)_minmax(0,40%)]">
        {/* Left cell is shorter than the right (PythonCodeBlock fills its cell
            exactly). justify-center spreads the slack equally above and below
            the builder so the variables toolbar + expression row sit centred
            rather than huddled at the top of the cell. */}
        <div className="flex min-w-0 h-full flex-col justify-center">{builderContent}</div>
        <div className="min-w-0 h-full">
          <PythonCodeBlock
            code={pythonCode}
            className="h-full"
            contentClassName="h-full"
          />
        </div>
      </div>
    </div>
  );
}
