import { useState, useCallback, useMemo } from 'react';
import { validateAll } from '../engine/validation.js';
import { generatePython } from '../engine/pythonCodegen.js';
import { buildVariableColors, SYMMETRY_ICONS, contrastText } from '../engine/colorPalette.js';
import { parseCycleNotation, generatorIndices } from '../engine/cycleParser.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CUSTOM_IDX = -1;

const SYM_TYPES = ['none', 'symmetric', 'cyclic', 'dihedral', 'custom'];
const SYM_LABELS = {
  none: 'dense',
  symmetric: 'S_k',
  cyclic: 'C_k',
  dihedral: 'D_k',
  custom: 'custom',
};

/** Compute the order of a named symmetry group on k axes. */
function groupOrder(symmetry, k) {
  switch (symmetry) {
    case 'symmetric': {
      let f = 1;
      for (let i = 2; i <= k; i++) f *= i;
      return f;
    }
    case 'cyclic':
      return k;
    case 'dihedral':
      return 2 * k;
    default:
      return 1;
  }
}

/** Human-readable group name for the summary badge. */
function badgeLabel(variable) {
  const { symmetry, rank, symAxes, generators } = variable;
  if (symmetry === 'none') return 'dense';
  if (symmetry === 'custom') {
    if (!generators || !generators.trim()) return 'custom';
    const parsed = parseCycleNotation(generators);
    if (parsed.error || !parsed.generators) return 'custom';
    // Try to compute order from generators — not trivial in general,
    // so just show generator count.
    return `custom (${parsed.generators.length} gen${parsed.generators.length !== 1 ? 's' : ''})`;
  }
  const k = (symAxes && symAxes.length) || rank;
  const prefix = symmetry === 'symmetric' ? 'S' : symmetry === 'cyclic' ? 'C' : 'D';
  const order = groupOrder(symmetry, k);
  return `${prefix}${k}`;
}

function badgeOrder(variable) {
  const { symmetry, rank, symAxes } = variable;
  if (symmetry === 'none') return null;
  if (symmetry === 'custom') return null;
  const k = (symAxes && symAxes.length) || rank;
  return groupOrder(symmetry, k);
}

// ---------------------------------------------------------------------------
// Load preset helper
// ---------------------------------------------------------------------------

/**
 * Given an EXAMPLES entry, extract the state shape:
 * { variables, subscriptsStr, outputStr, operandNamesStr }
 */
function presetToState(ex) {
  return {
    variables: ex.variables.map(v => ({
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

// ---------------------------------------------------------------------------
// Build perOpSymmetry for the onCustomExample callback
// ---------------------------------------------------------------------------

/**
 * Map each operand SLOT to its symmetry descriptor.
 *
 * The operand names string (e.g. "T, W") references variables by name.
 * Each slot gets the symmetry of the variable it references, translated
 * into the perOpSymmetry format the algorithm engine expects.
 */
function buildPerOpSymmetry(variables, operandNamesStr, subscriptsStr) {
  const varMap = new Map();
  for (const v of variables) {
    varMap.set(v.name.trim(), v);
  }

  const opNames = operandNamesStr.split(',').map(s => s.trim()).filter(Boolean);
  const subs = subscriptsStr.split(',').map(s => s.trim());

  return opNames.map((name, i) => {
    const v = varMap.get(name);
    if (!v || v.symmetry === 'none') return null;

    const sub = subs[i] || '';
    const allAxes = sub.length;

    if (v.symmetry === 'symmetric') {
      // If ALL axes of this operand are symmetric, return shorthand
      if (!v.symAxes || v.symAxes.length === allAxes) return 'symmetric';
      return { type: 'symmetric', axes: [...v.symAxes] };
    }
    if (v.symmetry === 'cyclic') {
      if (!v.symAxes || v.symAxes.length === allAxes) {
        return { type: 'cyclic', axes: [...Array(allAxes).keys()] };
      }
      return { type: 'cyclic', axes: [...v.symAxes] };
    }
    if (v.symmetry === 'dihedral') {
      if (!v.symAxes || v.symAxes.length === allAxes) {
        return { type: 'dihedral', axes: [...Array(allAxes).keys()] };
      }
      return { type: 'dihedral', axes: [...v.symAxes] };
    }
    if (v.symmetry === 'custom') {
      const parsed = parseCycleNotation(v.generators || '');
      const axes = v.symAxes || [...Array(allAxes).keys()];
      return {
        type: 'custom',
        axes: [...axes],
        generators: parsed.generators || [],
      };
    }

    return null;
  });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ExampleChooser({
  examples, selected, onSelect, example, dimensionN, onDimensionChange, onCustomExample,
}) {
  // ── State ──────────────────────────────────────────────────────────────

  const initial = presetToState(examples[0]);

  const [variables, setVariables] = useState(initial.variables);
  const [subscriptsStr, setSubscriptsStr] = useState(initial.subscriptsStr);
  const [outputStr, setOutputStr] = useState(initial.outputStr);
  const [operandNamesStr, setOperandNamesStr] = useState(initial.operandNamesStr);
  const [activePresetIdx, setActivePresetIdx] = useState(0);

  const [copied, setCopied] = useState(false);

  // ── Load preset ────────────────────────────────────────────────────────

  const loadPreset = useCallback((idx) => {
    const state = presetToState(examples[idx]);
    setVariables(state.variables);
    setSubscriptsStr(state.subscriptsStr);
    setOutputStr(state.outputStr);
    setOperandNamesStr(state.operandNamesStr);
    setActivePresetIdx(idx);
    onSelect(idx);
  }, [examples, onSelect]);

  // ── Mark dirty (no longer a preset) ────────────────────────────────────

  const markCustom = useCallback(() => {
    setActivePresetIdx(CUSTOM_IDX);
  }, []);

  // ── Variable mutations ─────────────────────────────────────────────────

  const updateVar = useCallback((idx, field, value) => {
    setVariables(prev => {
      const next = [...prev];
      next[idx] = { ...next[idx], [field]: value };

      // Side-effects on symmetry change
      if (field === 'symmetry') {
        if (value === 'none') {
          next[idx].symAxes = null;
          next[idx].generators = '';
        } else if (value === 'custom') {
          // Keep symAxes if present, default to all axes
          if (!next[idx].symAxes) {
            next[idx].symAxes = [...Array(next[idx].rank).keys()];
          }
        } else {
          // Named group: default symAxes to all axes
          next[idx].symAxes = [...Array(next[idx].rank).keys()];
          next[idx].generators = '';
        }
      }

      // When rank changes, clamp symAxes
      if (field === 'rank') {
        const newRank = value;
        if (next[idx].symAxes) {
          next[idx].symAxes = next[idx].symAxes.filter(a => a < newRank);
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
    setVariables(prev => {
      const next = [...prev];
      const v = { ...next[varIdx] };
      const axes = new Set(v.symAxes || []);
      if (axes.has(axisIdx)) axes.delete(axisIdx);
      else axes.add(axisIdx);
      v.symAxes = [...axes].sort((a, b) => a - b);
      next[varIdx] = v;
      return next;
    });
    markCustom();
  }, [markCustom]);

  const addVar = useCallback(() => {
    setVariables(prev => [
      ...prev,
      { name: 'A', rank: 2, symmetry: 'none', symAxes: null, generators: '' },
    ]);
    markCustom();
  }, [markCustom]);

  const removeVar = useCallback((idx) => {
    setVariables(prev => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== idx)));
    markCustom();
  }, [markCustom]);

  // ── Expression mutations ───────────────────────────────────────────────

  const handleSubscriptsChange = useCallback((val) => {
    setSubscriptsStr(val);
    markCustom();
  }, [markCustom]);

  const handleOutputChange = useCallback((val) => {
    setOutputStr(val);
    markCustom();
  }, [markCustom]);

  const handleOperandNamesChange = useCallback((val) => {
    setOperandNamesStr(val);
    markCustom();
  }, [markCustom]);

  // ── Variable colors ────────────────────────────────────────────────────

  const varColors = useMemo(() => buildVariableColors(variables), [variables]);

  // ── Validation ─────────────────────────────────────────────────────────

  const validation = useMemo(
    () => validateAll(variables, subscriptsStr, outputStr, operandNamesStr),
    [variables, subscriptsStr, outputStr, operandNamesStr],
  );

  // ── Python code ────────────────────────────────────────────────────────

  const pythonCode = useMemo(
    () => generatePython(variables, subscriptsStr, outputStr, operandNamesStr, dimensionN),
    [variables, subscriptsStr, outputStr, operandNamesStr, dimensionN],
  );

  // ── Copy handler ───────────────────────────────────────────────────────

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(pythonCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [pythonCode]);

  // ── Analyze handler ────────────────────────────────────────────────────

  const handleAnalyze = useCallback(() => {
    if (!validation.valid) return;

    const subs = subscriptsStr.split(',').map(s => s.trim());
    const out = outputStr.trim();
    const opsArr = operandNamesStr.split(',').map(s => s.trim()).filter(Boolean);
    const perOpSymArr = buildPerOpSymmetry(variables, operandNamesStr, subscriptsStr);
    const hasAnySym = perOpSymArr.some(s => s !== null);

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
  }, [validation, variables, subscriptsStr, outputStr, operandNamesStr, activePresetIdx, examples, onCustomExample]);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="example-chooser">

      {/* ── 1. Preset grid ── */}
      <div className="example-grid">
        {examples.map((ex, i) => (
          <button
            key={ex.id}
            className={`example-card ${activePresetIdx === i ? 'active' : ''}`}
            style={{ '--accent': ex.color }}
            onClick={() => loadPreset(i)}
          >
            <div className="example-name">{ex.name}</div>
            <code className="example-formula">{ex.formula}</code>
            <div className="example-group">{ex.expectedGroup}</div>
            <div className="example-desc">{ex.description}</div>
          </button>
        ))}
      </div>

      {/* ── 2. Variable cards ── */}
      <div className="builder-section-label">Variables</div>
      <div className="var-cards">
        {variables.map((v, i) => {
          const vc = varColors[v.name] || {};
          const color = vc.color || '#888';

          return (
            <div
              key={i}
              className="var-card"
              style={{ borderColor: color }}
            >
              <div className="var-card-header">
                {/* Name input */}
                <input
                  className="var-name-input"
                  value={v.name}
                  onChange={e => updateVar(i, 'name', e.target.value)}
                  placeholder="X"
                  maxLength={8}
                />

                {/* Rank stepper */}
                <div className="rank-stepper">
                  <button
                    onClick={() => updateVar(i, 'rank', Math.max(1, v.rank - 1))}
                    disabled={v.rank <= 1}
                  >-</button>
                  <span className="rank-value">{v.rank}</span>
                  <button
                    onClick={() => updateVar(i, 'rank', Math.min(8, v.rank + 1))}
                    disabled={v.rank >= 8}
                  >+</button>
                </div>

                {/* Remove button */}
                <button
                  className="var-remove-btn"
                  onClick={() => removeVar(i)}
                  disabled={variables.length <= 1}
                  title="Remove variable"
                >
                  &#x2715;
                </button>
              </div>

              {/* Symmetry type toggles */}
              <div className="sym-toggles">
                {SYM_TYPES.map(st => (
                  <button
                    key={st}
                    className={`sym-toggle ${v.symmetry === st ? 'active' : ''}`}
                    onClick={() => updateVar(i, 'symmetry', st)}
                  >
                    {SYMMETRY_ICONS[st] ? `${SYMMETRY_ICONS[st]} ` : ''}{SYM_LABELS[st]}
                  </button>
                ))}
              </div>

              {/* Axis chips (for named groups and custom) */}
              {v.symmetry !== 'none' && v.rank > 0 && (
                <div className="axis-chips-row">
                  {Array.from({ length: v.rank }, (_, ai) => {
                    const isSelected = v.symAxes && v.symAxes.includes(ai);
                    return (
                      <button
                        key={ai}
                        className={`axis-chip ${isSelected ? 'selected' : ''}`}
                        onClick={() => toggleAxis(i, ai)}
                        title={`Axis ${ai}`}
                      >
                        {ai}
                      </button>
                    );
                  })}
                </div>
              )}

              {/* Cycle notation input (custom only) */}
              {v.symmetry === 'custom' && (
                <>
                  <input
                    className="gen-input"
                    value={v.generators}
                    onChange={e => updateVar(i, 'generators', e.target.value)}
                    placeholder="(0 1)(2 3), (0 2)(1 3)"
                  />
                  <span className="gen-hint">
                    Cycle notation, comma-separated generators. E.g. <code>(0 1)</code> swaps axes 0 and 1.
                  </span>
                </>
              )}

              {/* Summary badge */}
              <div className="sym-badge" style={{ backgroundColor: color, color: contrastText(color) }}>
                {badgeLabel(v)}
                {badgeOrder(v) != null && (
                  <span className="sym-order"> order {badgeOrder(v)}</span>
                )}
              </div>
            </div>
          );
        })}

        {/* Add variable card */}
        <button className="var-card var-card-add" onClick={addVar}>
          + Add Variable
        </button>
      </div>

      {/* ── 3. Expression panel ── */}
      <div className="builder-section-label">Expression</div>
      <div className="expr-panel">
        <div className="expr-row">
          <span className="expr-chrome">einsum('</span>
          <input
            className={`expr-input ${validation.errors.some(e => e.includes('subscript') || e.includes('Subscript') || e.includes('operand')) ? 'has-error' : ''}`}
            value={subscriptsStr}
            onChange={e => handleSubscriptsChange(e.target.value.toLowerCase())}
            placeholder="e.g. ia,ib"
          />
          <span className="expr-chrome">-&gt;</span>
          <input
            className={`expr-input ${validation.errors.some(e => e.includes('utput')) ? 'has-error' : ''}`}
            value={outputStr}
            onChange={e => handleOutputChange(e.target.value.toLowerCase())}
            placeholder="e.g. ab (empty = scalar)"
          />
          <span className="expr-chrome">',</span>
          <input
            className={`expr-input ${validation.errors.some(e => e.includes('operand') || e.includes('Operand')) ? 'has-error' : ''}`}
            value={operandNamesStr}
            onChange={e => handleOperandNamesChange(e.target.value)}
            placeholder="e.g. X, X"
          />
          <span className="expr-chrome">)</span>
        </div>
        <div className="expr-panel-label">
          subscripts &rarr; output, operands
        </div>
      </div>

      {/* ── 4. Real-time validation errors ── */}
      {validation.errors.length > 0 && (
        <div className="validation-errors">
          {validation.errors.map((err, i) => (
            <div key={i} className="validation-error">{err}</div>
          ))}
        </div>
      )}

      {/* ── Analyze button ── */}
      <button
        className="analyze-btn"
        onClick={handleAnalyze}
        disabled={!validation.valid}
      >
        &#x25B6; Analyze
      </button>

      {/* ── 5. Real-time Python preview ── */}
      <div className="python-preview">
        <div className="python-preview-header">
          <span className="python-preview-label">Python equivalent</span>
          <button className="copy-btn" onClick={handleCopy}>
            {copied ? '\u2713 Copied' : 'Copy'}
          </button>
        </div>
        <PythonHighlight code={pythonCode} />
      </div>

      {/* ── 6. Dimension slider ── */}
      <div className="dimension-slider">
        <label>
          Dimension <strong>n = {dimensionN}</strong>
          <input
            type="range"
            min={2}
            max={20}
            value={dimensionN}
            onChange={e => onDimensionChange(Number(e.target.value))}
          />
        </label>
        <span className="dim-hint">Affects Burnside counts &amp; cost (steps 6-7)</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Python syntax highlighting (kept from previous implementation)
// ---------------------------------------------------------------------------

/** Lightweight Python syntax highlighting via regex -> spans */
function PythonHighlight({ code }) {
  const html = useMemo(() => highlightPython(code), [code]);
  return (
    <pre className="python-code">
      <code dangerouslySetInnerHTML={{ __html: html }} />
    </pre>
  );
}

function highlightPython(code) {
  // Tokenize then reassemble with spans -- avoids regex-on-HTML issues
  const tokens = [];
  const lines = code.split('\n');

  for (const line of lines) {
    let rest = line;

    // Extract comment first
    const commentIdx = rest.indexOf('#');
    let comment = '';
    if (commentIdx >= 0) {
      comment = rest.slice(commentIdx);
      rest = rest.slice(0, commentIdx);
    }

    // Extract strings from the non-comment part
    let i = 0;
    while (i < rest.length) {
      const ch = rest[i];
      if (ch === "'" || ch === '"') {
        // Find matching close
        const close = rest.indexOf(ch, i + 1);
        if (close >= 0) {
          // Push text before string
          if (i > 0) tokens.push({ type: 'code', text: rest.slice(0, i) });
          tokens.push({ type: 'str', text: rest.slice(i, close + 1) });
          rest = rest.slice(close + 1);
          i = 0;
          continue;
        }
      }
      i++;
    }

    // Remaining code
    if (rest) tokens.push({ type: 'code', text: rest });
    if (comment) tokens.push({ type: 'comment', text: comment });
    tokens.push({ type: 'newline' });
  }

  // Render tokens to HTML
  const KEYWORDS = new Set(['import', 'from', 'as', 'for', 'in', 'if', 'else', 'def', 'return',
    'class', 'sum', 'range', 'list', 'True', 'False', 'None']);

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function highlightCode(text) {
    // Single-pass tokenizer to avoid nested span issues
    const re = /(\b\d+\.?\d*\b)|(\b[a-zA-Z_]\w*\b)/g;
    let result = '';
    let last = 0;
    let match;
    while ((match = re.exec(text)) !== null) {
      result += esc(text.slice(last, match.index));
      const num = match[1];
      const word = match[2];
      if (num) {
        result += `<span class="hl-num">${esc(num)}</span>`;
      } else if (KEYWORDS.has(word)) {
        result += `<span class="hl-kw">${esc(word)}</span>`;
      } else {
        // Check if next non-space char is '(' -> function call
        const after = text.slice(re.lastIndex).match(/^\s*\(/);
        if (after) {
          result += `<span class="hl-fn">${esc(word)}</span>`;
        } else {
          result += esc(word);
        }
      }
      last = re.lastIndex;
    }
    result += esc(text.slice(last));
    return result;
  }

  const parts = [];
  for (const tok of tokens) {
    if (tok.type === 'newline') parts.push('\n');
    else if (tok.type === 'str') parts.push(`<span class="hl-str">${esc(tok.text)}</span>`);
    else if (tok.type === 'comment') parts.push(`<span class="hl-cmt">${esc(tok.text)}</span>`);
    else parts.push(highlightCode(tok.text));
  }

  // Remove trailing newline
  const result = parts.join('');
  return result.endsWith('\n') ? result.slice(0, -1) : result;
}
