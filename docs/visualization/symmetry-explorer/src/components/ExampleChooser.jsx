import { useState, useCallback, useMemo } from 'react';

const CUSTOM_IDX = -1;

/** Helper: generate the Reynolds operator symmetrization snippet. */
function reynoldsSnippet(varName, shape, groupExpr) {
  return [
    `_data = numpy.random.randn(${shape})`,
    `_group = ${groupExpr}`,
    `_data = sum(numpy.transpose(_data, g.array_form) for g in _group.elements()) / _group.order()`,
    `${varName} = me.as_symmetric(_data, symmetry=${groupExpr.replace(/^_group$/, groupExpr)})`,
  ];
}

/** Pure function — generates Python code for ANY example object (preset or custom). */
function generatePythonCode(example, dimensionN) {
  const { subscripts, output, operandNames, perOpSymmetry } = example;
  const n = dimensionN;
  const lines = [];
  lines.push('import numpy');
  lines.push('import mechestim as me');
  lines.push('from mechestim import Cycle, Permutation, PermutationGroup');
  lines.push('');
  lines.push(`n = ${n}`);

  // Deduplicate variable definitions (same name = same object)
  const defined = new Set();
  for (let i = 0; i < subscripts.length; i++) {
    const name = operandNames[i];
    if (defined.has(name)) continue;
    defined.add(name);

    const sub = subscripts[i];
    const rank = sub.length;
    const shape = Array(rank).fill('n').join(', ');

    // Determine per-operand symmetry
    const opSym = Array.isArray(perOpSymmetry) ? perOpSymmetry[i] : perOpSymmetry;

    if (!opSym) {
      lines.push(`${name} = numpy.random.randn(${shape})`);
    } else if (opSym === 'symmetric') {
      if (rank === 2) {
        lines.push(`# ${name}: symmetric matrix (S2)`);
        lines.push(`_data = numpy.random.randn(${shape})`);
        lines.push(`${name} = me.as_symmetric((_data + _data.T) / 2, symmetric_axes=(0, 1))`);
      } else {
        const axes = [...Array(rank).keys()].join(', ');
        lines.push(`# ${name}: fully symmetric tensor (S${rank}) via Reynolds operator`);
        lines.push(`_group = PermutationGroup.symmetric(${rank}, axes=(${axes}))`);
        lines.push(`_data = numpy.random.randn(${shape})`);
        lines.push(`_data = sum(numpy.transpose(_data, g.array_form) for g in _group.elements()) / _group.order()`);
        lines.push(`${name} = me.as_symmetric(_data, symmetry=_group)`);
      }
    } else if (opSym === 'cyclic') {
      const axes = [...Array(rank).keys()].join(', ');
      lines.push(`# ${name}: cyclic symmetry (C${rank}) — T[i,j,k] = T[j,k,i] but T[i,j,k] ≠ T[j,i,k]`);
      lines.push(`_group = PermutationGroup.cyclic(${rank}, axes=(${axes}))`);
      lines.push(`_data = numpy.random.randn(${shape})`);
      lines.push(`_data = sum(numpy.transpose(_data, g.array_form) for g in _group.elements()) / _group.order()`);
      lines.push(`${name} = me.as_symmetric(_data, symmetry=_group)`);
    } else if (opSym === 'dihedral') {
      const axes = [...Array(rank).keys()].join(', ');
      lines.push(`# ${name}: dihedral symmetry (D${rank}) — rotations + reflections`);
      lines.push(`_group = PermutationGroup.dihedral(${rank}, axes=(${axes}))`);
      lines.push(`_data = numpy.random.randn(${shape})`);
      lines.push(`_data = sum(numpy.transpose(_data, g.array_form) for g in _group.elements()) / _group.order()`);
      lines.push(`${name} = me.as_symmetric(_data, symmetry=_group)`);
    } else if (opSym === 'block-swap') {
      // Block swap: (0,1) ↔ (2,3) via Cycle(0,2)(1,3)
      lines.push(`# ${name}: block swap — axes (0,1) swap with (2,3) as a unit`);
      lines.push(`_group = PermutationGroup(Permutation(Cycle(0, 2)(1, 3)), axes=(0, 1, 2, 3))`);
      lines.push(`_data = numpy.random.randn(${shape})`);
      lines.push(`_data = sum(numpy.transpose(_data, g.array_form) for g in _group.elements()) / _group.order()`);
      lines.push(`${name} = me.as_symmetric(_data, symmetry=_group)`);
    }
  }

  lines.push('');

  // For declared-symmetry examples (single tensor, no einsum), show inspection
  if (example.declared) {
    lines.push(`# Inspect the symmetry`);
    lines.push(`print(f"Group order: {${operandNames[0]}.symmetry_info.groups[0].order()}")`);
    lines.push(`print(f"Is full symmetric: {${operandNames[0]}.symmetry_info.groups[0].is_symmetric()}")`);
    lines.push(`print(f"Unique elements: {${operandNames[0]}.symmetry_info.unique_elements}")`);
    lines.push(`print(f"Dense elements:  {${operandNames[0]}.size}")`);
  } else {
    const subsStr = subscripts.join(',');
    const args = operandNames.join(', ');
    lines.push(`# Symmetry is detected automatically from identical operands`);
    lines.push(`path, info = me.einsum_path('${subsStr}->${output}', ${args})`);
    lines.push(`print(info)`);
  }

  return lines.join('\n');
}

/** Pure function — generates richer Python code for custom builder examples,
 *  reflecting user-chosen symmetry types and axis selections. */
function generateCustomPythonCode(variables, outputStr, dimensionN) {
  const n = dimensionN;
  const lines = [];
  lines.push('import numpy');
  lines.push('import mechestim as me');
  lines.push('from mechestim import Cycle, Permutation, PermutationGroup');
  lines.push('');
  lines.push(`n = ${n}`);

  // Deduplicate variable definitions (same name = same object)
  const defined = new Set();
  for (const v of variables) {
    const name = v.name.trim() || 'X';
    if (defined.has(name)) continue;
    defined.add(name);

    const rank = v.subscript.length || 2;
    const shape = Array(rank).fill('n').join(', ');

    if (v.symmetry === 'none') {
      lines.push(`${name} = numpy.random.randn(${shape})`);
    } else if (v.symmetry === 'custom') {
      // Parse cycle notation string like "(0 1)(2 3)" into valid Python
      const gens = v.generators.trim() || '(0, 1)';
      // Convert "(0 1)(2 3)" → "Cycle(0, 1)(2, 3)"
      const pyGens = gens.replace(/\((\d[\d\s,]*)\)/g, (_, inner) => {
        const nums = inner.trim().split(/[\s,]+/).join(', ');
        return `(${nums})`;
      });
      // If it starts with "(", wrap in Cycle
      const genExpr = pyGens.startsWith('(') ? `Cycle${pyGens}` : pyGens;
      lines.push(`# ${name}: custom symmetry group`);
      lines.push(`_group = PermutationGroup(Permutation(${genExpr}), axes=(${[...Array(rank).keys()].join(', ')}))`);
      lines.push(`_data = numpy.random.randn(${shape})`);
      lines.push(`_data = sum(numpy.transpose(_data, g.array_form) for g in _group.elements()) / _group.order()`);
      lines.push(`${name} = me.as_symmetric(_data, symmetry=_group)`);
    } else {
      const groupFn = v.symmetry === 'symmetric' ? 'symmetric'
        : v.symmetry === 'cyclic' ? 'cyclic' : 'dihedral';
      const axesArr = v.symAxes || [...Array(rank).keys()];
      const k = axesArr.length;

      if (axesArr.length === rank && v.symmetry === 'symmetric' && rank === 2) {
        lines.push(`# ${name}: symmetric matrix (S2)`);
        lines.push(`_data = numpy.random.randn(${shape})`);
        lines.push(`${name} = me.as_symmetric((_data + _data.T) / 2, symmetric_axes=(0, 1))`);
      } else {
        const axes = axesArr.join(', ');
        lines.push(`# ${name}: ${groupFn} symmetry on axes (${axes})`);
        lines.push(`_group = PermutationGroup.${groupFn}(${k}, axes=(${axes}))`);
        lines.push(`_data = numpy.random.randn(${shape})`);
        lines.push(`_data = sum(numpy.transpose(_data, g.array_form) for g in _group.elements()) / _group.order()`);
        lines.push(`${name} = me.as_symmetric(_data, symmetry=_group)`);
      }
    }
  }

  lines.push('');
  const subs = variables.map(v => v.subscript).join(',');
  const out = outputStr.trim() || 'ab';
  const args = variables.map(v => v.name.trim() || 'X').join(', ');
  lines.push(`path, info = me.einsum_path('${subs}->${out}', ${args})`);
  lines.push(`print(info)`);

  return lines.join('\n');
}

const SYM_HINTS = {
  none: 'No per-operand symmetry — each axis is independent',
  symmetric: 'S_k: full symmetric group — all axis permutations',
  cyclic: 'C_k: cyclic group — only cyclic rotations of axes',
  dihedral: 'D_k: dihedral group — cyclic rotations + reflections',
  custom: 'Custom generators in cycle notation — e.g. (0 1)(2 3)',
};

const SYM_LABELS = {
  symmetric: 'S',
  cyclic: 'C',
  dihedral: 'D',
};

export default function ExampleChooser({
  examples, selected, onSelect, example, dimensionN, onDimensionChange, onCustomExample
}) {
  const isCustom = selected === CUSTOM_IDX;

  const [variables, setVariables] = useState([
    { name: 'X', subscript: 'ia', symmetry: 'none', symAxes: null, generators: '' },
    { name: 'X', subscript: 'ib', symmetry: 'none', symAxes: null, generators: '' },
  ]);
  const [outputStr, setOutputStr] = useState('ab');
  const [error, setError] = useState(null);

  const updateVar = useCallback((idx, field, value) => {
    setVariables(prev => {
      const next = [...prev];
      next[idx] = { ...next[idx], [field]: value };
      if (field === 'symmetry' && value !== 'none' && value !== 'custom') {
        const sub = next[idx].subscript;
        next[idx].symAxes = sub ? [...sub].map((_, i) => i) : [];
        next[idx].generators = '';
      }
      if (field === 'symmetry' && value === 'none') {
        next[idx].symAxes = null;
        next[idx].generators = '';
      }
      if (field === 'symmetry' && value === 'custom') {
        next[idx].symAxes = null;
      }
      if (field === 'subscript' && next[idx].symmetry !== 'none' && next[idx].symmetry !== 'custom') {
        next[idx].symAxes = value ? [...value].map((_, i) => i) : [];
      }
      return next;
    });
    setError(null);
  }, []);

  const toggleAxis = useCallback((varIdx, axisIdx) => {
    setVariables(prev => {
      const next = [...prev];
      const v = { ...next[varIdx] };
      const axes = new Set(v.symAxes || []);
      if (axes.has(axisIdx)) axes.delete(axisIdx);
      else axes.add(axisIdx);
      v.symAxes = [...axes].sort();
      next[varIdx] = v;
      return next;
    });
    setError(null);
  }, []);

  const addVar = useCallback(() => {
    setVariables(prev => [...prev, { name: 'A', subscript: '', symmetry: 'none', symAxes: null, generators: '' }]);
    setError(null);
  }, []);

  const removeVar = useCallback((idx) => {
    setVariables(prev => prev.length <= 2 ? prev : prev.filter((_, i) => i !== idx));
    setError(null);
  }, []);

  const handleAnalyze = useCallback(() => {
    for (let i = 0; i < variables.length; i++) {
      const v = variables[i];
      if (!v.name.trim()) { setError(`Variable ${i + 1}: name is empty`); return; }
      if (!v.subscript.trim()) { setError(`Variable ${i + 1}: subscript is empty`); return; }
      if (!/^[a-z]+$/.test(v.subscript)) {
        setError(`Variable ${i + 1}: subscript must be lowercase letters (got "${v.subscript}")`);
        return;
      }
      if (v.symmetry !== 'none' && v.symmetry !== 'custom' && (!v.symAxes || v.symAxes.length < 2)) {
        setError(`Variable ${i + 1}: symmetry requires at least 2 axes selected`);
        return;
      }
      if (v.symmetry === 'custom' && !v.generators.trim()) {
        setError(`Variable ${i + 1}: enter at least one generator in cycle notation`);
        return;
      }
    }

    const allInputLabels = new Set(variables.flatMap(v => [...v.subscript]));
    const outLabels = outputStr.trim();
    if (!outLabels) { setError('Output subscript is empty'); return; }
    if (!/^[a-z]+$/.test(outLabels)) { setError('Output must be lowercase letters'); return; }
    for (const ch of outLabels) {
      if (!allInputLabels.has(ch)) {
        setError(`Output label "${ch}" not found in any input subscript`);
        return;
      }
    }

    const subscripts = variables.map(v => v.subscript);
    const operandNames = variables.map(v => v.name.trim());

    const perOpSymmetry = variables.map(v => {
      if (v.symmetry === 'none') return null;
      if (v.symmetry === 'custom') {
        // Parse generators — for graph construction, custom generators collapse
        // all axes that appear in any cycle together
        return 'symmetric'; // conservative: collapse all axes
      }
      const allAxes = v.subscript.length;
      if (v.symAxes && v.symAxes.length === allAxes) return 'symmetric';
      return { type: v.symmetry, axes: v.symAxes };
    });
    const hasAnySym = perOpSymmetry.some(s => s !== null);

    const subsStr = subscripts.join(',') + '→' + outLabels;
    const argsStr = operandNames.join(', ');
    const formula = `einsum('${subsStr}', ${argsStr})`;

    const customExample = {
      id: 'custom',
      name: 'Custom',
      formula,
      subscripts,
      output: outLabels,
      operandNames,
      perOpSymmetry: hasAnySym ? perOpSymmetry : null,
      description: 'User-defined expression',
      expectedGroup: '',
      color: '#7C3AED',
    };

    setError(null);
    onCustomExample(customExample);
  }, [variables, outputStr, onCustomExample]);

  const autoOutput = useCallback(() => {
    const counts = {};
    for (const v of variables) {
      for (const ch of v.subscript) {
        counts[ch] = (counts[ch] || 0) + 1;
      }
    }
    const out = Object.entries(counts)
      .filter(([, c]) => c === 1)
      .map(([ch]) => ch)
      .sort()
      .join('');
    setOutputStr(out);
  }, [variables]);

  // Generate equivalent Python code — uses dimensionN for concrete shapes.
  // For custom examples: uses the richer per-variable generator.
  // For preset examples: uses the example object directly.
  const pythonCode = useMemo(() => {
    if (isCustom) {
      return generateCustomPythonCode(variables, outputStr, dimensionN);
    }
    if (!example) return '';
    return generatePythonCode(example, dimensionN);
  }, [isCustom, example, variables, outputStr, dimensionN]);

  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(pythonCode).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [pythonCode]);

  return (
    <div className="example-chooser">
      <div className="example-grid">
        {examples.map((ex, i) => (
          <button
            key={ex.id}
            className={`example-card ${selected === i ? 'active' : ''}`}
            style={{ '--accent': ex.color }}
            onClick={() => onSelect(i)}
          >
            <div className="example-name">{ex.name}</div>
            <code className="example-formula">{ex.formula}</code>
            <div className="example-group">{ex.expectedGroup}</div>
            <div className="example-desc">{ex.description}</div>
          </button>
        ))}

        <button
          className={`example-card example-card-custom ${isCustom ? 'active' : ''}`}
          style={{ '--accent': '#7C3AED' }}
          onClick={() => onSelect(CUSTOM_IDX)}
        >
          <div className="example-name">+ Custom</div>
          <div className="example-desc">Define your own einsum expression and operand symmetries</div>
        </button>
      </div>

      {isCustom && (
        <div className="custom-builder">
          <h4 className="builder-heading">Define Variables</h4>

          <div className="var-list">
            {variables.map((v, i) => (
              <div key={i} className="var-card">
                <div className="var-row">
                  <div className="var-field">
                    <span className="var-field-label">Name</span>
                    <input
                      className="var-input var-name"
                      value={v.name}
                      onChange={e => updateVar(i, 'name', e.target.value)}
                      placeholder="X"
                      maxLength={8}
                    />
                  </div>
                  <div className="var-field var-field-grow">
                    <span className="var-field-label">Subscript</span>
                    <input
                      className="var-input var-subscript"
                      value={v.subscript}
                      onChange={e => updateVar(i, 'subscript', e.target.value.toLowerCase())}
                      placeholder="ij"
                      maxLength={12}
                    />
                  </div>
                  <div className="var-field">
                    <span className="var-field-label">Symmetry</span>
                    <select
                      className="var-select"
                      value={v.symmetry}
                      onChange={e => updateVar(i, 'symmetry', e.target.value)}
                      title={SYM_HINTS[v.symmetry]}
                    >
                      <option value="none">dense</option>
                      <option value="symmetric">S_k (symmetric)</option>
                      <option value="cyclic">C_k (cyclic)</option>
                      <option value="dihedral">D_k (dihedral)</option>
                      <option value="custom">Custom generators</option>
                    </select>
                  </div>
                  <button
                    className="var-remove"
                    onClick={() => removeVar(i)}
                    disabled={variables.length <= 2}
                    title="Remove variable"
                  >
                    ✕
                  </button>
                </div>

                {/* Axis selector for named groups */}
                {v.symmetry !== 'none' && v.symmetry !== 'custom' && v.subscript.length > 0 && (
                  <div className="axis-selector">
                    <span className="axis-label">Symmetric axes:</span>
                    <div className="axis-chips">
                      {[...v.subscript].map((ch, ai) => {
                        const active = v.symAxes && v.symAxes.includes(ai);
                        return (
                          <button
                            key={ai}
                            className={`axis-chip ${active ? 'axis-active' : ''}`}
                            onClick={() => toggleAxis(i, ai)}
                            title={`Axis ${ai}: label "${ch}"`}
                          >
                            {ch}
                          </button>
                        );
                      })}
                    </div>
                    <span className="axis-hint">
                      {v.symAxes && v.symAxes.length >= 2
                        ? `${SYM_LABELS[v.symmetry]}(${v.symAxes.map(a => v.subscript[a]).join(',')})`
                        : 'select ≥ 2 axes'}
                    </span>
                  </div>
                )}

                {/* Generator input for custom groups */}
                {v.symmetry === 'custom' && (
                  <div className="generator-input">
                    <span className="axis-label">Generators (cycle notation):</span>
                    <input
                      className="var-input generator-text"
                      value={v.generators}
                      onChange={e => updateVar(i, 'generators', e.target.value)}
                      placeholder="(0 1)(2 3), (0 2)(1 3)"
                    />
                    <span className="axis-hint">
                      Use 0-indexed axis positions, e.g. <code>(0 1)</code> swaps first two axes.
                      Separate multiple generators with commas.
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>

          <button className="add-var-btn" onClick={addVar}>+ Add Variable</button>

          <div className="output-row">
            <label className="output-label">Output →</label>
            <input
              className="var-input var-output"
              value={outputStr}
              onChange={e => setOutputStr(e.target.value.toLowerCase())}
              placeholder="ab"
              maxLength={12}
            />
            <button className="auto-btn" onClick={autoOutput} title="Auto-detect from subscripts">
              auto
            </button>
          </div>

          {error && <div className="builder-error">{error}</div>}

          <button className="analyze-btn" onClick={handleAnalyze}>
            ▶ Analyze
          </button>
        </div>
      )}

      {/* Python code preview — shown for ALL examples (preset and custom) */}
      {example && pythonCode && (
        <div className="python-preview">
          <div className="python-preview-header">
            <span className="python-preview-label">Python equivalent</span>
            <button className="copy-btn" onClick={handleCopy}>
              {copied ? '✓ Copied' : 'Copy'}
            </button>
          </div>
          <PythonHighlight code={pythonCode} />
        </div>
      )}

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
        <span className="dim-hint">Affects Burnside counts & cost (steps 6-7)</span>
      </div>
    </div>
  );
}

/** Lightweight Python syntax highlighting via regex → spans */
function PythonHighlight({ code }) {
  const html = useMemo(() => highlightPython(code), [code]);
  return (
    <pre className="python-code">
      <code dangerouslySetInnerHTML={{ __html: html }} />
    </pre>
  );
}

function highlightPython(code) {
  // Tokenize then reassemble with spans — avoids regex-on-HTML issues
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
        // Check if next non-space char is '(' → function call
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
