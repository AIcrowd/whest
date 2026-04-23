import React, { useState, useEffect, useMemo } from 'react';
import { analyzeExample } from '../engine/pipeline.js';
import { encodePlaygroundState, decodePlaygroundState } from '../lib/urlState.js';
import LabelClusterPanel from './LabelClusterPanel.wip.jsx';
import RegimeTrace from './RegimeTrace.jsx';
import Latex from './Latex.jsx';

const DEFAULT_STATE = {
  subscripts: 'ij,jk',
  output: 'ik',
  operands: [
    { name: 'A', rank: 2, symmetry: 'none' },
    { name: 'B', rank: 2, symmetry: 'none' },
  ],
  labelSizes: {},
};

const SYMMETRY_OPTIONS = ['none', 'symmetric'];

function parseStateFromLocation() {
  if (typeof window === 'undefined') return null;
  const params = new URLSearchParams(window.location.search);
  const raw = params.get('state');
  if (!raw) return null;
  return decodePlaygroundState(raw);
}

export default function Playground() {
  const [state, setState] = useState(() => parseStateFromLocation() || DEFAULT_STATE);
  const [defaultSize, setDefaultSize] = useState(5);

  // Push state into URL query string on change.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    params.set('state', encodePlaygroundState(state));
    const next = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState(null, '', next);
  }, [state]);

  const example = useMemo(() => ({
    id: 'playground',
    expression: {
      subscripts: state.subscripts,
      output: state.output,
      operandNames: state.operands.map((o) => o.name).join(', '),
    },
    variables: state.operands,
    labelSizes: state.labelSizes,
  }), [state]);

  const analysis = useMemo(() => {
    try {
      return analyzeExample(example, defaultSize);
    } catch (err) {
      // Parse or build errors surface here; show null so the UI degrades.
      return null;
    }
  }, [example, defaultSize]);

  const clusters = analysis?.clusters ?? [];

  function updateSubscripts(value) {
    setState((s) => ({ ...s, subscripts: value }));
  }

  function updateOutput(value) {
    setState((s) => ({ ...s, output: value }));
  }

  function updateOperandName(idx, name) {
    setState((s) => ({
      ...s,
      operands: s.operands.map((o, i) => (i === idx ? { ...o, name } : o)),
    }));
  }

  function updateOperandRank(idx, rank) {
    setState((s) => ({
      ...s,
      operands: s.operands.map((o, i) => (i === idx ? { ...o, rank: Math.max(1, Math.floor(rank || 1)) } : o)),
    }));
  }

  function updateOperandSymmetry(idx, symmetry) {
    setState((s) => ({
      ...s,
      operands: s.operands.map((o, i) => (i === idx ? { ...o, symmetry } : o)),
    }));
  }

  function setClusterSize(id, size) {
    setState((s) => ({ ...s, labelSizes: { ...s.labelSizes, [id]: size } }));
  }

  function setAllClusterSizes(n) {
    setDefaultSize(n);
    setState((s) => {
      const next = {};
      for (const k of Object.keys(s.labelSizes)) next[k] = n;
      return { ...s, labelSizes: next };
    });
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Playground
        </div>
        <div className="text-[11px] text-muted-foreground">
          State mirrors the URL — paste to share.
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <label className="text-sm">
          Subscripts
          <input
            id="playground-subscripts"
            value={state.subscripts}
            onChange={(e) => updateSubscripts(e.target.value)}
            className="mt-1 w-full rounded border border-gray-300 px-2 py-1 font-mono text-sm"
            spellCheck={false}
          />
        </label>
        <label className="text-sm">
          Output
          <input
            value={state.output}
            onChange={(e) => updateOutput(e.target.value)}
            className="mt-1 w-full rounded border border-gray-300 px-2 py-1 font-mono text-sm"
            spellCheck={false}
          />
        </label>
      </div>

      <div className="mt-4 space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Operands
        </div>
        {state.operands.map((operand, idx) => (
          <div key={idx} className="flex flex-wrap items-center gap-2 text-sm">
            <input
              value={operand.name}
              onChange={(e) => updateOperandName(idx, e.target.value)}
              className="w-20 rounded border border-gray-300 px-2 py-1 font-mono"
              aria-label={`Operand ${idx} name`}
            />
            <span className="text-muted-foreground">rank</span>
            <input
              type="number"
              min={1}
              max={8}
              value={operand.rank}
              onChange={(e) => updateOperandRank(idx, e.target.value)}
              className="w-14 rounded border border-gray-300 px-2 py-1 text-right"
              aria-label={`Operand ${idx} rank`}
            />
            <span className="text-muted-foreground">symmetry</span>
            <select
              value={operand.symmetry}
              onChange={(e) => updateOperandSymmetry(idx, e.target.value)}
              className="rounded border border-gray-300 px-2 py-1"
              aria-label={`Operand ${idx} symmetry`}
            >
              {SYMMETRY_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {clusters.length > 0 ? (
        <div className="mt-4">
          <LabelClusterPanel
            clusters={clusters.map((c) => ({ ...c, size: state.labelSizes[c.id] ?? c.size }))}
            onSizeChange={setClusterSize}
            vLabels={analysis?.symmetry?.vLabels || []}
            onSetAll={setAllClusterSizes}
          />
        </div>
      ) : null}

      {analysis?.componentData?.components?.length ? (
        <div className="mt-4 space-y-3">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Components
          </div>
          {analysis.componentData.components.map((comp, i) => (
            <div key={`pg-comp-${i}`} className="rounded border border-gray-200 bg-gray-50/40 p-3">
              <div className="mb-1 text-sm font-medium">
                Component {i + 1}: {comp.labels.join(', ') || '∅'}
              </div>
              <div className="text-xs text-gray-600">
                Regime <code className="font-mono">{comp.accumulation?.regimeId ?? '—'}</code> · Count <code className="font-mono">{comp.accumulation?.count?.toLocaleString() ?? '—'}</code>
              </div>
              {comp.accumulation?.latex ? (
                <div className="mt-2 text-xs">
                  <Latex math={comp.accumulation.latex} />
                </div>
              ) : null}
              {comp.accumulation?.trace ? (
                <div className="mt-2">
                  <RegimeTrace trace={comp.accumulation.trace} />
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-4 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          Unable to analyze the current expression. Check that subscripts and operand shapes are valid.
        </div>
      )}
    </div>
  );
}
