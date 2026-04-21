import { useMemo, useState } from 'react';
import ExplorerModal from './ExplorerModal.jsx';
import InlineMathText from './InlineMathText.jsx';
import Latex from './Latex.jsx';
import { explorerThemeColor, explorerThemeTint } from '../lib/explorerTheme.js';
import { getActiveExplorerThemeId, notationLatex } from '../lib/notationSystem.js';

const INITIAL_ROW_LIMIT = 10;

export default function WreathStructureView({ analysis, example }) {
  const explorerThemeId = getActiveExplorerThemeId();
  const symmetry = analysis?.symmetry;
  const wreathElements = symmetry?.wreathElements;
  const [showAllRows, setShowAllRows] = useState(false);
  const opNames = example?.operandNames
    || example?.expression?.operandNames?.split(',').map((s) => s.trim())
    || [];
  const variables = example?.variables || [];

  const factors = useMemo(() => {
    if (!symmetry) return [];
    const { identicalGroups = [] } = symmetry;
    return identicalGroups.map((group) => {
      const firstPos = group[0];
      const name = opNames[firstPos];
      const variable = variables.find((v) => v.name === name);
      const rank = example?.subscripts?.[firstPos]?.length
        ?? example?.expression?.subscripts?.split(',')[firstPos]?.trim().length
        ?? 0;
      const m = group.length;
      return {
        name,
        rank,
        m,
        symmetryLabel: describeSymmetry(variable, rank),
      };
    });
  }, [symmetry, opNames, variables, example]);

  if (!wreathElements || wreathElements.length === 0) return null;

  const validCount = wreathElements.filter((element) => element.classification === 'valid').length;
  const identityOnlyCount = wreathElements.filter((element) => element.classification === 'matrix-preserving').length;
  const rejectedCount = wreathElements.filter((element) => element.classification === 'rejected').length;
  const hiddenCount = Math.max(0, wreathElements.length - INITIAL_ROW_LIMIT);
  const visibleElements = wreathElements.slice(0, INITIAL_ROW_LIMIT);
  const summaryPillClass = 'inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] font-sans';

  const formulaLatex = factors.length === 0
    ? `${notationLatex('g_wreath')} = \\{e\\}`
    : `${notationLatex('g_wreath')} = ${factors.map((factor) => `(${factor.symmetryLabel} \\wr S_{${factor.m}})`).join(' \\times ')}`;

  return (
    <div className="bg-white p-4">
      <p className="font-serif text-[16px] leading-[1.75] text-gray-700">
        <InlineMathText>
          {`We build $${notationLatex('g_wreath')} = \\prod_i (${notationLatex('h_family')} \\wr S_{m_i})$ because it is the search space of row moves worth testing: declared axis symmetries inside each repeated-operand family together with permutations of identical copies. For each $${notationLatex('sigma_row_move')}$, we then ask whether there is a matching relabeling $${notationLatex('pi_relabeling')}$ of the labels.`}
        </InlineMathText>
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <div className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-[12px] text-gray-600">
          <Latex math={formulaLatex} />
        </div>
        {factors.length > 0 ? (
          factors.map((factor, index) => (
            <div
              key={`${factor.name ?? 'operand'}-${index}`}
              className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-[12px] text-gray-600"
            >
              <span className="font-semibold text-gray-900">{factor.name || `operand ${index + 1}`}</span>
              <span className="ml-2">rank {factor.rank}</span>
              <span className="ml-2">H = <Latex math={factor.symmetryLabel} /></span>
              <span className="ml-2">m = {factor.m}</span>
            </div>
          ))
        ) : null}
      </div>

      <WreathElementTable elements={visibleElements} />

      {hiddenCount > 0 ? (
        <div className="mt-3 flex items-center justify-between gap-3">
          <p className="text-[12px] leading-5 text-gray-500">
            Showing the first {INITIAL_ROW_LIMIT} row moves from a total of {wreathElements.length}.
          </p>
          <button
            type="button"
            onClick={() => setShowAllRows(true)}
            className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-[12px] font-medium text-gray-700 transition-colors hover:border-gray-300 hover:bg-gray-100"
          >
            {`Click to see ${hiddenCount} more rows`}
          </button>
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span
          className={`${summaryPillClass} border-gray-200 bg-white text-gray-700`}
          style={{ letterSpacing: '0.14em' }}
        >
          <Latex math={`|${notationLatex('g_wreath')}| = ${wreathElements.length}`} />
        </span>
        <span
          className={summaryPillClass}
          style={{
            borderColor: explorerThemeTint(explorerThemeId, 'hero', 0.24),
            background: explorerThemeTint(explorerThemeId, 'hero', 0.12),
            color: explorerThemeColor(explorerThemeId, 'hero'),
          }}
        >
          kept in G: {validCount}
        </span>
        <span
          className={summaryPillClass}
          style={{
            borderColor: explorerThemeTint(explorerThemeId, 'summedSide', 0.22),
            background: explorerThemeTint(explorerThemeId, 'summedSide', 0.12),
            color: explorerThemeColor(explorerThemeId, 'summedSide'),
          }}
        >
          identity only: {identityOnlyCount}
        </span>
        <span
          className={summaryPillClass}
          style={{
            borderColor: explorerThemeTint(explorerThemeId, 'editorialAccent', 0.24),
            background: explorerThemeTint(explorerThemeId, 'editorialAccent', 0.12),
            color: explorerThemeColor(explorerThemeId, 'editorialAccent'),
          }}
        >
          no matching relabeling: {rejectedCount}
        </span>
      </div>

      <ExplorerModal
        title={<Latex math={`\\text{All row moves in } ${notationLatex('g_wreath')}`} />}
        titleId="wreath-rows-title"
        open={showAllRows}
        onClose={() => setShowAllRows(false)}
        width="min(1180px, 96vw)"
      >
        <p className="mb-4 font-serif text-[15px] leading-7 text-gray-700">
          This modal shows every row move in the wreath-product search space, together with the matching relabeling test that decides whether the move is kept in the detected symmetry group.
        </p>
        <WreathElementTable elements={wreathElements} />
      </ExplorerModal>
    </div>
  );
}

function WreathElementTable({ elements }) {
  return (
    <div className="mt-4 overflow-x-auto bg-white">
      <table className="min-w-full border-collapse text-[12px]">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50 text-gray-900">
            <th className="px-3 py-2 text-center font-semibold">
              <Latex math={`${notationLatex('sigma_row_move')}\\text{ row move}`} />
            </th>
            <th className="px-3 py-2 text-center font-semibold">Factor decomposition</th>
            <th className="px-3 py-2 text-center font-semibold">Row action on M</th>
            <th className="px-3 py-2 text-center font-semibold">matching relabeling <Latex math={String.raw`\pi`} /></th>
            <th className="px-3 py-2 text-center font-semibold">Outcome</th>
          </tr>
        </thead>
        <tbody>
          {elements.map((element, index) => (
            <WreathElementRow key={element.id} element={element} index={index} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function WreathElementRow({ element, index }) {
  const explorerThemeId = getActiveExplorerThemeId();
  const classLabel = {
    valid: '✓ kept in G',
    'matrix-preserving': 'identity only',
    rejected: '✗ no matching relabeling',
  }[element.classification];

  const rowTone = {
    valid: {
      pillBorder: explorerThemeTint(explorerThemeId, 'hero', 0.24),
      pillBg: explorerThemeTint(explorerThemeId, 'hero', 0.12),
      pillText: explorerThemeColor(explorerThemeId, 'hero'),
      outcome: explorerThemeColor(explorerThemeId, 'hero'),
    },
    'matrix-preserving': {
      pillBorder: explorerThemeTint(explorerThemeId, 'summedSide', 0.22),
      pillBg: explorerThemeTint(explorerThemeId, 'summedSide', 0.12),
      pillText: explorerThemeColor(explorerThemeId, 'summedSide'),
      outcome: explorerThemeColor(explorerThemeId, 'summedSide'),
    },
    rejected: {
      pillBorder: explorerThemeTint(explorerThemeId, 'editorialAccent', 0.24),
      pillBg: explorerThemeTint(explorerThemeId, 'editorialAccent', 0.12),
      pillText: explorerThemeColor(explorerThemeId, 'editorialAccent'),
      outcome: explorerThemeColor(explorerThemeId, 'editorialAccent'),
    },
  }[element.classification];

  const matrixEffect = element.matrixPreserving
    ? String.raw`\sigma(M) = M`
    : String.raw`\sigma(M) \ne M`;
  const piStr = element.derivePiResult === null ? '—' : piCycleNotation(element.derivePiResult);
  const factorStr = element.factorization
    ? element.factorization.map((factor) => {
        const baseStr = factor.baseTuple.map((perm) => (perm.isIdentity ? 'e' : perm.cycleNotation())).join(',');
        const topStr = factor.topPerm
          .map((value, factorIndex) => (value === factorIndex ? null : `${factorIndex}→${value}`))
          .filter(Boolean)
          .join(',') || 'id';
        return `(${baseStr}; ${topStr})`;
      }).join(' × ')
    : '';

  return (
    <tr className="border-b border-gray-200 last:border-b-0 bg-white text-gray-900">
      <td className="px-3 py-2">
        <span
          className="inline-flex items-center rounded-full border px-2 py-1 font-mono text-[11px]"
          style={{
            borderColor: rowTone.pillBorder,
            background: rowTone.pillBg,
            color: rowTone.pillText,
          }}
        >
          <Latex math={`${notationLatex('sigma_row_move')}_{${index}}`} />
        </span>
      </td>
      <td className="px-3 py-2 font-mono text-[11px]">{factorStr}</td>
      <td className="px-3 py-2 font-mono text-[11px]">
        <span style={{ color: explorerThemeColor(explorerThemeId, 'heroMuted') }}>
          <Latex math={matrixEffect} />
        </span>
      </td>
      <td className="px-3 py-2 font-mono text-[11px]">{piStr}</td>
      <td className="px-3 py-2 font-medium" style={{ color: rowTone.outcome }}>{classLabel}</td>
    </tr>
  );
}

function describeSymmetry(variable, rank) {
  if (!variable || variable.symmetry === 'none') return '\\{e\\}';
  const axes = variable.symAxes || Array.from({ length: rank }, (_, i) => i);
  const k = axes.length;
  if (variable.symmetry === 'symmetric') return `S_${k}`;
  if (variable.symmetry === 'cyclic') return `C_${k}`;
  if (variable.symmetry === 'dihedral') return `D_${k}`;
  if (variable.symmetry === 'custom') return `\\text{custom}_${k}`;
  return '\\{e\\}';
}

function piIsIdentity(pi) {
  if (!pi) return false;
  for (const [k, v] of Object.entries(pi)) {
    if (k !== v) return false;
  }
  return true;
}

function piCycleNotation(pi) {
  if (!pi) return '—';
  if (piIsIdentity(pi)) return 'identity';
  const visited = new Set();
  const cycles = [];
  const keys = Object.keys(pi).sort();
  for (const label of keys) {
    if (visited.has(label)) continue;
    const cycle = [];
    let cur = label;
    while (!visited.has(cur)) {
      visited.add(cur);
      cycle.push(cur);
      cur = pi[cur];
    }
    if (cycle.length >= 2) cycles.push(`(${cycle.join(' ')})`);
  }
  return cycles.length ? cycles.join('') : 'identity';
}
