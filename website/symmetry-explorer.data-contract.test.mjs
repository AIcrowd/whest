// V3.1 §C51 — Data Contract Notes (L5.T2.21).
//
// Source-grep tests covering the formal V3.1 data contract shipped in
// `engine/dataContract.ts`. The tests intentionally pin the smallest
// piece of source truth that proves each deliverable is in place, so
// they survive cosmetic edits and don't regress when neighbouring
// docstrings change.
//
// Deliverables verified here (per the L5.T2.21 spec):
//   1. dataContract module exists at the expected path.
//   2. AnalysisV3_1 / ComponentV3_1 / OQDataV3_1 interfaces are exported.
//   3. ActualAnalysis interface documents the current engine output.
//   4. FIELD_MAP cross-walk is exported.
//   5. validateAnalysisShape function is exported with the documented
//      result shape ({ ok, missingV3_1Fields, aliasMismatches }).
//   6. At least one concrete alias (V3.1 → actual) is documented.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_CONTRACT = 'components/symmetry-aware-einsum-contractions/engine/dataContract.ts';

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

// ─── 1. Module exists ───────────────────────────────────────────────

test('C51 — dataContract module exists at engine/dataContract.ts', () => {
  const path = resolve(__dirname, DATA_CONTRACT);
  assert.ok(existsSync(path), `expected dataContract.ts at ${DATA_CONTRACT}`);
});

// ─── 2. V3.1 interfaces exported ────────────────────────────────────

test('C51 — exports AnalysisV3_1 interface per V3.1 §C51 global shape', () => {
  const src = read(DATA_CONTRACT);
  assert.match(src, /export\s+interface\s+AnalysisV3_1\b/);
  // Spec keys spot-check: presetId, einsum, components, totalCost,
  // denseBaseline, productOrbitCounts, alphaCounts, certificationAudit.
  assert.match(src, /\bpresetId\b/);
  assert.match(src, /\beinsum\b/);
  assert.match(src, /\bproductOrbitCounts\b/);
  assert.match(src, /\balphaCounts\b/);
  assert.match(src, /\bcertificationAudit\b/);
});

test('C51 — exports ComponentV3_1 interface with the V3.1 per-component fields', () => {
  const src = read(DATA_CONTRACT);
  assert.match(src, /export\s+interface\s+ComponentV3_1\b/);
  // Spec keys: componentId, productOrbitCount, alphaCount, alphaMethod,
  // denseCount, classificationPath, isAvailable, unavailableReason.
  assert.match(src, /\bcomponentId\b/);
  assert.match(src, /\bproductOrbitCount\b/);
  assert.match(src, /\balphaCount\b/);
  assert.match(src, /\balphaMethod\b/);
  assert.match(src, /\bisAvailable\b/);
  assert.match(src, /\bunavailableReason\b/);
});

test('C51 — exports OQDataV3_1 interface with rows/columns/cells/rowBranchCounts/alpha', () => {
  const src = read(DATA_CONTRACT);
  assert.match(src, /export\s+interface\s+OQDataV3_1\b/);
  for (const key of ['rows', 'columns', 'cells', 'rowBranchCounts', 'alpha']) {
    assert.match(src, new RegExp(`\\b${key}\\b`), `OQDataV3_1 missing field ${key}`);
  }
});

// ─── 3. Actual engine shape documented ──────────────────────────────

test('C51 — exports ActualAnalysis interface reflecting the current engine output', () => {
  const src = read(DATA_CONTRACT);
  assert.match(src, /export\s+interface\s+ActualAnalysis\b/);
  // Field spot-check from `engine/pipeline.js#analyzeExample` return shape.
  for (const key of ['graph', 'symmetry', 'componentData', 'costModel', 'componentCosts']) {
    assert.match(src, new RegExp(`\\b${key}\\b`), `ActualAnalysis missing field ${key}`);
  }
});

// ─── 4. FIELD_MAP exported ──────────────────────────────────────────

test('C51 — exports FIELD_MAP cross-walk from V3.1 names to engine paths', () => {
  const src = read(DATA_CONTRACT);
  // FIELD_MAP is exported as a const (frozen Record).
  assert.match(src, /export\s+const\s+FIELD_MAP\b/);
  // Alias rows are documented inside the literal — keys on the LHS of `:`.
  assert.match(src, /\blabels\s*:/);
  assert.match(src, /\bcomponents\s*:/);
});

// ─── 5. Validator exported with documented result shape ─────────────

test('C51 — exports validateAnalysisShape function', () => {
  const src = read(DATA_CONTRACT);
  assert.match(src, /export\s+function\s+validateAnalysisShape\b/);
});

test('C51 — validateAnalysisShape result shape declares ok / missingV3_1Fields / aliasMismatches', () => {
  const src = read(DATA_CONTRACT);
  // ValidateResult interface is exported and lists the three documented keys.
  assert.match(src, /export\s+interface\s+ValidateResult\b/);
  assert.match(src, /\bok\s*:\s*boolean/);
  assert.match(src, /\bmissingV3_1Fields\s*:\s*string\[\]/);
  assert.match(src, /\baliasMismatches\s*:\s*Array</);
});

// ─── 6. At least one renamed alias documented ───────────────────────

test('C51 — FIELD_MAP documents at least one V3.1 → actual rename (alphaCount → accumulation.count)', () => {
  const src = read(DATA_CONTRACT);
  // The component-side alphaCount alias must point to the engine's
  // accumulation.count field — this is the canonical example the audit
  // row calls out for future T3 rename work.
  assert.match(
    src,
    /'component\.alphaCount':\s*'accumulation\.count'/,
    'FIELD_MAP must document component.alphaCount → accumulation.count alias',
  );
});

test('C51 — FIELD_MAP documents the productOrbitCount → multiplication.count alias', () => {
  const src = read(DATA_CONTRACT);
  // The multiplication-side analogue: per-component product orbit count
  // currently lives on `multiplication.count`.
  assert.match(
    src,
    /'component\.productOrbitCount':\s*'multiplication\.count'/,
    'FIELD_MAP must document component.productOrbitCount → multiplication.count alias',
  );
});

test('C51 — RENAMED_FIELDS is exported as the actionable T3 rename list', () => {
  const src = read(DATA_CONTRACT);
  // Sibling export distilling FIELD_MAP into the (v31, actual) pairs that
  // are worth a future T3 rename pass.
  assert.match(src, /export\s+const\s+RENAMED_FIELDS\b/);
  // Must reference at least one canonical pair so the test fails if the
  // export is gutted.
  assert.match(src, /v31:\s*'alphaCount'/);
  assert.match(src, /actual:\s*'accumulation\.count'/);
});
