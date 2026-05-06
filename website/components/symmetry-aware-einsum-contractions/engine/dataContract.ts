// V3.1 §C51 — Data Contract Notes (L5.T2.21).
//
// This module formalizes the analysis-object data contract that the V3.1
// migration plan (§C51) prescribes, alongside the *actual* current shape
// produced by `engine/pipeline.js#analyzeExample`. Future T3 work can
// incrementally rename engine fields to converge on the V3.1 contract;
// until then, downstream consumers should rely on the FIELD_MAP below
// rather than re-deriving aliases ad hoc.
//
// This file is DOCUMENTATION-ONLY:
//   - No engine logic is changed.
//   - No production code imports from here yet.
//   - The runtime validator is a non-throwing introspection helper that
//     returns a structured warning report (used by tests and ad-hoc tools).
//
// Companion audit row (V3.1 §C51):
//   "The engine likely has analogous fields under different names.
//    Formalizing the contract prevents mismatched values across components."

// ─── V3.1 spec interfaces ────────────────────────────────────────────

/**
 * V3.1 §C51 — Global analysis object (per spec).
 *
 * Top-level fields the V3.1 migration plan prescribes for the analysis
 * payload returned by the analyze pipeline. These names are aspirational:
 * see `ActualAnalysis` (below) for the current engine output, and
 * `FIELD_MAP` for the alias mapping.
 */
export interface AnalysisV3_1 {
  /** Stable id of the worked example / preset. */
  presetId: string;
  /** The einsum string ("ij,jk->ik" etc.). */
  einsum: string;
  /** Per-operand metadata (subscripts, declared symmetry, …). */
  operands: ReadonlyArray<unknown>;
  /** Union of all label characters appearing in the expression. */
  labels: ReadonlyArray<string>;
  /** Labels that appear in the output (V-labels). */
  visibleLabels: ReadonlyArray<string>;
  /** Labels that are summed out (W-labels). */
  summedLabels: ReadonlyArray<string>;
  /** Per-label dimension overrides. */
  labelSizes: Readonly<Record<string, number>>;
  /** Symmetry the user *declared* (vs what was *detected*). */
  declaredSymmetries: ReadonlyArray<unknown>;
  /** Group description detected by the auto-symmetry pipeline. */
  detectedGroup: unknown;
  /** How the group acts on the output (free / projects / collapses). */
  outputAction: unknown;
  /** Per-component decomposition (independent V-blocks). */
  components: ReadonlyArray<ComponentV3_1>;
  /** Per-component product-orbit count M_a (multiplication side). */
  productOrbitCounts: ReadonlyArray<number>;
  /** Per-component output-orbit count α_a (accumulation side). */
  alphaCounts: ReadonlyArray<number>;
  /** Top-level total cost (μ + α plug-in). */
  totalCost: number;
  /** Dense (no-symmetry) baseline used as the reduction reference. */
  denseBaseline: number;
  /** Result of the regime classification ladder (per component). */
  classificationResults: ReadonlyArray<unknown>;
  /** O→Q partition data (orbit decomposition by output representative). */
  partitionData: OQDataV3_1 | null;
  /** Audit trail used by the certification block in the article. */
  certificationAudit: unknown;
}

/**
 * V3.1 §C51 — Per-component shape.
 *
 * One block per independent V-component of the symmetry group. The article's
 * Section 4 component cost view consumes this shape per row.
 */
export interface ComponentV3_1 {
  /** Stable component id (within an analysis). */
  componentId: string;
  /** All labels participating in this component. */
  labels: ReadonlyArray<string>;
  /** V-labels (visible / output) inside this component. */
  visibleLabels: ReadonlyArray<string>;
  /** W-labels (summed) inside this component. */
  summedLabels: ReadonlyArray<string>;
  /** Group acting on this component (name + presentation). */
  group: unknown;
  /** Output-action classification for this component. */
  outputAction: unknown;
  /** M_a — product-orbit count for the component. */
  productOrbitCount: number;
  /** α_a — output-orbit count for the component. */
  alphaCount: number | null;
  /** Which regime in the ladder produced α_a. */
  alphaMethod: string | null;
  /** Brute-force ground truth ∏ n_ℓ. */
  denseCount: number;
  /** Trace describing how the regime ladder arrived at α_a. */
  classificationPath: ReadonlyArray<unknown>;
  /** Whether this component was fully analyzed (false → see reason). */
  isAvailable: boolean;
  /** Human-readable reason if `isAvailable === false`. */
  unavailableReason: string | null;
}

/**
 * V3.1 §C51 — O→Q (output orbits → output representatives) data.
 *
 * Powers the dense-grid / orbit-rep matrix views in Sections 5-6.
 */
export interface OQDataV3_1 {
  /** Product-orbit rows (O). */
  rows: ReadonlyArray<unknown>;
  /** Output-representative columns (Q). */
  columns: ReadonlyArray<unknown>;
  /** Incidence cells (which O lands in which Q, with multiplicity). */
  cells: ReadonlyArray<unknown>;
  /** Branch counts per row (sum across that row's cells). */
  rowBranchCounts: ReadonlyArray<number>;
  /** α — total number of distinct output orbits. */
  alpha: number;
}

// ─── Actual current engine shape ─────────────────────────────────────

/**
 * Actual current shape returned by `engine/pipeline.js#analyzeExample`.
 *
 * Discovered by reading `pipeline.js` (line 100-115) and the modules it
 * delegates to (`algorithm.js`, `componentDecomposition.js`,
 * `costModel.js`, `expressionGroup.js`).
 *
 * The current engine groups data by *concern* (graph / symmetry / cost /
 * decomposition) rather than by V3.1's flat layout. Most V3.1 fields are
 * present, but under different names and at different nesting depths —
 * see `FIELD_MAP` for the cross-walk.
 */
export interface ActualAnalysis {
  /** Bipartite graph of operand labels (from `buildBipartite`). */
  graph: unknown;
  /** Incidence matrix derived from `graph`. */
  matrixData: unknown;
  /** Sigma-loop output (Shape-1/2/3 results). */
  sigmaResults: ReadonlyArray<unknown>;
  /** Detected symmetry group + V/W decomposition + classifier output. */
  symmetry: ActualSymmetry;
  /** Per-component decomposition + classification + counts. */
  componentData: ActualComponentData;
  /** Burnside count over the full group (orbit count over the V × W tuple). */
  burnside: ActualBurnside;
  /** Brute-force exact cost model used as ground truth. */
  costModel: ActualCostModel;
  /** Per-component aggregated μ / α (the values the hero displays). */
  componentCosts: ActualComponentCosts | null;
  /** Per-label dimension clusters (sizes after labelSizes overrides). */
  clusters: ReadonlyArray<unknown>;
  /** V-sub × S(W) expression-level group (pedagogical). */
  expressionGroup: unknown;
}

export interface ActualSymmetry {
  allLabels: ReadonlyArray<string>;
  vLabels: ReadonlyArray<string>;
  wLabels: ReadonlyArray<string>;
  vGenerators: ReadonlyArray<unknown>;
  vGeneratorsAll: ReadonlyArray<unknown>;
  wGenerators: ReadonlyArray<unknown>;
  vElements: ReadonlyArray<unknown>;
  wElements: ReadonlyArray<unknown>;
  vOrder: number;
  vGroupName: string;
  wOrder: number;
  wGroupName: string;
  vDegree: number;
  wDegree: number;
  fullGenerators: ReadonlyArray<unknown>;
  fullElements: ReadonlyArray<unknown>;
  fullOrder: number;
  fullDegree: number;
  fullGroupName: string;
  actionSummary: unknown;
  validPiResults: ReadonlyArray<unknown>;
  generatorSelection: unknown;
  wreathElements: ReadonlyArray<unknown>;
  identicalGroups: ReadonlyArray<unknown>;
}

export interface ActualComponentData {
  interactionGraph: unknown;
  components: ReadonlyArray<ActualComponent>;
}

export interface ActualComponent {
  indices: ReadonlyArray<number>;
  labels: ReadonlyArray<string>;
  va: ReadonlyArray<string>;
  wa: ReadonlyArray<string>;
  generators: ReadonlyArray<unknown>;
  elements: ReadonlyArray<unknown>;
  order: number;
  groupName: string;
  sizes: ReadonlyArray<number>;
  shape: string;
  /** α_a + which regime produced it + LaTeX + decision trace. */
  accumulation: ActualAccumulation;
  /** M_a — Burnside on this component's group action. */
  multiplication: { count: number };
}

export interface ActualAccumulation {
  regimeId: string | null;
  count: number | null;
  latex: string;
  latexSymbolic: string;
  trace: ReadonlyArray<unknown>;
}

export interface ActualBurnside {
  perElement: ReadonlyArray<unknown>;
  totalFixed: number;
  uniqueCount: number;
  totalCount: number;
  ratio: number;
  wPerElement: ReadonlyArray<unknown>;
  wUniqueCount: number;
  wTotalCount: number;
  wRatio: number;
  wHasSymmetry: boolean;
  fullPerElement: ReadonlyArray<unknown>;
  fullTotalFixed: number;
  orbitCount: number;
  totalTupleCount: number;
  dimensionN: number;
}

export interface ActualCostModel {
  orbitCount: number;
  evaluationCostExact: number;
  reductionCostExact: number;
  orbitRows: ReadonlyArray<unknown>;
}

export interface ActualComponentCosts {
  mu: number;
  alpha: number;
  mTotal: number;
  perComponent: ReadonlyArray<{
    labels: ReadonlyArray<string>;
    M_a: number;
    alpha_a: number;
    regimeId: string | null;
  }>;
}

// ─── Field-map cross-walk ────────────────────────────────────────────

/**
 * Maps V3.1 §C51 field names → the actual engine field path (dot-form).
 *
 * - LHS is the V3.1 key (top-level on `AnalysisV3_1` or
 *   nested inside a `ComponentV3_1`).
 * - RHS is the dot-path inside the engine's actual output (or `null`
 *   if the V3.1 field has no current analogue and would need to be
 *   computed from scratch in T3).
 *
 * The mapping intentionally uses string paths (not type-level lookups)
 * because the engine output is dynamic JS and the goal here is *runtime*
 * documentation, not a compile-time refactor.
 */
export const FIELD_MAP: Readonly<Record<string, string | null>> = Object.freeze({
  // ── Global analysis object ──────────────────────────────────────────
  presetId: null, // not stored on the analysis itself; kept on the example
  einsum: null,   // reconstructed from `example.subscripts` + `output`
  operands: null, // available via `graph.identicalGroups` and the example
  labels: 'symmetry.allLabels',
  visibleLabels: 'symmetry.vLabels',
  summedLabels: 'symmetry.wLabels',
  labelSizes: null, // sizes are denormalized into `clusters` and per-component `sizes`
  declaredSymmetries: null, // declared on the example, not echoed onto analysis
  detectedGroup: 'symmetry.fullGroupName',
  outputAction: 'symmetry.actionSummary',
  components: 'componentData.components',
  productOrbitCounts: 'componentCosts.perComponent[*].M_a',
  alphaCounts: 'componentCosts.perComponent[*].alpha_a',
  totalCost: 'componentCosts.mu', // V3.1 totalCost ≈ μ-side; full total = (k-1)*mTotal + alpha
  denseBaseline: 'costModel.evaluationCostExact', // brute-force ground truth
  classificationResults: 'componentData.components[*].accumulation',
  partitionData: null, // O→Q matrix lives in `costModel.orbitRows` (different encoding)
  certificationAudit: null, // certification audit row is rendered on top of analysis, not stored

  // ── Per-component (under componentData.components[*]) ──────────────
  componentId: null,                   // current engine uses positional index, not a string id
  'component.labels': 'labels',
  'component.visibleLabels': 'va',
  'component.summedLabels': 'wa',
  'component.group': 'groupName',
  'component.outputAction': 'shape',
  'component.productOrbitCount': 'multiplication.count',
  'component.alphaCount': 'accumulation.count',
  'component.alphaMethod': 'accumulation.regimeId',
  'component.denseCount': null,        // not stored — equals ∏ comp.sizes
  'component.classificationPath': 'accumulation.trace',
  'component.isAvailable': null,       // implicit: accumulation.count !== null
  'component.unavailableReason': null, // implicit: last `trace` entry's reason

  // ── O→Q data (lives on costModel) ──────────────────────────────────
  'oq.rows': 'costModel.orbitRows',
  'oq.columns': null,        // current engine collapses Q into orbitRows[*].outputCount
  'oq.cells': null,          // dense incidence matrix not materialized yet
  'oq.rowBranchCounts': 'costModel.orbitRows[*].outputCount',
  'oq.alpha': 'costModel.reductionCostExact',
});

/**
 * Subset of FIELD_MAP that documents *renamed* fields (i.e. both V3.1 and
 * actual engine carry the value, just under different names). This is the
 * actionable list T3 work would consume to do the rename pass.
 */
export const RENAMED_FIELDS: ReadonlyArray<{ v31: string; actual: string }> = Object.freeze([
  { v31: 'labels', actual: 'symmetry.allLabels' },
  { v31: 'visibleLabels', actual: 'symmetry.vLabels' },
  { v31: 'summedLabels', actual: 'symmetry.wLabels' },
  { v31: 'detectedGroup', actual: 'symmetry.fullGroupName' },
  { v31: 'outputAction', actual: 'symmetry.actionSummary' },
  { v31: 'components', actual: 'componentData.components' },
  { v31: 'productOrbitCount', actual: 'multiplication.count' },
  { v31: 'alphaCount', actual: 'accumulation.count' },
  { v31: 'alphaMethod', actual: 'accumulation.regimeId' },
  { v31: 'classificationPath', actual: 'accumulation.trace' },
  { v31: 'denseBaseline', actual: 'costModel.evaluationCostExact' },
  { v31: 'rows', actual: 'costModel.orbitRows' },
]);

// ─── Runtime validator ───────────────────────────────────────────────

/**
 * Top-level keys the V3.1 spec mandates on the analysis object.
 *
 * Mirrors the documented `AnalysisV3_1` interface literally so the
 * validator can detect drift (interface added → key not yet present
 * in the literal list).
 */
export const V3_1_TOP_LEVEL_KEYS: ReadonlyArray<string> = Object.freeze([
  'presetId',
  'einsum',
  'operands',
  'labels',
  'visibleLabels',
  'summedLabels',
  'labelSizes',
  'declaredSymmetries',
  'detectedGroup',
  'outputAction',
  'components',
  'productOrbitCounts',
  'alphaCounts',
  'totalCost',
  'denseBaseline',
  'classificationResults',
  'partitionData',
  'certificationAudit',
]);

export interface ValidateResult {
  /** True if the analysis has all V3.1 top-level keys present. */
  ok: boolean;
  /** V3.1 keys missing from the analysis (or empty if `ok`). */
  missingV3_1Fields: string[];
  /** For each missing V3.1 field whose value is reachable via the
   *  documented FIELD_MAP path, an entry { v31, actual } showing the
   *  current alias. Useful for surfacing "the data is there, just under
   *  a different name" warnings. */
  aliasMismatches: Array<{ v31: string; actual: string }>;
}

function getByPath(obj: unknown, path: string): unknown {
  if (obj == null || typeof obj !== 'object') return undefined;
  const parts = path.split('.');
  let cur: unknown = obj;
  for (const p of parts) {
    if (cur == null || typeof cur !== 'object') return undefined;
    // Step through arrays via the `[*]` placeholder by short-circuiting
    // — for documentation purposes, "value reachable" is enough; we don't
    // dereference per-element paths.
    if (p.endsWith('[*]')) return cur;
    cur = (cur as Record<string, unknown>)[p];
  }
  return cur;
}

/**
 * Inspect an analysis object and report how it lines up with the
 * V3.1 §C51 contract. Pure: returns a structured warning report,
 * never throws, never mutates.
 *
 * Usage (development only):
 *   const report = validateAnalysisShape(analysis);
 *   if (!report.ok) console.warn('V3.1 drift', report);
 */
export function validateAnalysisShape(analysis: unknown): ValidateResult {
  const result: ValidateResult = {
    ok: true,
    missingV3_1Fields: [],
    aliasMismatches: [],
  };

  if (analysis == null || typeof analysis !== 'object') {
    result.ok = false;
    result.missingV3_1Fields = [...V3_1_TOP_LEVEL_KEYS];
    return result;
  }

  const obj = analysis as Record<string, unknown>;

  for (const key of V3_1_TOP_LEVEL_KEYS) {
    if (!Object.prototype.hasOwnProperty.call(obj, key)) {
      result.ok = false;
      result.missingV3_1Fields.push(key);

      const aliasPath = FIELD_MAP[key];
      if (aliasPath != null && getByPath(obj, aliasPath) !== undefined) {
        result.aliasMismatches.push({ v31: key, actual: aliasPath });
      }
    }
  }

  return result;
}
