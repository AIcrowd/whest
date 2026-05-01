// V3.1 §36 — C36 Appendix-Linked Partition Formula Card.
//
// Pins the per-term hover bus on TypedPartitionDemo's formula header:
// each of the four V3.1 §36 sub-expressions (pattern, falling factorial,
// divisor, output reach) renders as a focusable hover target that tints
// the corresponding column in the per-pattern table coral-light. Also
// pins the "Full statement → Appendix C" link so readers can deep-link
// from the formula card to the canonical theorem in Appendix C.
//
// Why source-grep instead of a browser test: the hover bus is purely a
// state-driven JSX composition with no engine logic. Source-grep keeps the
// test light and locks the structural contract (state name, hover spans,
// column attributes, tint token, anchor href) so reviewers can spot
// drift in a one-line diff.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const source = fs.readFileSync(
  new URL(
    './components/symmetry-aware-einsum-contractions/components/TypedPartitionDemo.jsx',
    import.meta.url,
  ),
  'utf8',
);

// ─── 1. Hover state declared via useState ────────────────────────────────────
test('TypedPartitionDemo declares formulaTermHover state via useState', () => {
  // The V3.1 §36 hover bus is parented in TypedPartitionDemo so the four
  // formula spans can drive a single piece of state that the per-pattern
  // table reads. Pin both the state name and the literal `useState(null)`
  // initializer so an accidental rename or default change doesn't silently
  // break the cross-cutting hover contract.
  assert.match(
    source,
    /const \[formulaTermHover, setFormulaTermHover\] = useState\(null\)/,
    'TypedPartitionDemo must declare formulaTermHover state via useState(null)',
  );
});

// ─── 2. Four hover targets render via FormulaHoverSpan ───────────────────────
test('TypedPartitionDemo renders four FormulaHoverSpan targets (pattern, fallingFactorial, divisor, outputReach)', () => {
  // Each summand in α_a = Σ_p̃ (falling-factorial / |Ḡ_p̃|) · |A_p̃/H_a|
  // gets its own hover target. The four `term="…"` props are the canonical
  // identifiers wired to the per-pattern table's data-pattern-column
  // attributes — keep both lists in lockstep here so a typo on either side
  // shows up immediately.
  assert.match(
    source,
    /<FormulaHoverSpan[\s\S]{0,200}term="pattern"/,
    'Must render <FormulaHoverSpan term="pattern" />',
  );
  assert.match(
    source,
    /<FormulaHoverSpan[\s\S]{0,200}term="fallingFactorial"/,
    'Must render <FormulaHoverSpan term="fallingFactorial" />',
  );
  assert.match(
    source,
    /<FormulaHoverSpan[\s\S]{0,200}term="divisor"/,
    'Must render <FormulaHoverSpan term="divisor" />',
  );
  assert.match(
    source,
    /<FormulaHoverSpan[\s\S]{0,200}term="outputReach"/,
    'Must render <FormulaHoverSpan term="outputReach" />',
  );
});

// ─── 3. Hover targets are cursor-help + tabIndex={0} (keyboard reachable) ────
test('FormulaHoverSpan exposes cursor-help, tabIndex={0}, and focus/blur handlers', () => {
  // V3.1 §36 calls out "All hover targets must be reachable via Tab and
  // activate the highlight via focus too (not just mouse hover)." Pin the
  // cursor-help class (mouse affordance), tabIndex={0} (keyboard reach),
  // and the onFocus/onBlur pair (parity with onMouseEnter/onMouseLeave).
  // role="button" makes the span discoverable to assistive tech.
  assert.match(
    source,
    /cursor-help/,
    'FormulaHoverSpan must apply the cursor-help class',
  );
  assert.match(
    source,
    /tabIndex=\{0\}/,
    'FormulaHoverSpan must set tabIndex={0} so it is keyboard-reachable',
  );
  assert.match(
    source,
    /onFocus=\{enter\}/,
    'FormulaHoverSpan must wire onFocus to set the hover state',
  );
  assert.match(
    source,
    /onBlur=\{leave\}/,
    'FormulaHoverSpan must wire onBlur to clear the hover state',
  );
  assert.match(
    source,
    /onMouseEnter=\{enter\}/,
    'FormulaHoverSpan must wire onMouseEnter to set the hover state',
  );
  assert.match(
    source,
    /onMouseLeave=\{leave\}/,
    'FormulaHoverSpan must wire onMouseLeave to clear the hover state',
  );
  assert.match(
    source,
    /role="button"/,
    'FormulaHoverSpan must declare role="button" so AT users can perceive the affordance',
  );
});

// ─── 4. Each hover target carries an aria-label naming the column ────────────
test('FORMULA_HOVER_TARGETS carries an aria-label for each of the four terms', () => {
  // The aria-label is what screen-reader users hear; it must name what
  // hovering will highlight (V3.1 §36 spec). Pin the four aria-label
  // strings so a careless edit can't silently drop the AT cue.
  assert.match(
    source,
    /pattern:[\s\S]{0,200}ariaLabel: 'Hover to highlight the pattern column in the table below'/,
    'pattern hover target must have an aria-label naming the pattern column',
  );
  assert.match(
    source,
    /fallingFactorial:[\s\S]{0,200}ariaLabel: 'Hover to highlight the concrete labelings column in the table below'/,
    'fallingFactorial hover target must have an aria-label naming the concrete labelings column',
  );
  assert.match(
    source,
    /divisor:[\s\S]{0,200}ariaLabel: 'Hover to highlight the block-symmetry divisor column in the table below'/,
    'divisor hover target must have an aria-label naming the block-symmetry divisor column',
  );
  assert.match(
    source,
    /outputReach:[\s\S]{0,200}ariaLabel: 'Hover to highlight the output reach column in the table below'/,
    'outputReach hover target must have an aria-label naming the output reach column',
  );
  // The FormulaHoverSpan helper must read aria-label from this metadata so
  // the four span aria-labels stay aligned with the constant table.
  assert.match(
    source,
    /aria-label=\{meta\.ariaLabel\}/,
    'FormulaHoverSpan must source aria-label from FORMULA_HOVER_TARGETS metadata',
  );
});

// ─── 5. "Full statement → Appendix C" link with the right anchor ─────────────
test('TypedPartitionDemo renders a "Full statement → Appendix C" link to #appendix-section-6', () => {
  // The V3.1 §36 spec adds a deep-link from the formula card to the
  // canonical theorem statement in Appendix C. The anchor target
  // `#appendix-section-6` is the existing in-modal id used by
  // ExpressionLevelModal.jsx — keep them in lockstep so the link lands on
  // the right section. The aria-label gives screen-reader users the same
  // context the visible "→ Appendix C" cue gives sighted readers.
  assert.match(
    source,
    /href="#appendix-section-6"/,
    'Appendix C link must target the canonical #appendix-section-6 anchor',
  );
  assert.match(
    source,
    /aria-label="Read Appendix C — Typed partition counting theorem"/,
    'Appendix C link must carry the V3.1 §36 aria-label',
  );
  assert.match(
    source,
    />\s*Full statement → Appendix C\s*</,
    'Appendix C link text must read "Full statement → Appendix C"',
  );
  assert.match(
    source,
    /data-formula-link="appendix-c"/,
    'Appendix C link must carry data-formula-link="appendix-c" for testing',
  );
});

// ─── 6. Per-pattern table columns have data-pattern-column attributes ────────
test('Per-pattern table emits data-pattern-column attributes for each of the four columns', () => {
  // The four `data-pattern-column` values are the binding the formula
  // hover spans drive. Pin each of the four values so a column rename or
  // typo can't silently break the highlight wiring. The header and at
  // least one body cell must carry the attribute (we assert the body
  // cells implicitly via the per-row `tintFor(column)` helper below).
  assert.match(
    source,
    /data-pattern-column="pattern"/,
    'Table must emit data-pattern-column="pattern"',
  );
  assert.match(
    source,
    /data-pattern-column="fallingFactorial"/,
    'Table must emit data-pattern-column="fallingFactorial"',
  );
  assert.match(
    source,
    /data-pattern-column="divisor"/,
    'Table must emit data-pattern-column="divisor"',
  );
  assert.match(
    source,
    /data-pattern-column="outputReach"/,
    'Table must emit data-pattern-column="outputReach"',
  );
});

// ─── 7. Hover tints the matching column with var(--coral-light) ──────────────
test('When formulaTermHover is set, the matching column tints with var(--coral-light)', () => {
  // The tint token comes from the design system (coral-light = #FEF2F1).
  // Pin both the FORMULA_HOVER_TINT constant (CSS variable, not a hex) and
  // the per-cell tint helper that compares formulaTermHover to the
  // column id — drift on either side breaks the cross-component contract.
  assert.match(
    source,
    /const FORMULA_HOVER_TINT = 'var\(--coral-light\)'/,
    'FORMULA_HOVER_TINT must be the CSS variable token, not a raw hex',
  );
  assert.match(
    source,
    /formulaTermHover === 'pattern' \? FORMULA_HOVER_TINT : undefined/,
    'pattern column header must tint when formulaTermHover === "pattern"',
  );
  assert.match(
    source,
    /formulaTermHover === 'fallingFactorial' \? FORMULA_HOVER_TINT : undefined/,
    'fallingFactorial column header must tint when formulaTermHover === "fallingFactorial"',
  );
  assert.match(
    source,
    /formulaTermHover === 'divisor' \? FORMULA_HOVER_TINT : undefined/,
    'divisor column header must tint when formulaTermHover === "divisor"',
  );
  assert.match(
    source,
    /formulaTermHover === 'outputReach' \? FORMULA_HOVER_TINT : undefined/,
    'outputReach column header must tint when formulaTermHover === "outputReach"',
  );
  // Per-row body cells share a tintFor helper that returns the tint when
  // the column matches the active formula term, otherwise the row's
  // selection background. Pin both arms so either change shows up.
  assert.match(
    source,
    /formulaTermHover === column \? FORMULA_HOVER_TINT : rowBackground/,
    'tintFor helper must return FORMULA_HOVER_TINT for the active column, otherwise the row background',
  );
});

// ─── 8. FormulaHoverSpan helper exists + uses no raw hex tint ────────────────
test('FormulaHoverSpan helper exists and uses the FORMULA_HOVER_TINT token (no raw hex)', () => {
  // The hover span is a thin shell over the column-tint logic; pinning the
  // helper signature keeps a future refactor honest about its prop shape
  // (term / activeTerm / setTerm / children). Also assert the active-tint
  // path goes through FORMULA_HOVER_TINT so a refactor can't sneak a raw
  // hex back in (V3.1 token discipline — no notation hex, CSS vars only).
  assert.match(
    source,
    /function FormulaHoverSpan\(\{ term, activeTerm, setTerm, children \}\)/,
    'FormulaHoverSpan must accept (term, activeTerm, setTerm, children) props',
  );
  assert.match(
    source,
    /background: isActive \? FORMULA_HOVER_TINT : 'transparent'/,
    'FormulaHoverSpan active background must use FORMULA_HOVER_TINT, not a raw hex',
  );
  // No `#FEF2F1` (the coral-light hex) should appear inside the file —
  // we go through the CSS variable to honor the V3.1 token discipline.
  assert.doesNotMatch(
    source,
    /#FEF2F1/i,
    'TypedPartitionDemo must not use the raw coral-light hex; use var(--coral-light)',
  );
});
