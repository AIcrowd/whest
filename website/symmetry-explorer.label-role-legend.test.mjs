// V3.1 §C05 — Label Role Legend (visible | summed | declared symmetric axes)
//
// The legend in AlgorithmAtAGlance is a hover/click-lockable surface. Hovering
// a chip writes the corresponding label set to the page-wide hoveredLabels
// bus (so every component subscribed via the same prop chain lights up). A
// click locks the broadcast. The reverse direction — hovering a label
// somewhere else on the page — lights up the chip for the role that label
// belongs to (coral pulse/ring).
//
// These are source-grep tests: they assert the wiring contract that survives
// future refactors, not the rendered DOM. The behavior is exercised end-to-end
// by `symmetry-explorer.preamble.test.mjs` (placement) and the existing
// hoveredLabels-bus tests (cross-highlighting); this file pins the role-legend
// contract specifically.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const COMPONENTS = 'components/symmetry-aware-einsum-contractions/components';
const LIB = 'components/symmetry-aware-einsum-contractions/lib';

const readFile = (rel) => readFileSync(resolve(__dirname, rel), 'utf-8');

test('AlgorithmAtAGlance renders the V3.1 LabelRoleLegend with three roles (visible, summed, declared symmetric axes)', () => {
  const src = readFile(`${COMPONENTS}/AlgorithmAtAGlance.jsx`);
  // The new component is named LabelRoleLegend (replaces the static ColorLegend).
  assert.match(src, /function LabelRoleLegend\(/);
  // No leftover ColorLegend definition or call site.
  assert.doesNotMatch(src, /function ColorLegend\(/);
  assert.doesNotMatch(src, /<ColorLegend\b/);
  // Legend is mounted from EinsumIntroColumn with the bus props wired through.
  assert.match(src, /<LabelRoleLegend\b[\s\S]*?hoveredLabels=\{hoveredLabels\}[\s\S]*?onHoveredLabelsChange=\{onHoveredLabelsChange\}[\s\S]*?\/>/);
  // Three roles: 'visible' chip, 'summed' chip, 'declared' chip.
  assert.match(src, /id:\s*'visible'/);
  assert.match(src, /id:\s*'summed'/);
  assert.match(src, /id:\s*'declared'/);
  // Declared chip suffix mentions "symmetric axes" (the V3.1 spec wording).
  assert.match(src, /symmetric axes/);
});

test('Each legend chip is a clickable button with an accessible aria-label and lock state', () => {
  const src = readFile(`${COMPONENTS}/AlgorithmAtAGlance.jsx`);
  // Chips render as <button type="button"> (not bare divs) so keyboard users
  // get Enter/Space activation for free.
  assert.match(src, /<button[\s\S]*?type="button"[\s\S]*?onClick=\{isInteractive \? \(\) => handleClick\(item\.id, item\.labels\) : undefined\}/);
  // aria-label is sourced from the per-role item.ariaLabel.
  assert.match(src, /aria-label=\{item\.ariaLabel\}/);
  // aria-pressed reflects the locked state (toggle-button pattern).
  assert.match(src, /aria-pressed=\{isLocked\}/);
  // Each role declares a non-empty ariaLabel string.
  assert.match(src, /ariaLabel:\s*'Visible labels[^']*'/);
  assert.match(src, /ariaLabel:\s*'Summed labels[^']*'/);
  assert.match(src, /ariaLabel:\s*'Declared symmetric axes[^']*'/);
});

test('Each legend chip writes to the hoveredLabels bus on hover and on focus', () => {
  const src = readFile(`${COMPONENTS}/AlgorithmAtAGlance.jsx`);
  // onMouseEnter → handleHover(item.labels); onMouseLeave → handleLeave().
  assert.match(src, /onMouseEnter=\{isInteractive \? \(\) => handleHover\(item\.labels\) : undefined\}/);
  assert.match(src, /onMouseLeave=\{isInteractive \? \(\) => handleLeave\(\) : undefined\}/);
  // Keyboard parity: focus/blur fire the same handlers.
  assert.match(src, /onFocus=\{isInteractive \? \(\) => handleHover\(item\.labels\) : undefined\}/);
  assert.match(src, /onBlur=\{isInteractive \? \(\) => handleLeave\(\) : undefined\}/);
  // handleHover broadcasts a Set of the role's labels via onHoveredLabelsChange.
  assert.match(src, /onHoveredLabelsChange\(new Set\(labels\)\)/);
  // handleLeave clears the bus.
  assert.match(src, /handleLeave[\s\S]*?onHoveredLabelsChange\(null\)/);
});

test('Click toggles a useState-backed lock so the highlight persists until clicked again', () => {
  const src = readFile(`${COMPONENTS}/AlgorithmAtAGlance.jsx`);
  // Lock state lives in local React state.
  assert.match(src, /const \[lockedRole, setLockedRole\] = useState\(null\)/);
  // Click toggles: same role clicked twice → unlock + clear.
  assert.match(src, /if \(lockedRole === roleId\)\s*\{[\s\S]*?setLockedRole\(null\)[\s\S]*?onHoveredLabelsChange\(null\)/);
  // Click on a new role → lock that role and broadcast its labels.
  assert.match(src, /setLockedRole\(roleId\)[\s\S]*?onHoveredLabelsChange\(new Set\(labels\)\)/);
  // While locked, hover/leave on other chips must not clobber the broadcast.
  assert.match(src, /if \(lockedRole\) return; \/\/ lock supersedes hover broadcast/);
  // data-locked attribute pins the visual lock state to the DOM for testing.
  assert.match(src, /data-locked=\{isLocked \? 'true' : 'false'\}/);
});

test('Legend reads from the hoveredLabels bus (passed in as a prop) for reverse-highlight', () => {
  const src = readFile(`${COMPONENTS}/AlgorithmAtAGlance.jsx`);
  // The component accepts hoveredLabels as a prop (the same bus other
  // components in §1 read via prop drill — see StickyBar.jsx LabelChipList).
  assert.match(src, /function LabelRoleLegend\(\{[\s\S]*?hoveredLabels[\s\S]*?onHoveredLabelsChange[\s\S]*?\}\)/);
  // Reverse-highlight checks: any of the role's labels currently in the bus?
  assert.match(src, /const hoveredSet = hoveredLabels instanceof Set \? hoveredLabels : null/);
  assert.match(src, /isReverseHit = hoveredSet[\s\S]*?item\.labels\.some\(\(ch\) => hoveredSet\.has\(ch\)\)/);
  // The prop is forwarded from EinsumIntroColumn → LabelRoleLegend.
  assert.match(src, /<LabelRoleLegend[\s\S]*?hoveredLabels=\{hoveredLabels\}/);
});

test('Reverse-highlight applies a coral pulse/ring class when the bus matches the role', () => {
  const src = readFile(`${COMPONENTS}/AlgorithmAtAGlance.jsx`);
  // The reverse-hit state drives a visual class — a coral border + ring +
  // pulse animation. We check the conjunction (both isReverseHit and isLocked
  // share the same visual treatment).
  assert.match(src, /\(isReverseHit \|\| isLocked\)[\s\S]*?border-\[color:var\(--coral\)\]/);
  assert.match(src, /\(isReverseHit \|\| isLocked\)[\s\S]*?ring-2/);
  assert.match(src, /\(isReverseHit \|\| isLocked\)[\s\S]*?animate-pulse/);
  // data-reverse-hit attribute exposes the reverse state to the DOM.
  assert.match(src, /data-reverse-hit=\{isReverseHit \? 'true' : 'false'\}/);
});

test('section1ExampleView surfaces vFreeLabels, wSummedLabels, and declaredSymmetricLabels for the legend to consume', () => {
  const src = readFile(`${LIB}/section1ExampleView.js`);
  // The view contract returns three label arrays the legend reads.
  assert.match(src, /vFreeLabels:\s*freeLabels/);
  assert.match(src, /wSummedLabels:\s*summedLabels/);
  assert.match(src, /declaredSymmetricLabels/);
  // The helper that derives declared symmetric labels from per-operand
  // symAxes is named explicitly so future changes don't quietly reshape it.
  assert.match(src, /function collectDeclaredSymmetricLabels\(variables, subscripts\)/);
  // It walks every variable's symAxes (or the full subscript when symAxes is
  // empty/null but the symmetry is non-'none') and unions the labels.
  assert.match(src, /variable\.symmetry === 'none'/);
  assert.match(src, /Array\.isArray\(variable\.symAxes\) && variable\.symAxes\.length > 0/);
});

test('declaredSymmetricLabels collects every label under any declared symmetric axis', async () => {
  // This is a tiny behavioural test — pure-fn, no React. We assert that the
  // helper actually returns the right labels for the canonical S2-on-axes-[0,1]
  // operand pattern and an empty result for fully dense examples.
  const mod = await import('./components/symmetry-aware-einsum-contractions/lib/section1ExampleView.js');
  const denseExample = {
    expression: { subscripts: 'ij,jk', output: 'ik', operandNames: 'A,B' },
    variables: [
      { name: 'A', rank: 2, symmetry: 'none', symAxes: null },
      { name: 'B', rank: 2, symmetry: 'none', symAxes: null },
    ],
  };
  const denseView = mod.buildSection1ExampleView(denseExample);
  assert.deepEqual(denseView.declaredSymmetricLabels, []);

  const symExample = {
    expression: { subscripts: 'ij,jk', output: 'ik', operandNames: 'S,B' },
    variables: [
      // S is symmetric on its two axes → labels i, j are declared symmetric.
      { name: 'S', rank: 2, symmetry: 'symmetric', symAxes: [0, 1] },
      { name: 'B', rank: 2, symmetry: 'none', symAxes: null },
    ],
  };
  const symView = mod.buildSection1ExampleView(symExample);
  // i and j must be present; k must not (B is dense).
  assert.ok(symView.declaredSymmetricLabels.includes('i'));
  assert.ok(symView.declaredSymmetricLabels.includes('j'));
  assert.ok(!symView.declaredSymmetricLabels.includes('k'));
});
