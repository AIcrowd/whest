// V3.1 §35 — C35 Worked Pattern Mini-Ledger.
//
// Pins the prose mini-ledger surface that TypedPartitionDemo renders for
// the currently-selected pattern. The ledger walks the reader through
// the per-block falling-factorial multiplier, the total concrete
// labelings, a few member-assignment examples, the output reach, and
// the contribution this pattern family contributes to α — then closes
// with a required caveat that the prose only covers one pattern family
// and a "Show full α sum" toggle that expands the prose into the
// stacked sum across every pattern.
//
// Why source-grep instead of a browser test: the ledger and its caveat
// are pure JSX wiring with no engine logic. Source-grep keeps the test
// light and locks the V3.1 prose template plus the visual-callout
// styling so reviewers can spot drift in a one-line diff.

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

// ─── 1. V3.1 §35 prose ledger phrases ────────────────────────────────────────
test('TypedPartitionDemo renders the V3.1 §35 prose ledger phrases', () => {
  // The prose template walks the reader through three labeled prose
  // sections: the falling-factorial decomposition ("Concrete labelings:"),
  // the listed example tuples ("Member assignments:"), and the per-pattern
  // total ("Contribution shown for this pattern family:"). Pin each phrase
  // so a careless rename doesn't break the analogy with the V3.1 spec.
  assert.match(
    source,
    />Concrete labelings:</,
    'Mini-ledger must render the "Concrete labelings:" prose label',
  );
  assert.match(
    source,
    />Member assignments:</,
    'Mini-ledger must render the "Member assignments:" prose label (with up to MINI_LEDGER_EXAMPLE_LIMIT examples)',
  );
  assert.match(
    source,
    />Contribution shown for this pattern family:</,
    'Mini-ledger must render the "Contribution shown for this pattern family:" prose label',
  );
  // Stable container hook so QA / future tests can locate the ledger
  // without reaching into surrounding prose.
  assert.match(
    source,
    /data-testid="worked-pattern-mini-ledger"/,
    'Mini-ledger must carry data-testid="worked-pattern-mini-ledger"',
  );
});

// ─── 2. Required caveat copy ─────────────────────────────────────────────────
test('Mini-ledger emits the required V3.1 §35 caveat copy', () => {
  // The V3.1 spec requires the caveat sentence verbatim: this is one
  // pattern-family contribution, not the full alpha. Pin the "not the
  // full alpha" phrase plus the explicit mention of the all-distinct
  // and all-equal pattern families so readers always see the framing,
  // not just the per-pattern arithmetic.
  assert.ok(
    source.includes('not the full alpha'),
    'Caveat must include the phrase "not the full alpha"',
  );
  assert.ok(
    source.includes('one pattern-family contribution'),
    'Caveat must frame the ledger as "one pattern-family contribution"',
  );
  assert.ok(
    source.includes('all-distinct') && source.includes('all-equal'),
    'Caveat must name the all-distinct and all-equal pattern families that the prose ledger leaves out',
  );
});

// ─── 3. Caveat carries visual emphasis ───────────────────────────────────────
test('Caveat surface carries the Flopscope coral callout recipe', () => {
  // The caveat must read as a visual note, not as part of the prose
  // ledger above. Pin the single-accent design-system recipe: a left rail,
  // coral-light fill, and coral rail token. That keeps the note visually
  // tied to the active-state vocabulary instead of introducing a separate
  // warning palette.
  assert.match(
    source,
    /data-testid="worked-pattern-ledger-caveat"/,
    'Caveat must carry data-testid="worked-pattern-ledger-caveat"',
  );
  assert.match(
    source,
    /className="mt-2 border-l-4 px-3 py-2 text-\[11px\] leading-5"/,
    'Caveat must use a compact border-l-4 left-rail callout surface',
  );
  assert.match(
    source,
    /background: 'var\(--coral-light\)'/,
    'Caveat must use var(--coral-light), not a raw tint or Tailwind amber class',
  );
  assert.match(
    source,
    /borderColor: 'var\(--coral\)'/,
    'Caveat rail must use var(--coral), not a raw hex or unrelated palette',
  );
  // role="note" so AT users get the same "this is a sidebar callout"
  // cue sighted users get from the coral rail.
  assert.match(
    source,
    /data-testid="worked-pattern-ledger-caveat"[\s\S]{0,200}role="note"/,
    'Caveat must declare role="note" so AT readers receive the same callout cue',
  );
});

// ─── 4. "Show full α sum" toggle — aria-label and stable hook ────────────────
test('"Show full α sum" toggle exists with a stable data-action and aria-label', () => {
  // The toggle gates the stacked α sum across every pattern family.
  // Pin the data-action hook (so QA / future tests can locate it
  // without selector churn) and the aria-label (AT users get the same
  // "expand / collapse" cue sighted users get from the underlined
  // text label).
  assert.match(
    source,
    /data-action="toggle-full-alpha-sum"/,
    'Toggle button must carry data-action="toggle-full-alpha-sum"',
  );
  assert.match(
    source,
    /aria-label=\{showFullAlphaSum \? 'Hide the full alpha sum across every pattern family' : 'Show full α sum — stack every pattern-family contribution into the cumulative alpha'\}/,
    'Toggle button must carry an aria-label that flips with showFullAlphaSum state',
  );
  // The visible text on the button itself flips between
  // "show full α sum" and "hide full α sum" — pin the on-state phrase
  // so the alpha-sum surface is reachable by visible-text search too.
  assert.match(
    source,
    /show full α sum/,
    'Toggle button must surface the visible text "show full α sum" when collapsed',
  );
});

// ─── 5. Toggle announces expanded state via aria-expanded ────────────────────
test('"Show full α sum" toggle announces expanded state via aria-expanded', () => {
  // aria-expanded is the standard handshake for disclosure widgets
  // (WAI-ARIA APG). AT users need to know whether the panel is
  // currently open or closed without re-reading the visible label.
  // Pin the binding to showFullAlphaSum so the announcement stays in
  // lockstep with the actual rendered state.
  assert.match(
    source,
    /aria-expanded=\{showFullAlphaSum\}/,
    'Toggle button must announce expanded state via aria-expanded={showFullAlphaSum}',
  );
  // The state itself must be declared via useState(false) so the
  // panel starts collapsed (per V3.1 §35: default-off, opt-in).
  assert.match(
    source,
    /const \[showFullAlphaSum, setShowFullAlphaSum\] = useState\(false\)/,
    'TypedPartitionDemo must declare showFullAlphaSum via useState(false)',
  );
});

// ─── 6. Stacked α sum panel renders per-pattern contributions ────────────────
test('"Show full α sum" expansion renders the stacked per-pattern α sum', () => {
  // When the toggle flips on, the panel renders one
  // data-full-alpha-term entry per pattern family, joined into a
  // stacked sum that resolves to the cumulative α total.
  // Pin the container testid + the per-term hook so a future rename
  // of either fails fast.
  assert.match(
    source,
    /data-testid="worked-pattern-full-alpha-sum"/,
    'Full-α-sum panel must carry data-testid="worked-pattern-full-alpha-sum"',
  );
  assert.match(
    source,
    /data-full-alpha-term=\{row\.key\}/,
    'Each per-pattern term in the stacked sum must emit data-full-alpha-term with the pattern key',
  );
  // The panel must terminate the sum with the cumulative α total so
  // the prose closes the loop ("…= cumulativeAlpha"). Pin the equality
  // that joins the per-term contributions to cumulativeAlpha.
  assert.match(
    source,
    /\{cumulativeAlpha\}/,
    'Full-α-sum panel must close with the cumulativeAlpha total',
  );
  // The mini-ledger ledger helper enumerateMemberAssignments — pin its
  // existence + the example-limit constant so the prose "Member
  // assignments" list can never silently degrade to an empty bullet.
  assert.match(
    source,
    /function enumerateMemberAssignments\(partition, sizes, limit\)/,
    'enumerateMemberAssignments(partition, sizes, limit) helper must exist for the Member assignments list',
  );
  assert.match(
    source,
    /const MINI_LEDGER_EXAMPLE_LIMIT = 5/,
    'MINI_LEDGER_EXAMPLE_LIMIT must cap the Member assignments list at 5 examples',
  );
});
