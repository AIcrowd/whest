// V3.1 §46 Appendix Theorem Block — 4-field template + anchor + copy button.
//
// Pins the structural contract of components/AppendixTheoremBlock.jsx so the
// V3.1 narrative's appendix-block standardization (Condition / Claim /
// Reason / Main-page shortcut) doesn't silently regress to the pre-V3.1
// freeform paragraph layout. Also pins the demonstration migration in
// ExpressionLevelModal: B.1 (No detected product symmetry) is the canary
// case that proves the new template composes cleanly with the existing
// kicker eyebrow before the rest of B.x are migrated.
//
// Pins:
//   1. AppendixTheoremBlock declares the 4 new optional props
//      (condition, claim, reason, mainPageShortcut).
//   2. When those fields are set, the block renders the V3.1-standard
//      labels verbatim ("Condition.", "Claim.", "Reason.",
//      "Main-page shortcut.") so reviewers can scan cases consistently.
//   3. When `anchorId` is set, the block emits a <section id={anchorId}>
//      and a permalink anchor button so cases are linkable.
//   4. When `claim` is set, the block renders an inline "Copy" button
//      that copies the formula text via navigator.clipboard.
//   5. The copy button has an aria-label so it is reachable by AT users.
//   6. Backward compat: a children-only call still renders the original
//      italicised paragraph (no V3.1 fields needed).
//   7. Demonstration: ExpressionLevelModal's B.1 case is migrated to the
//      new AppendixTheoremBlock template with the canonical anchorId
//      "appendix-b-1".

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const root = new URL('./components/symmetry-aware-einsum-contractions/', import.meta.url);
const read = (relative) => fs.readFileSync(new URL(relative, root), 'utf8');

const blockSource = read('components/AppendixTheoremBlock.jsx');
const modalSource = read('components/ExpressionLevelModal.jsx');

// ─── 1. AppendixTheoremBlock accepts the 4 new props ─────────────────────────
test('AppendixTheoremBlock accepts the 4 new V3.1 props (condition, claim, reason, mainPageShortcut)', () => {
  // The signature must declare all four new optional props so the V3.1
  // 4-field template can be passed by callers. Defaults are `null` so the
  // legacy children-based rendering continues to work when fields are
  // omitted. Pin the literal default `= null` to prevent accidental
  // type/required regression — V3.1 §46 says these are all optional.
  assert.match(blockSource, /condition\s*=\s*null/,
    'AppendixTheoremBlock must accept an optional `condition` prop');
  assert.match(blockSource, /claim\s*=\s*null/,
    'AppendixTheoremBlock must accept an optional `claim` prop');
  assert.match(blockSource, /reason\s*=\s*null/,
    'AppendixTheoremBlock must accept an optional `reason` prop');
  assert.match(blockSource, /mainPageShortcut\s*=\s*null/,
    'AppendixTheoremBlock must accept an optional `mainPageShortcut` prop');
});

// ─── 2. AppendixTheoremBlock renders V3.1-standard labels ────────────────────
test('AppendixTheoremBlock renders V3.1-standard labels when fields are provided', () => {
  // The V3.1 §46 spec calls for exact label strings so reviewers can scan.
  // Render the labels verbatim (with trailing period) inside <span> tags
  // styled as bold prefixes — same visual grammar as the legacy "Kind."
  // heading. Drift here breaks reader expectations across cases.
  assert.match(blockSource, />Condition\.</,
    'AppendixTheoremBlock must emit a literal "Condition." label');
  assert.match(blockSource, />Claim\.</,
    'AppendixTheoremBlock must emit a literal "Claim." label');
  assert.match(blockSource, />Reason\.</,
    'AppendixTheoremBlock must emit a literal "Reason." label');
  assert.match(blockSource, />Main-page shortcut\.</,
    'AppendixTheoremBlock must emit a literal "Main-page shortcut." label');
});

// ─── 3. AppendixTheoremBlock accepts anchorId and renders an <a> ─────────────
test('AppendixTheoremBlock accepts anchorId and renders an <a> anchor element', () => {
  // The V3.1 §46 spec requires "Each theorem block has anchor link" so
  // method badges in the main page can deep-link into the appendix. The
  // implementation wraps the block in a <section id={anchorId}> and
  // renders an inline <a href={`#${anchorId}`}> button that copies the
  // permalink — same UX as ExplorerSectionCard's AnchorLink.
  assert.match(blockSource, /anchorId\s*=\s*null/,
    'AppendixTheoremBlock must accept an optional `anchorId` prop');
  assert.match(blockSource, /<section\s+id=\{anchorId\}/,
    'AppendixTheoremBlock must wrap in a <section id={anchorId}> when anchorId is set');
  assert.match(blockSource, /href=\{`#\$\{anchorId\}`\}/,
    'AppendixTheoremBlock must emit an <a href="#anchorId"> anchor link');
});

// ─── 4. Copy button rendered when claim is set ───────────────────────────────
test('AppendixTheoremBlock renders a CopyFormulaButton when claim is set', () => {
  // The V3.1 §46 spec marks the copy button as optional but worth shipping
  // because the Claim formula is the most-likely-to-be-pasted content in
  // the appendix. The implementation routes the claim text through a
  // CopyFormulaButton helper that uses navigator.clipboard.writeText so
  // readers can paste the raw $...$ source straight into a paper.
  assert.match(blockSource, /function CopyFormulaButton\(/,
    'AppendixTheoremBlock must declare a CopyFormulaButton helper');
  assert.match(blockSource, /<CopyFormulaButton text=\{claim\}/,
    'AppendixTheoremBlock must render <CopyFormulaButton text={claim} /> next to the claim');
  assert.match(blockSource, /navigator\.clipboard\.writeText\(text\)/,
    'CopyFormulaButton must call navigator.clipboard.writeText to copy the formula');
});

// ─── 5. Buttons have aria-label so they are reachable by AT users ────────────
test('AppendixTheoremBlock copy button + anchor link both have aria-label', () => {
  // V3.1 §46 calls out keyboard-focusable + aria-label as a requirement
  // because the buttons render as compact glyphs ("#", "copy") that are
  // meaningless without an accessible name. <button> and <a> are both
  // focusable by default, so the aria-label is the load-bearing piece.
  assert.match(blockSource, /aria-label=\{copied \? 'Formula copied' : 'Copy formula to clipboard'\}/,
    'CopyFormulaButton must have aria-label "Copy formula to clipboard"');
  assert.match(blockSource, /aria-label=\{copied \? 'Link copied' : 'Copy link to this theorem block'\}/,
    'AnchorLinkButton must have aria-label "Copy link to this theorem block"');
});

// ─── 6. Backward compat: children-only call still renders ────────────────────
test('AppendixTheoremBlock with only children prop still renders the legacy paragraph', () => {
  // The legacy call shape (kind / lead / children) must continue to work
  // because most appendix sections have not migrated to the V3.1 template
  // yet. The block detects "no V3.1 fields set" and falls through to the
  // original italicised <p> layout. Pin both the legacy children gate and
  // the italicised span to prevent silent regression.
  assert.match(blockSource, /children\s*!=\s*null/,
    'AppendixTheoremBlock must gate the legacy paragraph on children != null');
  assert.match(blockSource, /<span className="italic">/,
    'Legacy paragraph must render its body inside an italic <span>');
  // The default props (kind, lead) must continue to be supported. `lead`
  // still defaults to null so legacy callers that omit it stay unchanged.
  assert.match(blockSource, /lead\s*=\s*null/,
    'AppendixTheoremBlock must keep `lead = null` default for legacy callers');
});

// ─── 7. Modal demonstrates the new template on B.1 ───────────────────────────
test('ExpressionLevelModal migrates B.1 (No detected product symmetry) to the new template', () => {
  // The demo migration proves the template composes cleanly with the
  // existing per-case kicker eyebrow ("B.1 — No detected product symmetry")
  // and that an anchorId of "appendix-b-1" is canonical. Other B.x cases
  // remain on the legacy paragraph layout for incremental migration; this
  // pin is the canary so reviewers know one case has shipped end-to-end.
  assert.match(modalSource, /import AppendixTheoremBlock from '\.\/AppendixTheoremBlock\.jsx'/,
    'ExpressionLevelModal must import AppendixTheoremBlock');
  assert.match(modalSource, /<AppendixTheoremBlock[\s\S]{0,400}anchorId="appendix-b-1"/,
    'ExpressionLevelModal must render <AppendixTheoremBlock anchorId="appendix-b-1" /> for B.1');
  assert.match(modalSource, /<AppendixTheoremBlock[\s\S]{0,600}condition=\{conditionText\}/,
    'B.1 migration must pass the condition text into the template');
  assert.match(modalSource, /<AppendixTheoremBlock[\s\S]{0,600}claim=\{claimText\}/,
    'B.1 migration must pass the claim text into the template');
  assert.match(modalSource, /<AppendixTheoremBlock[\s\S]{0,600}reason=\{reasonText\}/,
    'B.1 migration must pass the reason text into the template');
  assert.match(modalSource, /<AppendixTheoremBlock[\s\S]{0,600}mainPageShortcut=/,
    'B.1 migration must pass a tooltip-sized mainPageShortcut into the template');
});
