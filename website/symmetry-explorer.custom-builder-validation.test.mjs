/**
 * V3.1 §3 — C03 Custom Einsum Builder polish (L5.T2.16).
 *
 * Source-grep tests that lock in the eight inline-validation categories,
 * the cycle-notation preview under custom generators, and the Analyze
 * button's success cue + aria-label.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

function read(rel) {
  return readFileSync(resolve(__dirname, rel), 'utf-8');
}

const CHOOSER = 'components/symmetry-aware-einsum-contractions/components/ExampleChooser.jsx';
const VALIDATION_MESSAGES = 'components/symmetry-aware-einsum-contractions/engine/validationMessages.js';
const VALIDATION = 'components/symmetry-aware-einsum-contractions/engine/validation.js';

test('validationMessages declares the eight V3.1 inline-validation categories', () => {
  const src = read(VALIDATION_MESSAGES);
  // The canonical taxonomy lives in the engine module so the chooser, tests,
  // and any downstream copy can share one source of truth.
  assert.match(src, /export const VALIDATION_ERROR_CATEGORIES/);
  for (const category of [
    'unknown-operand',
    'duplicate-output-label',
    'repeated-input-label',
    'ellipsis-unsupported',
    'invalid-generator-axis',
    'incompatible-domain-move',
    'output-label-missing-in-input',
    'malformed-einsum',
  ]) {
    const pattern = new RegExp(`'${category}'`);
    assert.match(src, pattern, `category "${category}" missing from VALIDATION_ERROR_CATEGORIES`);
  }
});

test('ExampleChooser re-exports the eight validation categories', () => {
  const src = read(CHOOSER);
  // Tests + downstream callers (e.g. category-aware tooltips) read this
  // without reaching into engine/validationMessages.js directly.
  assert.match(src, /export const errorCategories = VALIDATION_ERROR_CATEGORIES/);
  assert.match(src, /import \{[^}]*VALIDATION_ERROR_CATEGORIES[^}]*\} from '\.\.\/engine\/validationMessages\.js'/);
});

test('engine surfaces ellipsis-unsupported and incompatible-domain-move builders', () => {
  // These two are V3.1 additions — every other category has a pre-existing
  // builder. Locking the new ones in by name keeps them from being silently
  // collapsed back into the lowercase / parse buckets later.
  const messagesSrc = read(VALIDATION_MESSAGES);
  assert.match(messagesSrc, /export function ellipsisUnsupportedError/);
  assert.match(messagesSrc, /export function incompatibleDomainMoveError/);
  assert.match(messagesSrc, /ELLIPSIS_UNSUPPORTED:\s*'ellipsis-unsupported'/);
  assert.match(messagesSrc, /INCOMPATIBLE_DOMAIN_MOVE:\s*'incompatible-domain-move'/);

  const validationSrc = read(VALIDATION);
  // The validator must actually invoke the new builders, not just import.
  assert.match(validationSrc, /ellipsisUnsupportedError\(/);
  assert.match(validationSrc, /incompatibleDomainMoveError\(/);
});

test('ExampleChooser renders inline error rows with data-error-category attribute', () => {
  const src = read(CHOOSER);
  // Every error row carries its category id so QA + downstream styling can
  // filter by category without parsing the human-readable message.
  assert.match(src, /data-error-category=\{error\.category \|\| 'malformed-einsum'\}/);
  assert.match(src, /data-error-code=\{error\.code\}/);
  // The error block is announced as a live region for SR users.
  assert.match(src, /role="alert"/);
  assert.match(src, /aria-live="polite"/);
});

test('ExampleChooser marks invalid input fields with aria-invalid', () => {
  const src = read(CHOOSER);
  // The four primary fields (subscripts, output, operands, variable name +
  // generators) all flip aria-invalid on when their key sits in errorFieldSet.
  assert.match(src, /aria-invalid=\{errorFieldSet\.has\('subscripts'\) \? 'true' : undefined\}/);
  assert.match(src, /aria-invalid=\{errorFieldSet\.has\('output'\) \? 'true' : undefined\}/);
  assert.match(src, /aria-invalid=\{errorFieldSet\.has\('operands'\) \? 'true' : undefined\}/);
  assert.match(src, /aria-invalid=\{errorFieldSet\.has\(varField\(idx, 'name'\)\) \? 'true' : undefined\}/);
  assert.match(src, /aria-invalid=\{errorFieldSet\.has\(varField\(idx, 'generators'\)\) \? 'true' : undefined\}/);
  // Fields with errors also get a red border (existing behavior we lock in).
  assert.match(src, /errorFieldSet\.has\('subscripts'\) \? 'border-red-300' : 'border-gray-200'/);
});

test('ExampleChooser renders a custom-generator cycle preview in cycle notation', () => {
  const src = read(CHOOSER);
  // Reuses the canonical cycle helpers — same vocabulary as SigmaLoop /
  // DiminoView / fullGroup.cycleNotation. The preview is wrapped with a
  // data-generator-cycle-preview attribute so QA can read it directly.
  assert.match(src, /function generatorPreviewText/);
  assert.match(src, /import \{ Permutation \} from '\.\.\/engine\/permutation\.js'/);
  assert.match(src, /import \{ parseCycleNotation, cyclesToArrayForm \} from '\.\.\/engine\/cycleParser\.js'/);
  assert.match(src, /perm\.cycleNotation\(labels\)/);
  assert.match(src, /data-generator-cycle-preview=\{previewText\}/);
});

test('Analyze button has aria-label and shows a success state cue', () => {
  const src = read(CHOOSER);
  // Aria-label gives SR users a stable, descriptive name even when the
  // visible label flips between "Analyze" and "Analyzed" on success.
  assert.match(src, /aria-label="Analyze custom einsum expression"/);
  // The success state lives in a useState hook that auto-clears after 2s.
  assert.match(src, /const \[analyzeSuccess, setAnalyzeSuccess\]/);
  assert.match(src, /setAnalyzeSuccess\(true\)/);
  // Visible-state cue: button data attribute + the inline status banner.
  assert.match(src, /data-analyze-success=\{analyzeSuccess \? 'true' : 'false'\}/);
  assert.match(src, /data-analyze-status="success"/);
  // The button text flips to a checkmark + "Analyzed" while the cue is on.
  assert.match(src, /Analyzed/);
});

test('All eight V3.1 categories appear in the validationMessages source verbatim', () => {
  const src = read(VALIDATION_MESSAGES);
  // Every existing builder also gained a `category:` key wiring it into the
  // V3.1 taxonomy. Spot-check a sample of the eight phrases by direct match
  // so a future refactor can't silently drop one.
  for (const phrase of [
    "category: 'unknown-operand'",
    "category: 'duplicate-output-label'",
    "category: 'repeated-input-label'",
    "category: 'ellipsis-unsupported'",
    "category: 'invalid-generator-axis'",
    "category: 'incompatible-domain-move'",
    "category: 'output-label-missing-in-input'",
    "category: 'malformed-einsum'",
  ]) {
    assert.ok(
      src.includes(phrase),
      `validationMessages.js missing wiring for ${phrase}`,
    );
  }
});
