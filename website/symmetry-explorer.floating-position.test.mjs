import test from 'node:test';
import assert from 'node:assert/strict';
import {
  flipPosition,
} from './components/symmetry-aware-einsum-contractions/components/branchingViews/floatingPosition.js';

test('flipPosition: card fits to right + below click point — no flip', () => {
  const result = flipPosition({
    clickX: 100, clickY: 100,
    cardW: 400, cardH: 300,
    viewportW: 1280, viewportH: 800,
    padding: 12,
  });
  assert.equal(result.left, 100 + 12);
  assert.equal(result.top, 100 + 12);
});

test('flipPosition: card overflows right edge — flips to left of click', () => {
  const result = flipPosition({
    clickX: 1100, clickY: 100,
    cardW: 400, cardH: 300,
    viewportW: 1280, viewportH: 800,
    padding: 12,
  });
  // Default right placement would be 1100 + 12 = 1112, card extends to 1512 > 1280 → flip
  assert.equal(result.left, 1100 - 12 - 400); // left of click - padding - cardW
  assert.equal(result.top, 100 + 12);
});

test('flipPosition: card overflows bottom — flips to above click', () => {
  const result = flipPosition({
    clickX: 100, clickY: 700,
    cardW: 400, cardH: 300,
    viewportW: 1280, viewportH: 800,
    padding: 12,
  });
  // Default below would be 700 + 12 = 712, card extends to 1012 > 800 → flip
  assert.equal(result.left, 100 + 12);
  assert.equal(result.top, 700 - 12 - 300);
});

test('flipPosition: card overflows BOTH right and bottom — flips both', () => {
  const result = flipPosition({
    clickX: 1100, clickY: 700,
    cardW: 400, cardH: 300,
    viewportW: 1280, viewportH: 800,
    padding: 12,
  });
  assert.equal(result.left, 1100 - 12 - 400);
  assert.equal(result.top, 700 - 12 - 300);
});

test('flipPosition: card too tall for viewport — clamps top to padding', () => {
  // Card is 1000 tall, viewport 800. No fit. Clamp to top: padding.
  const result = flipPosition({
    clickX: 100, clickY: 400,
    cardW: 400, cardH: 1000,
    viewportW: 1280, viewportH: 800,
    padding: 12,
  });
  assert.equal(result.top, 12); // clamp to viewport padding
});

test('flipPosition: card too wide for viewport — clamps left to padding', () => {
  const result = flipPosition({
    clickX: 100, clickY: 100,
    cardW: 1500, cardH: 300,
    viewportW: 1280, viewportH: 800,
    padding: 12,
  });
  assert.equal(result.left, 12); // clamp to viewport padding
});
