import test from 'node:test';
import assert from 'node:assert/strict';
import { encodePlaygroundState, decodePlaygroundState } from './components/symmetry-aware-einsum-contractions/lib/urlState.js';

test('round-trips a playground state', () => {
  const state = {
    subscripts: 'ij,jk',
    output: 'ik',
    operands: [
      { name: 'A', rank: 2, symmetry: 'symmetric' },
      { name: 'B', rank: 2, symmetry: 'none' },
    ],
    labelSizes: { 'i,j,k': 3 },
  };
  const encoded = encodePlaygroundState(state);
  const decoded = decodePlaygroundState(encoded);
  assert.deepEqual(decoded, state);
});

test('decode returns null on malformed URL state', () => {
  assert.equal(decodePlaygroundState('not-a-valid-payload'), null);
  assert.equal(decodePlaygroundState(''), null);
  assert.equal(decodePlaygroundState(null), null);
});

test('encodePlaygroundState returns a base64-like string', () => {
  const state = { subscripts: 'a', output: 'a', operands: [], labelSizes: {} };
  const encoded = encodePlaygroundState(state);
  assert.equal(typeof encoded, 'string');
  assert.ok(encoded.length > 0);
});
