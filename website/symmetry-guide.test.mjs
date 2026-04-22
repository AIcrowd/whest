import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const websiteRoot = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.dirname(websiteRoot);
const symmetryGuidePath = path.join(
  websiteRoot,
  'content/docs/guides/symmetry.mdx',
);

async function readSymmetryGuide() {
  return readFile(symmetryGuidePath, 'utf8');
}

async function runPythonInRepo(source) {
  const { stdout } = await execFileAsync(
    'uv',
    ['run', 'python', '-c', source],
    {
      cwd: repoRoot,
      maxBuffer: 1024 * 1024,
    },
  );

  return stdout;
}

test('symmetry guide documents every declaration style', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /symmetry=we\.SymmetryGroup\.symmetric\(axes=\(0, 1\)\)/);
  assert.match(source, /symmetry=we\.SymmetryGroup\.young\(blocks=\(\(0, 1\), \(2, 3\)\)\)/);
  assert.match(
    source,
    /SymmetryGroup\.symmetric\(axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /SymmetryGroup\.cyclic\(axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /SymmetryGroup\.dihedral\(axes=\(0, 1, 2, 3\)\)/,
  );
  assert.match(
    source,
    /we\.SymmetryGroup\.from_generators\(\s*\[\[2, 3, 0, 1\]\]/,
  );
});

test('symmetry guide includes the Reynolds section and whest-only indexing examples', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /Generating example data with the Reynolds operator/);
  assert.match(source, /we\.newaxis/);
  assert.match(source, /we\.array\(\[0, 1\]\)/);
});

test('symmetry guide avoids overclaiming Reynolds helper exactness', async () => {
  const source = await readSymmetryGuide();

  assert.doesNotMatch(source, /How to generate exact example tensors/);
  assert.doesNotMatch(
    source,
    /Generating exact example data with the Reynolds operator/,
  );
  assert.doesNotMatch(
    source,
    /it produces data that exactly satisfies the declared symmetry/,
  );
  assert.match(source, /validation tolerances/);
});

test('symmetry guide explains why the einsum savings example is cheaper', async () => {
  const source = await readSymmetryGuide();

  assert.doesNotMatch(source, /the output is symmetric in `\(0, 1\)`/);
  assert.match(source, /repeated `x` operand/);
  assert.match(
    source,
    /does not reduce that `einsum`'s own cost by itself/,
  );
});

test('symmetry guide documents the current unary pointwise caveat for non-full groups', async () => {
  const source = await readSymmetryGuide();

  assert.doesNotMatch(source, /unary pointwise ops keep the same groups/);
  assert.match(source, /Unary pointwise ops preserve symmetry-aware costs/);
  assert.match(source, /non-full groups such as `C_k`, `D_k`, or custom groups the current implementation widens the metadata to full symmetry on those axes/);
  assert.match(source, /non-full groups such as `C_k`, `D_k`, or custom groups/);
});

test('symmetry guide representative propagation claims match runtime behavior', async () => {
  const stdout = await runPythonInRepo(`
import json
import whest as we

we.configure(symmetry_warnings=False)

def symmetrize_with_group(shape, symmetry_group):
    raw_data = we.random.randn(*shape)
    group_axes = (
        symmetry_group.axes
        if symmetry_group.axes is not None
        else tuple(range(symmetry_group.degree))
    )
    accumulated = we.zeros_like(raw_data)

    for group_element in symmetry_group.elements():
        axis_permutation = list(range(len(shape)))
        for group_index, tensor_axis in enumerate(group_axes):
            axis_permutation[tensor_axis] = group_axes[group_element.array_form[group_index]]
        accumulated = accumulated + we.transpose(raw_data, axis_permutation)

    return we.as_symmetric(
        accumulated / symmetry_group.order(),
        symmetry=symmetry_group,
    )

s2_matrix = we.as_symmetric(
    we.ones((6, 6)),
    symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
)
row_slice = s2_matrix[0]
advanced_index_slice = s2_matrix[we.array([0, 1])]

s3_group = we.SymmetryGroup.symmetric(axes=(0, 1, 2))
c3_group = we.SymmetryGroup.cyclic(axes=(0, 1, 2))
c4_group = we.SymmetryGroup.cyclic(axes=(0, 1, 2, 3))

s3_tensor = symmetrize_with_group((4, 4, 4), s3_group)
c3_tensor = symmetrize_with_group((4, 4, 4), c3_group)
c4_tensor = symmetrize_with_group((4, 4, 4, 4), c4_group)
d4_group = we.SymmetryGroup.dihedral(axes=(0, 1, 2, 3))
d4_tensor = symmetrize_with_group((4, 4, 4, 4), d4_group)

s3_slice = s3_tensor[:, :, 0]
c3_slice = c3_tensor[:, :, 0]
c4_reduced = we.sum(c4_tensor, axis=(1, 3))
intersection_tensor = we.add(s3_tensor, c3_tensor)
unary_c3 = we.exp(c3_tensor)
unary_d4 = we.exp(d4_tensor)

with we.BudgetContext(flop_budget=10**8) as budget:
    repeated_input = we.ones((5, 3))
    repeated_einsum = we.einsum(
        'ki,kj->ij',
        repeated_input,
        repeated_input,
        symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    repeated_einsum_cost = budget.flops_used

with we.BudgetContext(flop_budget=10**8) as budget:
    distinct_left = we.ones((5, 3))
    distinct_right = we.ones((5, 3))
    distinct_einsum = we.einsum(
        'ki,kj->ij',
        distinct_left,
        distinct_right,
        symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
    )
    distinct_einsum_cost = budget.flops_used

left_tensor = we.as_symmetric(
    we.ones((3, 3, 5, 5)),
    symmetry=we.SymmetryGroup.young(blocks=((0, 1), (2, 3))),
)
right_tensor = we.as_symmetric(
    we.ones((3, 3, 5, 5)),
    symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
)
shared_group_tensor = we.add(left_tensor, right_tensor)

stretched_s2_tensor = we.as_symmetric(
    we.ones((1, 1, 4)),
    symmetry=we.SymmetryGroup.symmetric(axes=(0, 1)),
)
plain_tensor = we.ones((3, 3, 4))
broadcast_sum = we.add(stretched_s2_tensor, plain_tensor)

print(json.dumps({
    "row_slice_type": type(row_slice).__name__,
    "advanced_index_slice_type": type(advanced_index_slice).__name__,
    "s3_slice_type": type(s3_slice).__name__,
    "s3_slice_order": s3_slice.symmetry.order(),
    "s3_slice_axes": list(s3_slice.symmetry.axes),
    "c3_slice_type": type(c3_slice).__name__,
    "c3_slice_has_symmetry": getattr(c3_slice, "symmetry", None) is not None,
    "c4_reduced_type": type(c4_reduced).__name__,
    "c4_reduced_order": c4_reduced.symmetry.order(),
    "c4_reduced_axes": list(c4_reduced.symmetry.axes),
    "intersection_type": type(intersection_tensor).__name__,
    "intersection_order": intersection_tensor.symmetry.order(),
    "shared_group_type": type(shared_group_tensor).__name__,
    "shared_group_axes": list(shared_group_tensor.symmetry.axes),
    "broadcast_has_symmetry": getattr(broadcast_sum, "symmetry", None) is not None,
    "unary_c3_order": unary_c3.symmetry.order(),
    "unary_d4_order": unary_d4.symmetry.order(),
    "repeated_einsum_cost": repeated_einsum_cost,
    "distinct_einsum_cost": distinct_einsum_cost,
    "repeated_einsum_type": type(repeated_einsum).__name__,
    "distinct_einsum_type": type(distinct_einsum).__name__,
}))
`);
  const runtimeBehavior = JSON.parse(stdout.trim());

  assert.deepEqual(runtimeBehavior, {
    row_slice_type: 'ndarray',
    advanced_index_slice_type: 'ndarray',
    s3_slice_type: 'SymmetricTensor',
    s3_slice_order: 2,
    s3_slice_axes: [0, 1],
    c3_slice_type: 'ndarray',
    c3_slice_has_symmetry: false,
    c4_reduced_type: 'SymmetricTensor',
    c4_reduced_order: 2,
    c4_reduced_axes: [0, 1],
    intersection_type: 'SymmetricTensor',
    intersection_order: 3,
    shared_group_type: 'SymmetricTensor',
    shared_group_axes: [0, 1],
    broadcast_has_symmetry: false,
    unary_c3_order: 6,
    unary_d4_order: 24,
    repeated_einsum_cost: 30,
    distinct_einsum_cost: 45,
    repeated_einsum_type: 'SymmetricTensor',
    distinct_einsum_type: 'SymmetricTensor',
  });
});

test('symmetry guide includes all major propagation sections', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /## Symmetry propagation at a glance/);
  assert.match(source, /## Slicing rules/);
  assert.match(source, /## Reduction rules/);
  assert.match(source, /## Binary pointwise ops and broadcasting/);
  assert.match(source, /## Warnings, conservative behavior, and re-tagging/);
  assert.match(source, /## Under the hood/);
});
