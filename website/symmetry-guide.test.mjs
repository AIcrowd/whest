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

  assert.match(source, /symmetric_axes=\(0, 1\)/);
  assert.match(source, /symmetric_axes=\[\(0, 1\), \(2, 3\)\]/);
  assert.match(
    source,
    /PermutationGroup\.symmetric\(3, axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /PermutationGroup\.cyclic\(3, axes=\(0, 1, 2\)\)/,
  );
  assert.match(
    source,
    /PermutationGroup\.dihedral\(4, axes=\(0, 1, 2, 3\)\)/,
  );
  assert.match(
    source,
    /flops\.PermutationGroup\(\s*[\s\S]*flops\.Permutation\(flops\.Cycle\(0, 2\)\(1, 3\)\)/,
  );
});

test('symmetry guide includes the Reynolds section and flopscope-only indexing examples', async () => {
  const source = await readSymmetryGuide();

  assert.match(source, /Generating example data with the Reynolds operator/);
  assert.match(source, /fnp\.newaxis/);
  assert.match(source, /fnp\.array\(\[0, 1\]\)/);
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
  assert.match(source, /Unary pointwise ops preserve symmetry-aware costs and keep the same symmetric axes/);
  assert.match(source, /non-full groups such as `C_k` or `D_k`, the current implementation widens the metadata to full symmetry on those axes/);
  assert.match(source, /`fnp\.exp\(c3_tensor\)` still gets symmetry-aware cost savings, but the result is currently tagged as full symmetry on `\(0, 1, 2\)`/);
});

test('symmetry guide representative propagation claims match runtime behavior', async () => {
  const stdout = await runPythonInRepo(`
import json
import flopscope as flops
import flopscope.numpy as fnp

flops.configure(symmetry_warnings=False)

def symmetrize_with_group(shape, symmetry_group):
    raw_data = fnp.random.randn(*shape)
    group_axes = (
        symmetry_group.axes
        if symmetry_group.axes is not None
        else tuple(range(symmetry_group.degree))
    )
    accumulated = fnp.zeros_like(raw_data)

    for group_element in symmetry_group.elements():
        axis_permutation = list(range(len(shape)))
        for group_index, tensor_axis in enumerate(group_axes):
            axis_permutation[tensor_axis] = group_axes[group_element.array_form[group_index]]
        accumulated = accumulated + fnp.transpose(raw_data, axis_permutation)

    return flops.as_symmetric(
        accumulated / symmetry_group.order(),
        symmetry=symmetry_group,
    )

s2_matrix = flops.as_symmetric(
    fnp.ones((6, 6)),
    symmetric_axes=(0, 1),
)
row_slice = s2_matrix[0]
advanced_index_slice = s2_matrix[fnp.array([0, 1])]

s3_group = flops.PermutationGroup.symmetric(3, axes=(0, 1, 2))
c3_group = flops.PermutationGroup.cyclic(3, axes=(0, 1, 2))
c4_group = flops.PermutationGroup.cyclic(4, axes=(0, 1, 2, 3))

s3_tensor = symmetrize_with_group((4, 4, 4), s3_group)
c3_tensor = symmetrize_with_group((4, 4, 4), c3_group)
c4_tensor = symmetrize_with_group((4, 4, 4, 4), c4_group)
d4_group = flops.PermutationGroup.dihedral(4, axes=(0, 1, 2, 3))
d4_tensor = symmetrize_with_group((4, 4, 4, 4), d4_group)

s3_slice = s3_tensor[:, :, 0]
c3_slice = c3_tensor[:, :, 0]
c4_reduced = fnp.sum(c4_tensor, axis=(1, 3))
intersection_tensor = fnp.add(s3_tensor, c3_tensor)
unary_c3 = fnp.exp(c3_tensor)
unary_d4 = fnp.exp(d4_tensor)

with flops.BudgetContext(flop_budget=10**8) as budget:
    repeated_input = fnp.ones((5, 3))
    repeated_einsum = fnp.einsum(
        'ki,kj->ij',
        repeated_input,
        repeated_input,
        symmetric_axes=[(0, 1)],
    )
    repeated_einsum_cost = budget.flops_used

with flops.BudgetContext(flop_budget=10**8) as budget:
    distinct_left = fnp.ones((5, 3))
    distinct_right = fnp.ones((5, 3))
    distinct_einsum = fnp.einsum(
        'ki,kj->ij',
        distinct_left,
        distinct_right,
        symmetric_axes=[(0, 1)],
    )
    distinct_einsum_cost = budget.flops_used

left_tensor = flops.as_symmetric(
    fnp.ones((3, 3, 5, 5)),
    symmetric_axes=[(0, 1), (2, 3)],
)
right_tensor = flops.as_symmetric(
    fnp.ones((3, 3, 5, 5)),
    symmetric_axes=[(0, 1)],
)
shared_group_tensor = fnp.add(left_tensor, right_tensor)

stretched_s2_tensor = flops.as_symmetric(
    fnp.ones((1, 1, 4)),
    symmetry=flops.PermutationGroup.symmetric(2, axes=(0, 1)),
)
plain_tensor = fnp.ones((3, 3, 4))
broadcast_sum = fnp.add(stretched_s2_tensor, plain_tensor)

print(json.dumps({
    "row_slice_type": type(row_slice).__name__,
    "advanced_index_slice_type": type(advanced_index_slice).__name__,
    "s3_slice_type": type(s3_slice).__name__,
    "s3_slice_orders": [group.order() for group in s3_slice.symmetry_info.groups],
    "s3_slice_axes": [group.axes for group in s3_slice.symmetry_info.groups],
    "c3_slice_type": type(c3_slice).__name__,
    "c3_slice_has_symmetry": hasattr(c3_slice, "symmetry_info"),
    "c4_reduced_type": type(c4_reduced).__name__,
    "c4_reduced_orders": [group.order() for group in c4_reduced.symmetry_info.groups],
    "c4_reduced_axes": [group.axes for group in c4_reduced.symmetry_info.groups],
    "intersection_type": type(intersection_tensor).__name__,
    "intersection_orders": [group.order() for group in intersection_tensor.symmetry_info.groups],
    "shared_group_type": type(shared_group_tensor).__name__,
    "shared_group_axes": [group.axes for group in shared_group_tensor.symmetry_info.groups],
    "broadcast_has_symmetry": hasattr(broadcast_sum, "symmetry_info"),
    "unary_c3_orders": [group.order() for group in unary_c3.symmetry_info.groups],
    "unary_d4_orders": [group.order() for group in unary_d4.symmetry_info.groups],
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
    s3_slice_orders: [2],
    s3_slice_axes: [[0, 1]],
    c3_slice_type: 'ndarray',
    c3_slice_has_symmetry: false,
    c4_reduced_type: 'SymmetricTensor',
    c4_reduced_orders: [2],
    c4_reduced_axes: [[0, 1]],
    intersection_type: 'SymmetricTensor',
    intersection_orders: [3],
    shared_group_type: 'SymmetricTensor',
    shared_group_axes: [[0, 1]],
    broadcast_has_symmetry: false,
    unary_c3_orders: [6],
    unary_d4_orders: [24],
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
