"""Debug: what does the oracle detect vs what Φ needs?"""

from math import comb

import numpy as np

from flopscope._opt_einsum._contract import contract_path
from flopscope._opt_einsum._subgraph_symmetry import SubgraphSymmetryOracle
from flopscope._opt_einsum._symmetry import unique_elements

n = 6
X = np.ones((n, n, n))

oracle = SubgraphSymmetryOracle([X, X], ["ijk", "ilm"], [None, None], "jklm")

# What symmetry does the oracle see for the merged subset {0, 1}?
sym = oracle.sym(frozenset({0, 1}))
print("Oracle output symmetry:", sym.output)
print("Oracle inner symmetry:", sym.inner)

# The contraction: ijk,ilm->jklm
# s=2 (j,k from op0), t=2 (l,m from op1), v=1 (i contracted)
# ω=5, all indices: {i, j, k, l, m}
all_indices = frozenset("ijklm")
output_indices = frozenset("jklm")
inner_indices = frozenset("i")
size_dict = dict.fromkeys("ijklm", n)

# What unique_elements gives with oracle-detected symmetry
combined_sym = list(sym.output or []) + list(sym.inner or [])
unique_oracle = unique_elements(all_indices, size_dict, combined_sym or None)
print(f"\nunique_elements (oracle combined sym): {unique_oracle}")
print(f"n^ω = {n**5}")
print(f"ratio: {unique_oracle / n**5:.4f}")

# What Φ needs: full ω=5 symmetry across all indices
# ((n, ω)) = C(n+ω-1, ω) = C(10, 5) = 252
full_sym = [frozenset((c,) for c in "ijklm")]  # all 5 in one group
unique_full = unique_elements(all_indices, size_dict, full_sym)
print(f"\nunique_elements (full ω-symmetry): {unique_full}")
print(f"((n={n}, ω=5)) = C({n + 4}, 5) = {comb(n + 4, 5)}")

# Φ cost with full symmetry
omega = 5
s, t, v = 2, 2, 1
add_factor = comb(omega, s) + comb(omega, t) + comb(omega, v)
phi_correct = unique_full * (1 + add_factor)
# lower order terms
phi_correct += comb(n + s + v - 1, s + v)  # ((n, s+v))
phi_correct += comb(n + t + v - 1, t + v)  # ((n, t+v))
phi_correct += comb(n + s + t - 1, s + t)  # ((n, s+t))
print(f"\nΦ cost (correct, full sym): {phi_correct}")
print(f"Dense cost: {2 * n**5}")
print(f"Φ / dense: {phi_correct / (2 * n**5):.4f}")

# What happens with oracle sym (potentially incomplete)?
unique_oracle_all = unique_elements(all_indices, size_dict, combined_sym or None)
phi_oracle = unique_oracle_all * (1 + add_factor)
phi_oracle += comb(n + s + v - 1, s + v)
phi_oracle += comb(n + t + v - 1, t + v)
phi_oracle += comb(n + s + t - 1, s + t)
print(f"\nΦ cost (oracle sym): {phi_oracle}")
print(f"Φ oracle / dense: {phi_oracle / (2 * n**5):.4f}")
