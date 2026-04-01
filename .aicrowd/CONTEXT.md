# mechestim - Mechanistic Estimation Challenge Context

> Synthesized from #arc-aicrowd Slack channel discussions between ARC (Alignment Research Center) and AIcrowd teams (Feb-Mar 2026).

## Project Overview

**mechestim** is a Python library that provides a constrained set of mathematical primitives for participants in ARC's Mechanistic Estimation Challenge. The challenge asks participants to design algorithms that estimate statistics of neural network outputs (specifically ReLU MLPs) more efficiently than brute-force sampling.

The library serves two purposes:
1. Provide participants with a controlled set of operations they can use
2. Count FLOPs for each operation to enforce a computational budget, so the competition is about **algorithmic innovation** rather than **performance engineering**

## Challenge Description

### The Problem
Given a randomly initialized ReLU MLP (with He initialization: zero mean, variance 2/width), participants must estimate the expected output (mean) at the **final layer only**, using fewer FLOPs than brute-force sampling would require.

### Network Parameters
- **Width (W):** 256
- **Depth (D):** Multiple depths being considered; D=4 and D=32 discussed as candidates (shallow vs deep regimes)
- **Activation:** ReLU
- **Weights:** Random Gaussian, zero mean, variance 2/width (He init)

### What Participants Submit
Participants submit an **estimator** function that takes network parameters and a FLOP budget, and returns an estimate of the expected output at the final layer. The estimator must use only the operations provided by the `mechestim` library.

### Scoring
- Primary metric: MSE (mean squared error) of the estimate vs ground truth
- Ground truth is computed via large-sample Monte Carlo (may be precomputed)
- Scoring is on the **final layer only** (not per-layer), simplifying the evaluation contract
- Geometric mean of MSE across test cases was discussed as the aggregation method
- An epsilon term (epsilon = 1/n where n = number of samples) is added before taking the geometric mean to avoid over-incentivizing driving already-small errors to zero
- Score relative to a sampling baseline at the same FLOP budget

### Known Baseline Algorithms
These are reference algorithms from ARC's research that participants would try to beat or improve upon:
- **meanprop**: Propagates mean statistics through the network layer by layer
- **covprop**: Propagates covariance statistics through the network
- **Edgeworth expansion**: More advanced technique involving higher-order moment propagation

## The mechestim Library - Architecture Decisions

### Core Design Decision (Agreed 2026-03-31)
Instead of unrestricted Python/NumPy/SciPy usage with wall-clock timing, the competition uses a **restricted operation set with FLOP counting**. This was chosen because:
- Wall-clock timing makes it a "performance engineering exercise" rather than an "algorithms research problem"
- FLOP counting is deterministic and hardware-independent
- A restricted op set forces participants to think algorithmically

### Agreed Operation Set (Leading Proposal from Paul Christiano, 2026-03-31)

The minimum viable set of operations:

1. **einsum** (with symmetrization support) - The primary workhorse
   - Must support correct FLOP counting for symmetrized einsums
   - E.g., `einsum('ai,bj,ck,abc->ijk', x, x, x, A)` should get optimal symmetrized FLOP count
   - Support for partially symmetric tensors (e.g., 4-tensor where dims (0,1) and (2,3) are separately symmetric)
   - Support for repeated index patterns like `'i->ii'` (equivalent to `torch.diag`)
   - Nearly all algorithms of interest can be expressed primarily with einsums

2. **Pointwise operations**: max, exp, log, divide, multiply, add
   - Need appropriate FLOP savings for symmetric tensors

3. **SVD** (top-k singular values/vectors)
   - FLOP cost: O(m*n*k) for top-k singular vectors of an m x n matrix
   - Considered important for enabling better algorithms
   - Exact constant factor on m*n*k may matter; suggestion to calibrate based on optimized GPU implementations

4. **Sparse matmul** - Paul mentioned this as potentially needed

### FLOP Counting Philosophy
- FLOP count is **conceptually decoupled** from actual implementation performance
- Each op has a `calculate_flops` function that computes the **analytical optimal FLOP count**, regardless of backend efficiency
- This means the FLOP budget is about algorithmic complexity, not runtime
- Both Paul and S.P. Mohanty agreed on this decoupling

### Budget Enforcement Model
- There is a **FLOP limit** for mechestim operations (defined by the sampling budget)
- There is a **wall-clock time limit** for everything outside mechestim ops
  - This time limit should be ~10% of the time an optimized implementation would take for the FLOP budget
  - This prevents participants from doing significant computation outside the tracked ops
- Any operation NOT in the allowlist gets rejected immediately

### Library Interface Design
The agreed interface pattern is:
```python
import mechestim

# Participants use mechestim's wrapped operations
mechestim.einsum('ij,jk->ik', A, B)
mechestim.exp(x)
mechestim.svd(A, k=10)
```

Alternative discussed: alias as `numpy`/`scipy` drop-in, but the explicit `mechestim.*` prefix was preferred because it forces participants to acknowledge they're working with a subset of ops.

An AI coding assistant skill/prompt will be provided to help participants convert their code to use only mechestim operations.

### Precomputation
- Algorithms typically have a precomputation step (computing coefficients, diagram sums, etc.) that only depends on width and epsilon
- This precomputation happens before the FLOP-counted forward pass
- After precomputation, almost everything reduces to einsums with pre-computed coefficient arrays

## Evaluation Infrastructure

### Runtime Environment
- **CPU-only** (no GPU) - chosen for better computational control and cheaper scaling
- Docker-based sandboxing controlled by AIcrowd
- Sandbox restrictions:
  - No direct NumPy/SciPy (must go through mechestim wrappers)
  - No subprocess, multiprocessing, cffi, ctypes
  - No compiler toolchains in the sandbox
  - Blanket wall-clock timeouts as additional safety net
- Participants can scale with more CPU cores and time allotted

### Anti-Gaming Measures
- Automated FLOP counting handles good-faith participants
- Top leaderboard submissions get manual code review before prize allocation
- LLM-as-a-judge code review for rule adherence is being considered
- Gaming surfaces documented: NumPy bypass, custom kernels, parallelism tricks, lookup tables
- Most mitigable via the controlled Docker image

### Evaluation Flow (Simplified from earlier circuit-based design)
1. Generate random MLP with specified width/depth
2. Participant's estimator receives network parameters + FLOP budget
3. Estimator returns a single prediction for the final layer's expected output
4. Score = MSE vs ground truth (computed via large-sample Monte Carlo)
5. No per-layer streaming/yield required (simplified from earlier design)

## Timeline and Project History

### Evolution
1. **Feb 2026**: Project started as "circuit estimation" challenge using boolean circuits
   - Initial MVP repo: `alignment-research-center/circuit-estimation-mvp`
   - Baselines: meanprop, covprop on circuits
2. **Late Feb - Early Mar 2026**: Decision to pivot to **MLPs** instead of circuits
   - More relevant to neural network research
   - Targeting NeurIPS submission for a later round
3. **Mar 12, 2026**: Key decisions - final layer only, single sampling budget, MLP parameters (W256, multiple depths)
4. **Mar 19, 2026**: Paul shared `mlp_contest_mvp.py` - minimal MLP code
5. **Mar 26-27, 2026**: Backend profiling (PyTorch, JAX, NumPy compared); ops budget POC built
6. **Mar 27, 2026**: S.P. Mohanty presented ops ledger POC (PyTorch via TorchDispatchMode, JAX via jaxpr)
7. **Mar 31, 2026**: Agreement on mechestim library approach - einsum + pointwise + SVD as core ops

### Target Milestones
- Beta version: mid-late March 2026 (delayed)
- Private beta: early-mid April 2026 (tied to ARC paper release)
- Warm-up round: used to collect feedback, ops set can evolve with backward compatibility
- NeurIPS submission for later contest round

## Key People

### ARC (Alignment Research Center)
- **Paul Christiano** - Challenge designer, algorithm expert, key decision maker on ops set and scoring
- **Jacob Hilton** - Project coordination from ARC side, technical feedback
- **Wilson Wu** - Algorithm implementation, particularly Edgeworth expansion work
- **Victor** - ARC team member
- **George Robinson** - ARC team member
- **Mike Winer** - ARC team member
- **Eric Neyman** - ARC team member
- **Kyle Scott** - ARC team member

### AIcrowd
- **S.P. Mohanty** - Lead technical POC from AIcrowd, building the mechestim library and evaluation infrastructure
- **Sneha Nanavati** - Challenge comms, rules, and coordination
- **Harshita Khera** - SOW/contract and challenge rules coordination

## Key Repos
- `alignment-research-center/circuit-estimation-mvp` - Original circuit-based MVP (now superseded by MLP focus)
- `AIcrowd/circuit-estimation-challenge-internal` - AIcrowd's internal prototype (circuit era, includes React visualizer)
- `AIcrowd/network-estimation-ops-ledger` - POC for operation budgeting (PyTorch + JAX)
- `AIcrowd/network-estimation-challenge-internal` - Updated internal codebase
- Challenge assets repo (working title): `network-estimation-challenge-assets`

## Open Questions and Implementation Notes

### For mechestim Library Implementation
1. **Exact op set**: Paul + Wilson Wu to finalize after in-person discussion about Edgeworth expansion needs
2. **Sparse operations**: Paul hopes these aren't needed, but not confirmed
3. **Symmetrization details**: Need runtime checks that tensors claimed symmetric actually are, or static analysis
4. **SVD cost constant**: Whether to use exactly m*n*k or calibrate based on GPU implementation benchmarks
5. **Precomputation boundary**: How to separate precomputation (not FLOP-counted) from the estimation pass (FLOP-counted)
6. **Error handling**: What happens when participants call unsupported ops - currently "rejected immediately"

### For Evaluation Pipeline
1. **Ground truth computation**: May need more samples; if too slow to run locally, precompute
2. **Depth configurations**: D=4 and D=32 at W=256 were discussed but not finalized
3. **FLOP budget size**: Needs to be set relative to sampling baseline
4. **Warm-up round feedback loop**: Need process for participants to request new ops

### Backend Profiling Results (Mar 26, 2026)
- PyTorch and JAX perform best compared to NumPy
- Matmul performance saturates due to BLAS backend delegation
- ReLU performance varies across backends
- Framework-specific dispatching overhead is non-significant
- Dashboard: `nestim-profiling-266509514657.s3.amazonaws.com/dashboards/2026-03-26-jax-jit/dashboard.html`
