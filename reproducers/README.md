# Matrix decomposition and gradient reproducers

Small, standalone examples for linking from issues. Each script prints observed
results alongside a mathematical expectation; gradient examples also include
controls or forward finite differences where applicable. These are diagnostic
scripts, so exit code zero means execution completed, **not** that Warp is correct.
No fixes are included.

## Run

From a built Warp checkout:

```sh
uv run reproducers/eig3_disconnected_loss.py
```

Replace the filename to run any example below. Alternatively, download one script
and run it in an environment with `warp-lang` and NumPy installed.

All scripts explicitly select CPU. Change `wp.set_device("cpu")` to
`wp.set_device("cuda:0")` to investigate CUDA separately.

Verified on Apple Silicon CPU with Warp `1.19.0.dev0`, Python `3.12.3`, NumPy
`2.5.3`.

## Findings

| Script | Expected result | Observed on the verified build |
|---|---|---|
| [eig3_gradients.py](eig3_gradients.py) | Eigenvalue-sum gradient `I`; reconstruction directional derivative `2`. | Repeated eigenvalues give NaNs. A gap of `1e-7` gives reconstruction AD `0.2`, while FD is `2`. Distinct-value controls agree. |
| [eig3_disconnected_loss.py](eig3_disconnected_loss.py) | `trace(A)` has gradient `I`, with or without unrelated diagnostics. | An `eig3` diagnostic recorded on the same tape changes all nine gradient entries to NaN. Loss remains `7`. |
| [eig3_masked_parameter.py](eig3_masked_parameter.py) | Shared-parameter gradients `6` and `0` for seeds `[0,1]` and `[0,0]`. | Both gradients, and a resulting SGD step, are NaN. |
| [svd2_forward.py](svd2_forward.py) | Reconstruction equals input; squared-norm gradient is `2A`; singular values are finite. | `-2I` reconstructs as `+2I`, with reversed gradient sign. Separate nonsingular inputs give NaN singular values in both float32 and float64. |
| [svd_gradients.py](svd_gradients.py) | At `2I`, reconstruction/stretch directional derivatives are `1`; polar derivative is `0.5`. | SVD2 and SVD3 give incorrect AD values. SVD2's polar FD is also wrong, reflecting a forward defect. |
| [svd_half.py](svd_half.py) | Squared-norm gradient at `2I` is `4I`. | Half-precision SVD2/SVD3 give NaN gradients; float32 controls agree within rounding error. |
| [svd3_elastic_energy.py](svd3_elastic_energy.py) | Shear derivative about `-0.991`. | AD is about `+1.596`; Warp and independent NumPy FD agree on the negative sign. |
| [qr3_rank_deficient.py](qr3_rank_deficient.py) | Squared Frobenius norms of `R` and `A` agree; gradient `2A`. | At `diag(0,0,3)`, AD is zero, while the `E22` FD is `6`. Full-rank control agrees. |
| [scalar_roots.py](scalar_roots.py) | At zero, derivatives of `sqrt(x*x)²` and `cbrt(x)³` are `0` and `1`; unused/zero-weight roots preserve gradient `1`. | All four cases give NaNs. Nonzero and squared-length controls agree. |
| [fem_eigenvalue_gradient.py](fem_eigenvalue_gradient.py) | Reconstruction directional derivative `2`. | AD is zero; FD is `2`. **Candidate:** the iterative solver's AD guarantees need clarification. |

## Relationships and scope

The disconnected diagnostic and masked-parameter cases extend the original
`eig3` issue (GH-2010): reciprocal zero eigenvalue gaps can turn zero adjoints
into NaNs, which then contaminate accumulated gradients. Nearby gaps also expose
finite incorrect derivatives due to gap regularization.

These findings are not all caused by `eig3`. `eig3` and SVD3 share a forward
Jacobi helper; SVD3 and QR3 share a QR helper, but their backward rules are
separate. SVD2 and the FEM eigenvalue solver are separate implementations.
Related failure patterns include singular intermediate derivatives, gap clamps,
half-precision reciprocal overflow, and rank-deficient inverses. SVD2 additionally
has independent forward defects: an invalid isotropic branch and cancellation
in the small-singular-value formula.

The elastic-energy example reduces the stretch/Hooke-energy formulas used by
the FEM benchmark to a standalone kernel. It demonstrates an incorrect energy
gradient, not a broken complete FEM solver; the benchmark's explicit force and
Hessian paths use an analytic strain derivative. The public FEM Dirichlet
normalization passed the separate forward controls in the wider investigation.

The examples concern smooth overall functions, even where individual decomposition
bases are nonunique. They demonstrate NaNs **and** finite wrong results; a NaN-only
fix would not cover the entire collection. Link individual scripts at a fixed
commit when reporting an issue, so the reproducer remains stable.
