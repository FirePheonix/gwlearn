# GWPCA Reference Comparison Checklist

This note tracks what `spml.decomposition.GWPCA` already matches against the
external reference implementation, what still differs, and where the remaining
work lives in the codebase.

## Current Target

Keep the `spml` package design:

- `libpysal`-based spatial weights
- `pandas` / `geopandas` inputs
- labeled `pandas` outputs
- optional global baseline model

while making the local GWPCA results line up with the reference outputs where
they should.

## Already Matching The Reference

- Local component loadings match the external reference up to PCA sign flips.
- Local explained variance proportions match the reference `local_pv` values.
- The comparison is covered by
  [spml/tests/test_decomposition.py](E:/gsoc-2026/gwlearn/spml/tests/test_decomposition.py)
  using the fixture in
  [spml/tests/data/gwpca_reference_fixture.json](E:/gsoc-2026/gwlearn/spml/tests/data/gwpca_reference_fixture.json).
- The matching setup is documented in
  [spml/tests/data/generate_gwpca_reference_fixture.R](E:/gsoc-2026/gwlearn/spml/tests/data/generate_gwpca_reference_fixture.R).

## Known Differences From The Reference

- Sign flips in loadings are expected and should not be treated as a mismatch.
- Reference agreement currently depends on matching preprocessing and fit setup
  exactly:
  standardized data, fixed-distance bandwidth, bisquare kernel, and
  `include_focal=True`.
- Raw local eigenvalue magnitudes are not asserted against the reference
  because the local
  covariance normalization differs between the current `spml` implementation
  and the shipped reference fixture.
- The current comparison still needs broader validation on additional spatial
  configurations and adaptive neighborhoods.

## Code Areas That Control Parity

- Weight construction:
  [spml/base.py](E:/gsoc-2026/gwlearn/spml/base.py)
  in `_BaseModel._build_weights`.
- Local covariance and eigendecomposition:
  [spml/decomposition/pca.py](E:/gsoc-2026/gwlearn/spml/decomposition/pca.py)
  in `GWPCA._fit_local`.
- Leave-one-out reconstruction path:
  [spml/decomposition/pca.py](E:/gsoc-2026/gwlearn/spml/decomposition/pca.py)
  in `GWPCA._compute_cv_score`.
- Public decomposition output shaping:
  [spml/decomposition/_base.py](E:/gsoc-2026/gwlearn/spml/decomposition/_base.py)
  in `components_`, `explained_variance_`, `scores_`, and related properties.

## Remaining Parity Work

1. Pin down the exact covariance normalization used by the external reference
   implementation if raw
   local variances are meant to match exactly.
2. Confirm neighborhood membership is identical for edge cases, especially
   around bandwidth boundaries and coplanar points.
3. Confirm `include_focal` semantics are identical to R for both fixed and
   adaptive neighborhoods.
4. Confirm `libpysal` bisquare weights match the reference implementation
   exactly at the cutoff
   and for any epsilon handling around adaptive bandwidths.
5. Expand the external reference test to cover additional outputs and
   geometries once
   broader parity is confirmed.
6. Keep the simple synthetic tests because they isolate weighted covariance,
   rank-deficient behavior, and CV math independently of the reference fixture.

## Suggested Priority

1. Validate loadings and variance proportions on more reference fixtures.
2. [x] Recheck adaptive bandwidth behavior against the reference fixture.
3. Only then decide whether exact raw variance parity is worth the extra
   normalization layer.

## Non-Goals

- Replacing `pandas` outputs with raw NumPy arrays.
- Dropping the global baseline just for reference compatibility.
- Changing the public `spml` API to imitate another implementation.
- Treating exact parity in every internal number as more important than package
  consistency.
