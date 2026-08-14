# ADR 0021 (hephaestus): Metal fluent linear-algebra parity

- Status: Accepted
- Class: [minor]
- Date: 2026-07-25

## Context

WGPU, CUDA, and ROCm expose the same fluent rank-2 matrix trait families:
operand conversion, products, norms, decompositions, solves, matrix
properties, and matrix functions. Metal exposes the corresponding direct
operations but does not provide those trait entry points, so generic fluent
callers cannot use the common method shape across all providers.

## Decision

Add Metal-owned `AsGpuMatrixOperand`, `MatrixProduct`, `MatrixNorm`,
`MatrixDecompose`, `MatrixSolve`, `MatrixProperties`, and `MatrixFunction`
traits. Their device and result types are Metal-specific for immediate
operations. Decomposition methods retain the existing shared WGPU-backed
factorization handle types because those handles already execute through the
Metal-selected WGPU context; solve methods adapt their WGPU buffer arguments at
the Metal boundary and return `MetalBuffer` values.

Each trait delegates to an existing Metal application operation. The change
adds no algorithm copy, device-native kernel, CPU fallback, or dynamic-rank
adapter.

## Alternatives rejected

- Keep only direct Metal functions: this preserves a concrete public
  capability gap for fluent callers.
- Re-export the WGPU traits unchanged: their signatures require
  `WgpuDevice`/`WgpuBuffer` contracts and would not provide a Metal-owned
  result surface.
- Copy WGPU algorithms into Metal: this duplicates provider logic instead of
  using the existing native Metal-selected WGPU path.

## Verification

The Metal contract exercises fluent product, norm, property, function, solve,
inverse, determinant, pseudoinverse, and LU decomposition calls against closed
form values. The macOS Metal CI lane runs the feature, warning-denied Clippy,
Nextest, doctest, and rustdoc gates. At code head `8491592`, CUDA run
`30179437377` / job `89733458943` passed in 7m24s, ROCm run `30179437392` /
job `89733459927` passed in 6m01s, and Metal run `30179437384` / job
`89733463248` passed in 5m09s. The AMD and NVIDIA required-device jobs were
skipped because hosted GPU labels were unavailable. Earlier Metal failures
were corrected at the private module export and default-feature unused import
diagnostics before this evidence was accepted.
