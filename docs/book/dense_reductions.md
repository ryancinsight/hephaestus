# Dense Reductions

Hephaestus separates full-array reduction from rank-2 axis reduction while
using the same typed device and strided-view boundaries.

## Full reduction

`FullReductionOps<D, T>::reduce_full_into` reduces an `N`-rank input into a
rank-1 output view of length one. The output shape is part of the contract, so
an arbitrary output buffer cannot receive a scalar by accident. Repeated calls
can use `prepare_reduce_full` and `dispatch_full`.

## Axis reduction

`AxisReductionOps<D, T>` provides `reduce_axis_into`,
`prod_axis_into`, `mean_axis_into`, `min_axis_into`, and
`max_axis_into`. The input is rank two; the reduced axis remains present at
length one. For example, reducing a `[3, 4]` input along axis `0` writes a
`[1, 4]` output. The prepared form is `prepare_reduce_axis_into` followed
by `dispatch_prepared`.

The seam validates the axis, output shape, strided layout, aliasing, and the
non-empty requirement for a mean. It reports failures through the typed
`HephaestusError` result rather than returning an invented scalar.

## Shared launch planning

The `plan_axis_reduction` function contains the backend-neutral launch
metadata calculation used by the accelerator implementations. Backends still
own allocation, device-specific validation, dialect, shader, and raw dispatch
details. Centralizing the launch metadata keeps the mathematical shape
contract in one implementation without claiming that backend orchestration is
otherwise identical.
