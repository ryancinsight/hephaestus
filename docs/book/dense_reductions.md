# Dense Reductions

Dense reductions collapse one or more axes of a buffer into a scalar or
lower-rank result. Hephaestus implements them through two traits.

## `FullReductionOps`

Reduces an entire buffer to a single scalar:

```rust,ignore
let sum: f32 = device.reduce_sum(&input_buf)?;
let max: f32 = device.reduce_max(&input_buf)?;
```

## `AxisReductionOps`

Reduces along a specified axis:

```rust,ignore
let plan = device.plan_axis_reduction(input_shape, axis)?;
device.reduce_sum_axis(&plan, &input_buf, &mut output_buf)?;
```

`plan_axis_reduction` computes the dispatch shape and pass count ahead of
time. The plan is reusable across calls with the same shape.

## `AxisScanOps`

Cumulative sum/product along an axis:

```rust,ignore
let plan = device.plan_axis_scan(input_shape, axis)?;
device.scan_sum(&plan, &input_buf, &mut output_buf)?;
```

## `RetainedReductions`

`RetainedReductions` caches intermediate reduction state for multi-pass
algorithms (e.g., layer normalization mean/variance for the backward pass).
The retained state avoids re-computing the forward statistics during backward.

## `DenseVectorOps`

BLAS-1 style operations on flat buffers:

| Op | Description |
|----|-------------|
| `dot(a, b)` | Inner product |
| `axpy(alpha, x, y)` | `y = alpha*x + y` in-place |
| `norm2(x)` | Euclidean norm |
| `scale(alpha, x)` | Scale in-place |
