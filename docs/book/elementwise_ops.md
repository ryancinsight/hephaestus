# Elementwise Operations

`ElementwiseOps<D, T>` is the device-neutral seam for unary, binary, and
scalar-aware operations over `StridedView` operands. `D` is a
`ComputeDevice`; `T` is an `eunomia::Pod` element type.

The operation expression is a type parameter constrained by the backend's
`KernelDialect`. `UnaryExpr`, `BinaryExpr`, and `TypedBinaryExpr`
therefore describe the operation without pretending that WGSL, CUDA C, HIP C,
and Metal share one shader syntax.

## One-shot and prepared calls

The trait's one-shot methods are `unary_into` and `binary_into`. They
validate the input/output shapes and layouts, reject an aliased output, prepare
the backend resources, and dispatch once. Repeated work uses
`prepare_unary_into` or `prepare_binary_into`, followed by
`dispatch_unary` or `dispatch_binary`; the prepared associated types are
lifetime-parameterized so a backend can borrow its operands without a boxed
trait object.

Empty views are no-ops. Binary views must satisfy the backend's shape
compatibility contract: implementations may require equal shapes or support
the broadcast forms they validate. Output views may not alias an input. These
are operation contracts rather than backend-specific conventions.

Floating-point special-value behavior is stated by the selected
`KernelDialect::IEEE_SPECIAL_VALUES` capability. A caller that needs a
particular NaN or infinity result must select a dialect/backend contract that
provides it and test that contract explicitly.
