# Elementwise Operations

Elementwise ops apply a scalar function independently to each element of an
operand buffer. Hephaestus implements them through the `ElementwiseOps` trait.

## `ElementwiseOps` Trait

```rust,ignore
pub trait ElementwiseOps {
    fn elementwise_unary<Op: UnaryStorageKernel>(
        &self, op: Op, input: &dyn DeviceBuffer<f32>, output: &mut dyn DeviceBuffer<f32>,
    ) -> Result<()>;

    fn elementwise_binary<Op: BinaryStorageKernel>(
        &self, op: Op,
        lhs: &dyn DeviceBuffer<f32>, rhs: &dyn DeviceBuffer<f32>,
        output: &mut dyn DeviceBuffer<f32>,
    ) -> Result<()>;
}
```

## Built-In Op Types

**Arithmetic:** `AddOp`, `SubOp`, `MulOp`, `DivOp`, `NegOp`, `AbsOp`

**Math:** `SqrtOp`, `ExpOp`, `LogOp`, `SinOp`, `CosOp`, `TanhOp`

**Activations:** `ReluOp`, `GeluOp`, `SigmoidOp`, `SiluOp`, `MishOp`,
`EluOp`, `HardtanhOp`, `ThresholdOp`

**Gradient variants:** Every activation has a corresponding backward op
(e.g. `ReluGradOp`, `GeluGradOp`).

## Plan-Then-Dispatch Pattern

For activations with parameters (e.g. `HardtanhOp { min_val, max_val }`),
Hephaestus validates the parameters once in a `ParameterizedUnaryOps::plan`
call, then dispatches on the hot path with the pre-validated plan:

```rust,ignore
let plan = device.plan_hardtanh(-6.0, 6.0)?;
for batch in batches {
    device.elementwise_unary_planned(&plan, &input, &mut output)?;
}
```

## GELU / LGAMMA / Error Function Parity

GELU, LGAMMA, and the error function are verified to produce parity results
across WGPU, CUDA, ROCm, and Metal backends (closed by PR #228 and #231).
