# WGPU Backend

`hephaestus-wgpu` is the cross-platform GPU backend using wgpu.
It supports Vulkan, Metal, DirectX 12, and WebGPU in a single codebase.

## Device Acquisition

```rust,ignore
use hephaestus_wgpu::WgpuDevice;

let device = WgpuDevice::acquire(DevicePreference::Discrete)?;
```

## Kernel Language

wgpu kernels are written in WGSL (WebGPU Shading Language). The
`KernelDialect::Wgsl` marker type selects the WGSL compile path.

## Shared Validation Helpers

`WgpuTransformBackend` exposes three canonical validation helpers
(delivered in Apollo PR #shared-validation):

| Helper | Description |
|--------|-------------|
| `validate_plan(plan)` | Rejects empty or invalid plans |
| `require_len(buf, len)` | Asserts buffer length equals required length |
| `validate_storage_profile(profile)` | Validates the storage tier and precision profile |

These helpers are consumed by `apollo-fft` and `apollo-gft` to avoid
duplicated validation logic.

## GPU Transform Backend

`WgpuTransformBackend` is also the base for Apollo's GPU FFT dispatch:

```rust,ignore
use apollo_fft::WgpuTransformBackend;

let backend = WgpuTransformBackend::new(&device)?;
let plan = backend.plan_1d(4096, PrecisionProfile::default())?;
backend.execute(&plan, &mut input_buf, &mut output_buf)?;
```

## Plan Cache

`WgpuTransformPlan` is cached by `GpuTransformPlanner` to avoid
re-compiling WGSL shader pipelines on every call. The planner key is
`(shape, precision_profile, direction)`.
