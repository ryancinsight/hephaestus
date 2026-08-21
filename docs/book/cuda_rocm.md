# CUDA, ROCm, and Metal Backends

The accelerator workspace keeps vendor implementations behind the same core
seams. The vendor crates expose concrete device handles for backend-specific
work and implement `ComputeDevice`, `ComputeDeviceCapabilities`, and the
operation traits they support.

## CUDA

`hephaestus-cuda::CudaDevice::try_with_ordinal` acquires a CUDA device by
ordinal. `CudaDevice` also implements the generic
`ComputeDeviceAcquisition` contract. The CUDA provider owns its driver
context, typed buffers, kernel compilation, and synchronization; callers do
not pass a raw driver context through `hephaestus-core`.

## ROCm

`hephaestus-rocm::RocmDevice::try_with_ordinal` is the explicit ordinal
constructor for the HIP runtime. `RocmDevice` implements the same core device
and capability contracts and reports typed acquisition or transfer failures.

## Metal

`hephaestus-metal` is retired by ADR 0047. It contains no native Metal API;
`MetalDevice` forwards to `WgpuDevice`, and Metal is selected as a WGPU
adapter preference. New consumers use `hephaestus-wgpu` directly, for example
through its `try_metal` acquisition family. A selected Metal adapter still
executes through the WGPU provider; this is not a separate
`hephaestus-metal` command-encoding implementation.

These backends require their respective runtime/toolchain support. The
`hephaestus-host` reference device is used for deterministic tests on
machines without CUDA, ROCm, Metal, or a WGPU adapter. A missing vendor
runtime is an environment result, not evidence that another backend ran.
