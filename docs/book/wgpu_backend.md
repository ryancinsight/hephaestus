# WGPU Backend

`hephaestus-wgpu` implements the core device and operation seams over
`wgpu`. The backend owns the adapter, device, queue, pipeline cache, and
transfer staging resources; consumers see `WgpuDevice` and the
backend-neutral traits.

## Acquisition

`WgpuDevice` implements `ComputeDeviceAcquisition`. Callers select
`DevicePreference`, optional `DeviceFeature` values, and required
`DeviceLimits` through `try_acquire_device` or `try_acquire_devices`.
Acquisition returns a typed error when probing, feature selection, or requested
limits fail. There is no implicit WGPU-to-host downgrade in this boundary.

`default_device_limits` provides the WGPU backend's mapped default request.
The actual device snapshot is available through `device_limits`,
`supports_device_feature`, and `topology`.

## Kernel dialect and operation seams

WGPU operation implementations use the core `Wgsl` `KernelDialect` marker
and WGSL shader sources. The core `ElementwiseOps`,
`AxisReductionOps`, `FullReductionOps`, and `DecompositionOps` contracts
remain the authoritative shape and error boundary. Prepared operation values
retain backend resources and can be dispatched again when the operand contract
permits it.

The WGPU backend requires a real adapter for execution. The host reference
backend is the deterministic local oracle for transfer and generic contract
tests; it is not a silent runtime fallback for a failed WGPU acquisition.
