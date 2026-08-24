# Device Capabilities

`ComputeDeviceCapabilities` keeps capability queries in the same backend
neutral vocabulary as [`ComputeDevice`](compute_device.md).

## Optional features

`DeviceFeature` currently contains these values:

| Value | Meaning |
|---|---|
| `TimestampQuery` | GPU timestamp queries |
| `ShaderF64` | 64-bit floating-point shader arithmetic |
| `ShaderF16` | 16-bit floating-point shader arithmetic |
| `MappablePrimaryBuffers` | host-mappable primary buffers |
| `ImmediateData` | immediate shader-data support |

`supports_device_feature` reports the features enabled on the acquired
device. It does not report every feature that another adapter might support.

## Limits

`device_limits` returns a `DeviceLimits` value with the common compute
limits: `max_buffer_size`, the three
`max_compute_workgroup_size_*` fields,
`max_compute_invocations_per_workgroup`,
`max_compute_workgroup_storage_size`, the optional
`max_storage_buffers_per_shader_stage`, the optional
`max_buffers_and_acceleration_structures_per_shader_stage`, and
`max_immediate_size`. Optional fields stay `None` when a backend has no
corresponding limit; a backend does not fabricate a value from a different
API.

## Topology

`ComputeDevice::topology` returns an optional `themis::GpuTopology`
snapshot. The host reference device returns `None` because it is not an
accelerator. GPU backends return a snapshot when their initialization probe
captured one. The topology is separate from the capability limits: a zero or
absent topology field means that the backend did not report that capacity.

The deterministic host query and its value assertions are maintained in the
[Capabilities example](examples/capabilities.md).
