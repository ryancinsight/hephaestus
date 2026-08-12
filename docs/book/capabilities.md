# Device Capabilities

Hephaestus exposes device capabilities through a structured query surface
so that callers can probe hardware features before dispatch.

## `DeviceFeature`

`DeviceFeature` is an enum of optional hardware capabilities:

| Feature | Description |
|---------|-------------|
| `F16` | Half-precision (fp16) arithmetic |
| `Bf16` | BFloat16 arithmetic |
| `F64` | Double-precision arithmetic |
| `SubgroupOps` | Subgroup / warp-level operations |
| `IndirectDispatch` | GPU-driven indirect dispatch |
| `AtomicFloat` | Atomic floating-point operations |
| `CooperativeMatrix` | Tensor core / matrix acceleration unit |

## `DeviceLimits`

`DeviceLimits` carries hardware constants read from the device API:

| Field | Description |
|-------|-------------|
| `max_buffer_bytes` | Maximum allocatable buffer |
| `max_workgroup_size` | Maximum threads per workgroup |
| `max_compute_units` | Streaming multiprocessors / compute units |
| `subgroup_size` | Warp / wavefront / subgroup width |

## Probing Capabilities

```rust,ignore
let caps = acq.capabilities;
if caps.has_feature(DeviceFeature::CooperativeMatrix) {
    // use tensor-core path
} else {
    // use general GEMM path
}
```

## Themis Integration

Hephaestus populates `GpuDeviceProperties` from the acquired device's
capabilities and passes the snapshot to Themis:

```rust,ignore
let props = GpuDeviceProperties {
    compute_units:         Some(NonZeroU32::new(128).unwrap()),
    warp_width:            Some(NonZeroU32::new(32).unwrap()),
    max_threads_per_unit:  Some(NonZeroU32::new(2048).unwrap()),
    registers_per_unit:    Some(NonZeroU32::new(65536).unwrap()),
    shared_mem_per_unit_bytes: Some(NonZeroUsize::new(98304).unwrap()),
    l2_bytes:              Some(NonZeroUsize::new(4_194_304).unwrap()),
    memory_tier:           MemoryTier::Gddr,
    memory_bytes:          Some(NonZeroU64::new(8_589_934_592).unwrap()),
};
let gpu_topo = GpuTopology::from_provider(props);
```

Moirai reads `GpuTopology` to decide whether to route a tensor operation
to a GPU worker.
