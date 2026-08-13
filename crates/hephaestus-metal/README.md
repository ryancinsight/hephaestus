# hephaestus-metal

**Retired by Accepted [ADR 0047]; use `hephaestus-wgpu` directly.**

This crate contains no Metal device API. There is no `metal::`, no `objc`, no
`MTLDevice`, and no MSL shader anywhere in it. `MetalDevice` is a newtype over
`WgpuDevice` acquired through `WgpuDevice::try_metal`, and every operation
module forwards to `hephaestus-wgpu`. Metal is an adapter preference of the wgpu
backend, not a backend of its own — which is why `hephaestus-wgpu` is where the
Metal selection actually lives, in a family of constructors (`try_metal` and
its device-preference and adapter-enumeration variants) that filter adapters on
`wgpu::Backend::Metal`.

No capability is lost by depending on `hephaestus-wgpu` instead: every operation
this crate exposes is a forward to it. The migration is a one-line substitution.

```rust,ignore
// before
let device = MetalDevice::try_default()?;

// after
let device = WgpuDevice::try_metal("my-device")?;
```

A caller that genuinely needs the adapter vendor reads it from the device:

```rust,ignore
device.adapter_info().map(|info| info.backend) == Some(wgpu::Backend::Metal)
```

Selecting Metal this way still executes through the native Apple Metal path; it
does not silently fall back to CPU, Vulkan, or another wgpu adapter.

Should a native Metal backend ever be written against `metal-rs`/`objc`, it
earns a crate the same way CUDA and ROCm do — by owning a device API. ADR 0047
does not foreclose that; it declines to reserve a crate-shaped placeholder for
it.

[ADR 0047]: https://github.com/ryancinsight/hephaestus/blob/main/docs/adr/0047-metal-as-a-wgpu-adapter-preference.md

## Documentation

- API reference: [docs.rs/hephaestus-metal](https://docs.rs/hephaestus-metal)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
