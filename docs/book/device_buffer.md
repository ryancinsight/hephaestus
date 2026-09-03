# Device Buffer

`DeviceBuffer<T>` is the typed storage handle used by every Hephaestus
operation seam. Its contract is deliberately small:

- `len` returns the logical element count;
- `tier` reports the `themis::MemoryTier` classification;
- `is_empty` derives the zero-length check from `len`.

The concrete buffer type is selected by `ComputeDevice::Buffer<T>`. The
associated type carries the backend and scalar type through the compiler, so
the core traits do not need `dyn DeviceBuffer` or a vendor-specific pointer.
`T` must implement `eunomia::Pod` at the device boundary.

## Host-to-device lifecycle

The lifecycle is allocation or upload, optional writes or device-local copies,
then download. `write_buffer` requires an equal-length slice;
`write_sub_buffer` checks the offset and range before copying;
`copy_buffer` requires equal-length buffers and stays on the selected device.
A transfer failure is returned, not silently converted into a default buffer.

`StridedView` supplies rank-const-generic views over a `D::Buffer<T>` for
operations that accept non-contiguous layouts. View validation belongs to the
operation seam; the buffer itself remains a logical contiguous allocation.

`DynamicStridedView` supplies the corresponding runtime-rank boundary for
expression fusion. It borrows a `D::Buffer<T>` and Leto's `LayoutDyn`; the
provider validates broadcast compatibility and writable-layout injectivity
before generating a kernel. Neither view materializes a contiguous copy.

The host reference example exercises these value contracts, including a
length-mismatch case for a short host output slice:

```rust
{{#include ../../crates/hephaestus-host/examples/book_host_device.rs}}
```

This example is compiled as part of the `hephaestus-host` package and is also
the source for the checked book example.
