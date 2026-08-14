# hephaestus-host

Host reference device for the Hephaestus seams (ADR 0046).

`HostDevice` implements `hephaestus_core::ComputeDevice` over plain host memory
so CPU reference implementors — leto adapters first — join the same role traits
the GPU backends implement, and the conformance suite can instantiate a CPU pair
for every clause.

This crate is the **reference substrate**: correctness and conformance first,
never a performance path. Consumers wanting fast CPU execution use `leto`
directly. Its value is that it gives every clause in `hephaestus-conformance` a
device that is always available — no adapter, no toolkit, no hardware — so a
contract can be asserted rather than skipped.

## Documentation

- API reference: [docs.rs/hephaestus-host](https://docs.rs/hephaestus-host)
- Workspace overview: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
