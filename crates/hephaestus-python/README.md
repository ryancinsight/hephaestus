# pyhephaestus

Python bindings for [Hephaestus](https://github.com/ryancinsight/hephaestus),
the Atlas GPU/accelerator substrate: typed device buffers and compute dispatch
over a shared device layer.

This package is a thin PyO3/NumPy boundary over the Rust device APIs. It
converts types, maps Rust errors to Python exceptions, and releases the GIL
around compute; the domain logic lives in the Rust crates.

## Install

```sh
pip install pyhephaestus
```

```python
import pyhephaestus
```

Published wheels are built for CPython 3.9–3.13 on Linux, Windows, and macOS,
and enable the portable WGPU backend. CUDA entry points are present in those
wheels and return the typed backend-unavailable error; to get the native CUDA
backend, build from source on a CUDA 13.2+ host with the `cuda` feature.

## Documentation

- Substrate overview, layer boundaries, and backend contracts: the
  [repository README](https://github.com/ryancinsight/hephaestus#readme)

## License

MIT OR Apache-2.0
