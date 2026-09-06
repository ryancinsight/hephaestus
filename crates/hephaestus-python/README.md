# hephaestus-python

GPU array compute for Python over
[Hephaestus](https://github.com/ryancinsight/hephaestus), the Atlas
accelerator substrate: typed device buffers and kernel dispatch behind one
device layer that runs on WGPU today and CUDA where it is built for.

## Install

```sh
pip install hephaestus-python
```

The distribution is `hephaestus-python`; the module you import is
`pyhephaestus`. Wheels are built for CPython 3.9 through 3.13 on Linux,
Windows, and macOS.

## Use

```python
import numpy as np
import pyhephaestus as hp

device = hp.Device()          # picks an available backend
print(device.backend_name)    # e.g. "wgpu"

a = hp.Array.from_numpy(np.arange(8, dtype=np.float32), device)

c = hp.add(a, a)                    # runs on the device
total = hp.sum(c)                   # still an Array, still on the device
print(total.to_numpy()[0])          # 56.0
print(c.to_numpy())                 # back to NumPy, host side
```

Arrays are one-dimensional, `float32`, and C-contiguous. Data crosses the
host/device boundary only where you ask it to — `from_numpy` uploads,
`to_numpy` downloads — and a reduction returns an `Array` rather than a Python
float, so a chain of operations stays resident on the device instead of
round-tripping through the host at every step.

## What is exposed

- **`Device`** — backend selection and identification.
- **`Array`** — device-resident `float32` buffers, `zeros`, `from_numpy`,
  `to_numpy`.
- **`SparseMatrix`** — sparse device matrices in compressed-row form.
- **Elementwise** — `add`, `sub`, `mul`, `div`, `pow`, `exp`, `log`, `sin`,
  `cos`, `sqrt`, `abs`, `neg`, also available as operators on `Array`.
- **Reductions** — `sum`, `min`, `max`, `mean`, `norm_l1`, `norm_l2`.
- **Linear algebra** — `matmul`, `dot`, `trace`, decompositions, matrix
  functions, spectral transforms, scans, and random generation.

## About CUDA

Published wheels enable the portable WGPU backend. The CUDA entry points are
present in those wheels and return a typed backend-unavailable error rather
than falling back silently — a fallback that quietly ran somewhere other than
where you asked would be the harder bug. For the native CUDA backend, build
from source on a CUDA 13.2+ host with `--features cuda`.

## Why the kernels are not here

This package is a binding surface: it converts types, maps Rust errors onto
Python exceptions, and releases the GIL around compute. Every kernel lives in
the Rust crates it wraps, so there is one implementation and one place to
verify it.

## Links

- [Source and issues](https://github.com/ryancinsight/hephaestus)

## Licence

MIT or Apache-2.0, at your option.
