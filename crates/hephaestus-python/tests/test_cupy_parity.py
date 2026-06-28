"""CuPy (GPU) output-parity tests for the ``pyhephaestus`` bindings.

Each test verifies that ``pyhephaestus`` (Hephaestus GPU compute, wgpu backend)
produces numerically equivalent results to the reference CuPy (CUDA) operation
on identical f32 inputs — GPU-vs-GPU array-op parity. The whole module skips
automatically when ``pyhephaestus`` or ``cupy`` is unavailable, or when no CUDA
device is present.

Hephaestus norm conventions are entrywise (flatten then vector norm):
- ``norm_l1`` = sum(|a|), ``norm_l2`` = Frobenius = sqrt(sum(a^2)),
  ``norm_max`` = max(|a|).

Run via::

    pytest crates/hephaestus-python/tests/test_cupy_parity.py -v
"""

import numpy as np
import pytest

hp = pytest.importorskip("pyhephaestus")
cp = pytest.importorskip("cupy")

# Skip the whole module if there is no usable CUDA device for the CuPy reference.
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device for the CuPy reference", allow_module_level=True)
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"CuPy CUDA runtime unavailable: {exc}", allow_module_level=True)

_DEVICE = hp.Device()

# Deterministic f32 fixtures.
_A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], dtype=np.float32)
_B = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
_V = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
_W = np.array([2.0, 0.0, 1.0, 3.0], dtype=np.float32)


def _arr(m):
    """Upload a numpy array to a Hephaestus device array, preserving 2-D shape."""
    flat = hp.Array.from_numpy(m.flatten(), _DEVICE)
    return flat.reshape(list(m.shape)) if m.ndim > 1 else flat


def _to2d(pa, shape):
    return np.asarray(pa.to_numpy()).reshape(shape)


def _scalar(pa):
    return float(np.asarray(pa.to_numpy()).flatten()[0])


def _close(label, got, expected, atol):
    diff = abs(float(got) - float(expected))
    assert diff <= atol, f"{label}: |{got} - {expected}| = {diff:.3e} > atol={atol:.3e}"


# ---------------------------------------------------------------------------
# Matrix / vector products
# ---------------------------------------------------------------------------


def test_matmul_matches_cupy() -> None:
    got = _to2d(_arr(_A).matmul(_arr(_B)), (3, 3))
    expected = cp.asnumpy(cp.asarray(_A) @ cp.asarray(_B))
    assert np.allclose(got, expected, atol=1e-4), f"matmul: max|diff|={np.max(np.abs(got - expected)):.3e}"


def test_dot_matches_cupy() -> None:
    got = _scalar(_arr(_V).dot(_arr(_W)))
    expected = float(cp.dot(cp.asarray(_V), cp.asarray(_W)))
    _close("dot", got, expected, atol=1e-4)


def test_trace_matches_cupy() -> None:
    got = _scalar(_arr(_A).trace())
    expected = float(cp.trace(cp.asarray(_A)))
    _close("trace", got, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Entrywise norms
# ---------------------------------------------------------------------------


def test_norm_l1_matches_cupy() -> None:
    got = _scalar(_arr(_A).norm_l1())
    expected = float(cp.sum(cp.abs(cp.asarray(_A))))
    _close("norm_l1", got, expected, atol=1e-3)


def test_norm_l2_matches_cupy() -> None:
    got = _scalar(_arr(_A).norm_l2())
    expected = float(cp.linalg.norm(cp.asarray(_A)))
    _close("norm_l2", got, expected, atol=1e-3)


def test_norm_max_matches_cupy() -> None:
    got = _scalar(_arr(_A).norm_max())
    expected = float(cp.max(cp.abs(cp.asarray(_A))))
    _close("norm_max", got, expected, atol=1e-4)
