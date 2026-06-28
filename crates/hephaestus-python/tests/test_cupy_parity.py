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


# ---------------------------------------------------------------------------
# Elementwise binary operators
# ---------------------------------------------------------------------------

# Two same-shape operands; _Q is held strictly positive so the same fixtures
# serve the log/sqrt unary tests below without domain issues.
_P = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
_Q = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float32)


def _close_arr(label, got, expected, atol):
    got_a = np.asarray(got)
    exp_a = cp.asnumpy(expected)
    assert got_a.shape == exp_a.shape, f"{label}: shape {got_a.shape} != {exp_a.shape}"
    diff = float(np.max(np.abs(got_a - exp_a))) if got_a.size else 0.0
    assert diff <= atol, f"{label}: max|diff|={diff:.3e} > atol={atol:.3e}"


def test_elementwise_operators_match_cupy() -> None:
    hp_p, hp_q = _arr(_P), _arr(_Q)
    cp_p, cp_q = cp.asarray(_P), cp.asarray(_Q)
    _close_arr("add", _to2d(hp_p + hp_q, (2, 3)), cp_p + cp_q, atol=1e-4)
    _close_arr("sub", _to2d(hp_p - hp_q, (2, 3)), cp_p - cp_q, atol=1e-4)
    _close_arr("mul", _to2d(hp_p * hp_q, (2, 3)), cp_p * cp_q, atol=1e-4)
    _close_arr("div", _to2d(hp_p / hp_q, (2, 3)), cp_p / cp_q, atol=1e-4)


# ---------------------------------------------------------------------------
# Unary elementwise math
# ---------------------------------------------------------------------------


def test_unary_math_matches_cupy() -> None:
    hp_p = _arr(_P)
    cp_p = cp.asarray(_P)
    _close_arr("abs", _to2d(_arr(-_P).abs(), (2, 3)), cp.abs(-cp_p), atol=1e-4)
    _close_arr("neg", _to2d(hp_p.neg(), (2, 3)), -cp_p, atol=1e-4)
    _close_arr("exp", _to2d(hp_p.exp(), (2, 3)), cp.exp(cp_p), atol=1e-3)
    _close_arr("log", _to2d(hp_p.log(), (2, 3)), cp.log(cp_p), atol=1e-4)
    _close_arr("sqrt", _to2d(hp_p.sqrt(), (2, 3)), cp.sqrt(cp_p), atol=1e-4)
    _close_arr("sin", _to2d(hp_p.sin(), (2, 3)), cp.sin(cp_p), atol=1e-4)
    _close_arr("cos", _to2d(hp_p.cos(), (2, 3)), cp.cos(cp_p), atol=1e-4)


# ---------------------------------------------------------------------------
# Reductions (whole-array)
# ---------------------------------------------------------------------------


def test_sum_matches_cupy() -> None:
    _close("sum", _scalar(_arr(_A).sum()), float(cp.sum(cp.asarray(_A))), atol=1e-3)


def test_mean_matches_cupy() -> None:
    _close("mean", _scalar(_arr(_A).mean()), float(cp.mean(cp.asarray(_A))), atol=1e-4)


def test_min_matches_cupy() -> None:
    _close("min", _scalar(_arr(_A).min()), float(cp.min(cp.asarray(_A))), atol=1e-5)


def test_max_matches_cupy() -> None:
    _close("max", _scalar(_arr(_A).max()), float(cp.max(cp.asarray(_A))), atol=1e-5)


# ---------------------------------------------------------------------------
# Linear-algebra extras: det / matpow / kron / matexp
# ---------------------------------------------------------------------------

# Well-conditioned, modest-magnitude square matrix so matexp/matpow stay in a
# numerically comfortable f32 range.
_LIN = np.array([[3.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 3.0]], dtype=np.float32)
_LIN_B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_det_matches_cupy() -> None:
    got = _scalar(_arr(_LIN).det())
    expected = float(cp.linalg.det(cp.asarray(_LIN)))
    _close("det", got, expected, atol=1e-3)


def test_matpow_matches_cupy() -> None:
    got = _to2d(_arr(_LIN).matpow(3), (3, 3))
    expected = cp.asnumpy(cp.linalg.matrix_power(cp.asarray(_LIN), 3))
    _close_arr("matpow", got, cp.asarray(expected), atol=1e-2)


def test_kron_matches_cupy() -> None:
    got = _to2d(_arr(_LIN).kron(_arr(_LIN_B)), (6, 6))
    expected = cp.kron(cp.asarray(_LIN), cp.asarray(_LIN_B))
    _close_arr("kron", got, expected, atol=1e-4)


def test_matexp_matches_cupy() -> None:
    cupyx_linalg = pytest.importorskip("cupyx.scipy.linalg")
    got = _to2d(_arr(_LIN).matexp(), (3, 3))
    expected = cupyx_linalg.expm(cp.asarray(_LIN))
    _close_arr("matexp", got, expected, atol=1e-3)
