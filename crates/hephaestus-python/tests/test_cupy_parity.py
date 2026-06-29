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


def test_batched_matmul_matches_cupy() -> None:
    # [batch, m, k] @ [batch, k, n] -> [batch, m, n].
    batch, m, k, n = 4, 3, 5, 2
    a = (np.arange(batch * m * k, dtype=np.float32).reshape(batch, m, k) * 0.1) - 1.0
    b = (np.arange(batch * k * n, dtype=np.float32).reshape(batch, k, n) * 0.05) - 0.5
    got = np.asarray(_arr(a).batched_matmul(_arr(b)).to_numpy()).reshape(batch, m, n)
    expected = cp.asarray(a) @ cp.asarray(b)
    _close_arr("batched_matmul", got, expected, atol=1e-3)


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


def test_matrix_rank_matches_cupy() -> None:
    # Full-rank and rank-deficient matrices.
    rank_def = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)  # rank 1
    assert _arr(_A).matrix_rank() == int(cp.linalg.matrix_rank(cp.asarray(_A)))
    assert _arr(rank_def).matrix_rank() == int(
        cp.linalg.matrix_rank(cp.asarray(rank_def))
    )


def test_pinv_matches_cupy() -> None:
    got = _to2d(_arr(_A).pinv(), (3, 3))
    expected = cp.linalg.pinv(cp.asarray(_A))
    _close_arr("pinv", got, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Axis reductions and cumulative sum
# ---------------------------------------------------------------------------


def _vec(pa):
    return np.asarray(pa.to_numpy()).flatten()


def test_axis_reductions_match_cupy() -> None:
    cp_a = cp.asarray(_A)
    for axis in (0, 1):
        _close_arr(f"sum_axis{axis}", _vec(_arr(_A).sum_axis(axis)), cp.sum(cp_a, axis=axis), atol=1e-3)
        _close_arr(f"mean_axis{axis}", _vec(_arr(_A).mean_axis(axis)), cp.mean(cp_a, axis=axis), atol=1e-4)
        _close_arr(f"min_axis{axis}", _vec(_arr(_A).min_axis(axis)), cp.min(cp_a, axis=axis), atol=1e-5)
        _close_arr(f"max_axis{axis}", _vec(_arr(_A).max_axis(axis)), cp.max(cp_a, axis=axis), atol=1e-5)


def test_cumsum_matches_cupy() -> None:
    cp_a = cp.asarray(_A)
    for axis in (0, 1):
        _close_arr(
            f"cumsum_axis{axis}",
            _to2d(_arr(_A).cumsum(axis), (3, 3)),
            cp.cumsum(cp_a, axis=axis),
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# Sparse (CSR) ops — Hephaestus GPU vs scipy.sparse reference
# ---------------------------------------------------------------------------

# Reference is scipy.sparse on CPU (mathematically identical to the GPU result);
# cupy's sparse module is not required so the rest of the suite still runs when
# only dense cupy is present.
_sp = pytest.importorskip("scipy.sparse")

# A small matrix with structural zeros (6 nonzeros).
_SP_DENSE = np.array(
    [[2.0, 0.0, 1.0, 0.0], [0.0, 3.0, 0.0, 0.0], [4.0, 0.0, 0.0, 5.0], [0.0, 0.0, 6.0, 0.0]],
    dtype=np.float32,
)


def _csr():
    return hp.SparseMatrix.from_dense(_arr(_SP_DENSE))


def test_sparse_roundtrip_and_nnz_match_scipy() -> None:
    csr = _csr()
    nnz = csr.nnz() if callable(getattr(csr, "nnz", None)) else csr.nnz
    assert nnz == _sp.csr_matrix(_SP_DENSE).nnz
    _close_arr("csr_to_dense", _to2d(csr.to_dense(), (4, 4)), cp.asarray(_SP_DENSE), atol=1e-5)


def test_spmv_matches_scipy() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    got = np.asarray(hp.spmv(_csr(), hp.Array.from_numpy(x, _DEVICE)).to_numpy())
    expected = _sp.csr_matrix(_SP_DENSE) @ x
    _close_arr("spmv", got, cp.asarray(expected), atol=1e-4)


def test_spmm_matches_scipy() -> None:
    b = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    got = _to2d(hp.spmm(_csr(), _arr(b)), (4, 2))
    expected = _sp.csr_matrix(_SP_DENSE) @ b
    _close_arr("spmm", got, cp.asarray(expected), atol=1e-4)


# ---------------------------------------------------------------------------
# LU decomposition and eigenvalues (verified by invariants vs numpy)
# ---------------------------------------------------------------------------

# General nonsingular matrix for LU.
_LU_M = np.array([[4.0, 3.0, 2.0], [1.0, 5.0, 1.0], [2.0, 1.0, 6.0]], dtype=np.float32)
# Symmetric matrix (real spectrum) for the complex-eigenvalue path.
_SYM_M = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]], dtype=np.float32)


def test_lu_matches_numpy() -> None:
    # leto-style LU: (L, U, perm) with P-permuted A == L @ U, L unit-lower, U upper.
    l, u, perm = hp.lu(_arr(_LU_M))
    l_np = _to2d(l, (3, 3))
    u_np = _to2d(u, (3, 3))
    p = np.asarray(perm).astype(int)
    _close_arr("lu_reconstruct", l_np @ u_np, cp.asarray(_LU_M[p]), atol=1e-4)
    _close_arr("lu_l_unit_lower", np.triu(l_np, 1), cp.zeros((3, 3)), atol=1e-5)
    _close_arr("lu_u_upper", np.tril(u_np, -1), cp.zeros((3, 3)), atol=1e-5)


def test_hessenberg_matches_numpy() -> None:
    # hessenberg() -> (Q, H), A = Q H Qᵀ, H upper-Hessenberg, Q orthonormal.
    # Hessenberg form is not unique, so verify invariants, not raw factors.
    q, h = hp.hessenberg(_arr(_SYM_M))
    q_np = _to2d(q, (3, 3))
    h_np = _to2d(h, (3, 3))
    _close_arr("hessenberg_reconstruct", q_np @ h_np @ q_np.T, cp.asarray(_SYM_M), atol=1e-3)
    _close_arr("hessenberg_q_orthonormal", q_np.T @ q_np, cp.asarray(np.eye(3, dtype=np.float32)), atol=1e-4)
    assert np.allclose(np.tril(h_np, -2), 0.0, atol=1e-4), "H must be upper-Hessenberg"


def test_eigenvalues_match_numpy() -> None:
    # eigenvalues() returns Complex32; for a symmetric input the spectrum is real.
    ev = np.asarray(hp.eigenvalues(_arr(_SYM_M)))
    assert np.max(np.abs(ev.imag)) < 1e-3, f"symmetric eigenvalues should be real, got {ev.imag}"
    expected = np.linalg.eigvalsh(_SYM_M.astype(np.float64))
    _close_arr(
        "eigenvalues", np.sort(ev.real.astype(np.float64)), cp.asarray(np.sort(expected)), atol=1e-3
    )


# ---------------------------------------------------------------------------
# Seeded RNG — determinism + distribution (no value oracle; different RNG)
# ---------------------------------------------------------------------------


def test_normal_with_seed_deterministic_and_distributed() -> None:
    a = np.asarray(hp.normal_with_seed([4096], 0.0, 1.0, 42, _DEVICE).to_numpy())
    b = np.asarray(hp.normal_with_seed([4096], 0.0, 1.0, 42, _DEVICE).to_numpy())
    c = np.asarray(hp.normal_with_seed([4096], 0.0, 1.0, 7, _DEVICE).to_numpy())
    assert np.array_equal(a, b), "same seed must reproduce identical samples"
    assert not np.array_equal(a, c), "different seed must differ"
    assert abs(float(a.mean())) < 0.05 and abs(float(a.std()) - 1.0) < 0.06


def test_uniform_with_seed_deterministic_and_bounded() -> None:
    u = np.asarray(hp.uniform_with_seed([4096], -2.0, 3.0, 42, _DEVICE).to_numpy())
    u2 = np.asarray(hp.uniform_with_seed([4096], -2.0, 3.0, 42, _DEVICE).to_numpy())
    assert np.array_equal(u, u2), "same seed must reproduce identical samples"
    assert float(u.min()) >= -2.0 and float(u.max()) <= 3.0, "samples must lie in [low, high]"
    assert abs(float(u.mean()) - 0.5) < 0.06
