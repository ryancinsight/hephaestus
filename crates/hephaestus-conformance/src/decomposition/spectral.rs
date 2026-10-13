//! Spectral clauses: symmetric eigen, SVD, general reductions, and Leto differentials.

use hephaestus_core::{ComputeDevice, DecompositionOps, StridedView};
use leto::Layout;

use hephaestus_core::{
    BidiagonalHandle, HessenbergHandle, SchurHandle, SvdHandle, SymmetricEigenHandle,
};

use super::common::DECOMP_BOUND;

/// Symmetric Jacobi eigendecomposition: the fixture `[[2,1],[1,2]]` has
/// the exact spectrum `{1, 3}`; the clause asserts the eigenvalue
/// multiset, the `f64` residual `‖A·V − V·Λ‖` and orthogonality
/// `‖VᵀV − I‖` under the column-eigenvector convention, and agreement of
/// the values-only path.
pub(crate) fn symmetric_eigen_diagonalizes_exactly<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [2.0f32, 1.0, 1.0, 2.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");
    let eigen = ops
        .symmetric_eigen(device, StridedView::new(&a, &square))
        .expect("symmetric eigendecomposition");
    assert_eq!(eigen.order(), 2, "{name}: eigen order");

    let mut values = [0.0f32; 2];
    device
        .download(eigen.eigenvalues(), &mut values)
        .expect("eigenvalue download");
    let mut sorted = values;
    sorted.sort_by(f32::total_cmp);
    assert!(
        (sorted[0] - 1.0).abs() < DECOMP_BOUND && (sorted[1] - 3.0).abs() < DECOMP_BOUND,
        "{name}: spectrum must be {{1, 3}}, got {values:?}"
    );

    let mut vectors = [0.0f32; 4];
    device
        .download(eigen.eigenvectors(), &mut vectors)
        .expect("eigenvector download");

    // Residual ‖A·V − V·Λ‖ and orthogonality ‖VᵀV − I‖ in f64.
    let mut max_residual = 0.0f64;
    let mut max_ortho = 0.0f64;
    for row in 0..2 {
        for col in 0..2 {
            let mut av = 0.0f64;
            let mut vtv = 0.0f64;
            for k in 0..2 {
                av += f64::from(a_host[row * 2 + k]) * f64::from(vectors[k * 2 + col]);
                vtv += f64::from(vectors[k * 2 + row]) * f64::from(vectors[k * 2 + col]);
            }
            let vl = f64::from(vectors[row * 2 + col]) * f64::from(values[col]);
            max_residual = max_residual.max((av - vl).abs());
            let identity = if row == col { 1.0 } else { 0.0 };
            max_ortho = max_ortho.max((vtv - identity).abs());
        }
    }
    assert!(
        max_residual < f64::from(DECOMP_BOUND),
        "{name}: eigen residual ‖A·V − V·Λ‖ = {max_residual}"
    );
    assert!(
        max_ortho < f64::from(DECOMP_BOUND),
        "{name}: eigenvector orthogonality ‖VᵀV − I‖ = {max_ortho}"
    );

    // The values-only path agrees with the full path's multiset.
    let only = ops
        .symmetric_eigenvalues(device, StridedView::new(&a, &square))
        .expect("values-only path");
    let mut got_only = [0.0f32; 2];
    device
        .download(&only, &mut got_only)
        .expect("values-only download");
    got_only.sort_by(f32::total_cmp);
    for (index, (full, fast)) in sorted.iter().zip(&got_only).enumerate() {
        assert!(
            (full - fast).abs() < DECOMP_BOUND,
            "{name}: values-only eigenvalue {index}: {fast} vs full path {full}"
        );
    }
}

/// SVD: the diagonal fixture `[[3,0],[0,4]]` has exact singular values
/// `{4, 3}`; the rank-1 fixture `[[1,2],[2,4]]` has exact values
/// `{5, 0}`. The clause asserts both multisets, the `f64` reconstruction
/// `‖A − U·Σ·Vᵀ‖` and both orthogonalities under the column-vector
/// convention, and values-only/full-path agreement.
pub(crate) fn svd_recovers_exact_spectra<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");

    let a_host = [3.0f32, 0.0, 0.0, 4.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let svd = ops
        .svd(device, StridedView::new(&a, &square))
        .expect("diagonal SVD");
    assert_eq!(svd.shape(), (2, 2), "{name}: SVD shape");

    let mut sigma = [0.0f32; 2];
    device
        .download(svd.singular_values(), &mut sigma)
        .expect("singular value download");
    let mut sorted = sigma;
    sorted.sort_by(f32::total_cmp);
    assert!(
        (sorted[0] - 3.0).abs() < DECOMP_BOUND && (sorted[1] - 4.0).abs() < DECOMP_BOUND,
        "{name}: diagonal fixture spectrum must be {{3, 4}}, got {sigma:?}"
    );
    for value in &sigma {
        assert!(*value >= 0.0, "{name}: singular values are non-negative");
    }

    let mut u = [0.0f32; 4];
    let mut v = [0.0f32; 4];
    device.download(svd.u(), &mut u).expect("U download");
    device.download(svd.v(), &mut v).expect("V download");

    let mut max_reconstruction = 0.0f64;
    let mut max_u_ortho = 0.0f64;
    let mut max_v_ortho = 0.0f64;
    for row in 0..2 {
        for col in 0..2 {
            let mut acc = 0.0f64;
            let mut utu = 0.0f64;
            let mut vtv = 0.0f64;
            for k in 0..2 {
                acc += f64::from(u[row * 2 + k]) * f64::from(sigma[k]) * f64::from(v[col * 2 + k]);
                utu += f64::from(u[k * 2 + row]) * f64::from(u[k * 2 + col]);
                vtv += f64::from(v[k * 2 + row]) * f64::from(v[k * 2 + col]);
            }
            max_reconstruction =
                max_reconstruction.max((acc - f64::from(a_host[row * 2 + col])).abs());
            let identity = if row == col { 1.0 } else { 0.0 };
            max_u_ortho = max_u_ortho.max((utu - identity).abs());
            max_v_ortho = max_v_ortho.max((vtv - identity).abs());
        }
    }
    assert!(
        max_reconstruction < f64::from(DECOMP_BOUND),
        "{name}: SVD reconstruction ‖A − U·Σ·Vᵀ‖ = {max_reconstruction}"
    );
    assert!(
        max_u_ortho < f64::from(DECOMP_BOUND),
        "{name}: left singular vectors ‖UᵀU − I‖ = {max_u_ortho}"
    );
    assert!(
        max_v_ortho < f64::from(DECOMP_BOUND),
        "{name}: right singular vectors ‖VᵀV − I‖ = {max_v_ortho}"
    );

    // Rank-1 fixture: exact spectrum {5, 0}.
    let deficient = device
        .upload(&[1.0f32, 2.0, 2.0, 4.0])
        .expect("rank-1 upload");
    let only = ops
        .singular_values(device, StridedView::new(&deficient, &square))
        .expect("values-only path");
    let mut got = [0.0f32; 2];
    device.download(&only, &mut got).expect("values download");
    got.sort_by(f32::total_cmp);
    assert!(
        got[0].abs() < DECOMP_BOUND && (got[1] - 5.0).abs() < DECOMP_BOUND,
        "{name}: rank-1 fixture spectrum must be {{5, 0}}, got {got:?}"
    );
}

/// General spectral reductions: complex eigenvalues of the rotation
/// generator `[[0,-1],[1,0]]` are exactly `±i`; the Schur form of an
/// already-upper fixture stays upper with similarity preserved;
/// Hessenberg and bidiagonal reductions hold their structural zero
/// patterns, orthogonality, and `f64` reconstruction at the derived
/// bound.
pub(crate) fn general_spectral_reductions_hold_their_invariants<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let square2 = Layout::c_contiguous([2, 2]).expect("2x2 layout");

    // Complex eigenvalues: the rotation generator has spectrum ±i.
    let rotation = device
        .upload(&[0.0f32, -1.0, 1.0, 0.0])
        .expect("rotation upload");
    let values = ops
        .eigenvalues(device, StridedView::new(&rotation, &square2))
        .expect("complex eigenvalues");
    let mut got = [eunomia::Complex::new(0.0f32, 0.0); 2];
    device.download(&values, &mut got).expect("eigen download");
    let mut imag: Vec<f32> = got.iter().map(|value| value.im).collect();
    imag.sort_by(f32::total_cmp);
    for value in &got {
        assert!(
            value.re.abs() < DECOMP_BOUND,
            "{name}: rotation eigenvalues are purely imaginary, got {got:?}"
        );
    }
    assert!(
        (imag[0] + 1.0).abs() < DECOMP_BOUND && (imag[1] - 1.0).abs() < DECOMP_BOUND,
        "{name}: rotation spectrum must be ±i, got {got:?}"
    );

    // Schur: real-spectrum fixture [[2,1],[0,3]] reduces with similarity
    // preserved and T upper (distinct real eigenvalues force 1×1 blocks).
    let a_host = [2.0f32, 1.0, 0.0, 3.0];
    let a = device.upload(&a_host).expect("schur upload");
    let schur = ops
        .schur(device, StridedView::new(&a, &square2))
        .expect("schur reduction");
    assert_eq!(schur.order(), 2, "{name}: schur order");
    let mut q = [0.0f32; 4];
    let mut t = [0.0f32; 4];
    device
        .download(schur.q_buffer(), &mut q)
        .expect("Q download");
    device
        .download(schur.t_buffer(), &mut t)
        .expect("T download");
    assert!(
        t[2].abs() < DECOMP_BOUND,
        "{name}: real distinct spectrum must give upper T, got subdiagonal {}",
        t[2]
    );
    similarity_and_orthogonality(&a_host, &q, &t, 2, name, "schur");

    // Hessenberg: a 3×3 fixture zeroes below the first subdiagonal.
    let square3 = Layout::c_contiguous([3, 3]).expect("3x3 layout");
    let h_host = [4.0f32, 1.0, 2.0, 2.0, 5.0, 1.0, 1.0, 2.0, 6.0];
    let h_input = device.upload(&h_host).expect("hessenberg upload");
    let hess = ops
        .hessenberg(device, StridedView::new(&h_input, &square3))
        .expect("hessenberg reduction");
    assert_eq!(hess.order(), 3, "{name}: hessenberg order");
    let mut hq = [0.0f32; 9];
    let mut hh = [0.0f32; 9];
    device
        .download(hess.q_buffer(), &mut hq)
        .expect("Q download");
    device
        .download(hess.h_buffer(), &mut hh)
        .expect("H download");
    assert!(
        hh[6].abs() < DECOMP_BOUND,
        "{name}: H must vanish below the first subdiagonal, got {}",
        hh[6]
    );
    similarity_and_orthogonality(&h_host, &hq, &hh, 3, name, "hessenberg");

    // Bidiagonal: a 3×2 fixture reduces to main + superdiagonal only,
    // with two-sided orthogonality and reconstruction.
    let tall = Layout::c_contiguous([3, 2]).expect("3x2 layout");
    let b_host = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_input = device.upload(&b_host).expect("bidiagonal upload");
    let bidiagonal = ops
        .bidiagonalize(device, StridedView::new(&b_input, &tall))
        .expect("bidiagonal reduction");
    assert_eq!(bidiagonal.shape(), (3, 2), "{name}: bidiagonal shape");
    use hephaestus_core::DeviceBuffer;
    let b_len = bidiagonal.b_buffer().len();
    let mut b = vec![0.0f32; b_len];
    device
        .download(bidiagonal.b_buffer(), &mut b)
        .expect("B download");
    // B rows beyond n and entries outside {diagonal, superdiagonal}
    // vanish (row-major over the stored extent, columns = 2).
    for (index, value) in b.iter().enumerate() {
        let (row, col) = (index / 2, index % 2);
        if col != row && col != row + 1 {
            assert!(
                value.abs() < DECOMP_BOUND,
                "{name}: B entry ({row},{col}) = {value} must vanish"
            );
        }
    }
    let u_len = {
        let mut u_probe = 0;
        u_probe += bidiagonal.u_buffer().len();
        u_probe
    };
    let mut u = vec![0.0f32; u_len];
    device
        .download(bidiagonal.u_buffer(), &mut u)
        .expect("U download");
    let mut v = [0.0f32; 4];
    device
        .download(bidiagonal.v_buffer(), &mut v)
        .expect("V download");
    // Reconstruction A = U·B·Vᵀ in f64, with U columns = stored width.
    let u_cols = u_len / 3;
    let b_rows = b_len / 2;
    let mut max_err = 0.0f64;
    for row in 0..3 {
        for col in 0..2 {
            let mut acc = 0.0f64;
            for j in 0..u_cols.min(b_rows) {
                for k in 0..2 {
                    acc += f64::from(u[row * u_cols + j])
                        * f64::from(b[j * 2 + k])
                        * f64::from(v[col * 2 + k]);
                }
            }
            max_err = max_err.max((acc - f64::from(b_host[row * 2 + col])).abs());
        }
    }
    assert!(
        max_err < f64::from(DECOMP_BOUND),
        "{name}: bidiagonal reconstruction error {max_err}"
    );
}

/// Shared `f64` similarity (`‖A − Q·T·Qᵀ‖`) and orthogonality
/// (`‖QᵀQ − I‖`) checks for square reductions.
fn similarity_and_orthogonality(
    a: &[f32],
    q: &[f32],
    t: &[f32],
    n: usize,
    name: &str,
    label: &str,
) {
    let mut max_similarity = 0.0f64;
    let mut max_ortho = 0.0f64;
    for row in 0..n {
        for col in 0..n {
            let mut qtqt = 0.0f64;
            let mut qtq = 0.0f64;
            for j in 0..n {
                for k in 0..n {
                    qtqt += f64::from(q[row * n + j])
                        * f64::from(t[j * n + k])
                        * f64::from(q[col * n + k]);
                }
                qtq += f64::from(q[j * n + row]) * f64::from(q[j * n + col]);
            }
            max_similarity = max_similarity.max((qtqt - f64::from(a[row * n + col])).abs());
            let identity = if row == col { 1.0 } else { 0.0 };
            max_ortho = max_ortho.max((qtq - identity).abs());
        }
    }
    assert!(
        max_similarity < f64::from(DECOMP_BOUND),
        "{name}: {label} similarity ‖A − Q·T·Qᵀ‖ = {max_similarity}"
    );
    assert!(
        max_ortho < f64::from(DECOMP_BOUND),
        "{name}: {label} orthogonality ‖QᵀQ − I‖ = {max_ortho}"
    );
}

/// Symmetric eigendecomposition of a diagonally dominant 4×4 agrees with
/// leto on the sorted eigenvalue multiset (ordering is backend-owned),
/// and every returned eigenpair satisfies `‖A·v − λ·v‖∞` within the
/// derived bound, reconstructed in `f64`. The values-only entry point
/// must agree with the paired decomposition.
pub(crate) fn symmetric_eigen_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [
        4.0f32, 1.0, 0.5, 0.25, 1.0, 3.0, 0.25, 0.125, 0.5, 0.25, 2.0, 0.0625, 0.25, 0.125, 0.0625,
        1.5,
    ];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([4, 4]).expect("matrix layout");
    let decomp = ops
        .symmetric_eigen(device, StridedView::new(&a, &layout))
        .expect("device symmetric eigendecomposition");

    let host_matrix = leto::Array::from_shape_vec([4, 4], a_host.to_vec()).expect("host matrix");
    let host_decomp =
        leto_ops::symmetric_eigen_jacobi(&host_matrix.view()).expect("leto symmetric eigen");

    assert_eq!(decomp.order(), 4, "{name}: symmetric eigen order");

    let mut values = [0.0f32; 4];
    device
        .download(decomp.eigenvalues(), &mut values)
        .expect("eigenvalue download");
    let mut sorted_values = values;
    sorted_values.sort_by(f32::total_cmp);
    let mut host_values: [f32; 4] = host_decomp
        .eigenvalues
        .as_slice()
        .try_into()
        .expect("leto eigenvalue count");
    host_values.sort_by(f32::total_cmp);
    for (index, (device_value, host_value)) in
        sorted_values.iter().zip(host_values.iter()).enumerate()
    {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: sorted eigenvalue {index}: device {device_value} vs leto {host_value}"
        );
    }

    // Residual check pins the eigenvectors without assuming leto's
    // ordering or sign: for each returned pair, ‖A·v − λ·v‖∞ ≤ bound.
    let mut vectors = [0.0f32; 16];
    device
        .download(decomp.eigenvectors(), &mut vectors)
        .expect("eigenvector download");
    for pair in 0..4 {
        let lambda = f64::from(values[pair]);
        for row in 0..4 {
            let mut accumulated = 0.0f64;
            for col in 0..4 {
                accumulated +=
                    f64::from(a_host[row * 4 + col]) * f64::from(vectors[col * 4 + pair]);
            }
            let residual = (accumulated - lambda * f64::from(vectors[row * 4 + pair])).abs();
            assert!(
                residual < f64::from(DECOMP_BOUND),
                "{name}: eigenpair {pair} residual row {row}: {residual}"
            );
        }
    }

    let values_only = ops
        .symmetric_eigenvalues(device, StridedView::new(&a, &layout))
        .expect("device values-only eigenvalues");
    let mut got_values_only = [0.0f32; 4];
    device
        .download(&values_only, &mut got_values_only)
        .expect("values-only download");
    got_values_only.sort_by(f32::total_cmp);
    for (index, (device_value, host_value)) in
        got_values_only.iter().zip(host_values.iter()).enumerate()
    {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: values-only eigenvalue {index}: device {device_value} vs leto {host_value}"
        );
    }
}

/// General eigenvalues of `[[0,1],[-2,3]]` (spectrum `{1, 2}`, exactly
/// representable) agree with leto within the derived bound after sorting
/// by real part (ordering is backend-owned; the spectrum is real and
/// tie-free, so the sort is deterministic).
pub(crate) fn eigenvalues_match_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [0.0f32, 1.0, -2.0, 3.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([2, 2]).expect("matrix layout");
    let spectrum = ops
        .eigenvalues(device, StridedView::new(&a, &layout))
        .expect("device eigenvalues");

    let mut got = [eunomia::Complex::new(0.0f32, 0.0); 2];
    device
        .download(&spectrum, &mut got)
        .expect("spectrum download");
    got.sort_by(|lhs, rhs| lhs.re.total_cmp(&rhs.re));

    let host_matrix = leto::Array::from_shape_vec([2, 2], a_host.to_vec()).expect("host matrix");
    let mut expected = leto_ops::eigenvalues(&host_matrix.view()).expect("leto eigenvalues");
    expected.sort_by(|lhs, rhs| lhs.re.total_cmp(&rhs.re));

    assert_eq!(got.len(), expected.len(), "{name}: spectrum length");
    for (index, (device_value, host_value)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (device_value.re - host_value.re).abs() < DECOMP_BOUND
                && (device_value.im - host_value.im).abs() < DECOMP_BOUND,
            "{name}: eigenvalue {index}: device {device_value:?} vs leto {host_value:?}"
        );
    }
}
