//! Symmetric-indefinite clauses: UDU, Bunch-Kaufman, and their Leto differentials.

use hephaestus_core::{ComputeDevice, DecompositionOps, StridedView};
use leto::Layout;

use hephaestus_core::{BunchKaufmanHandle, UduHandle};

use super::common::DECOMP_BOUND;

/// Symmetric-indefinite factorizations: `U·D·Uᵀ` on the SPD fixture
/// `[[4,2],[2,3]]` (det 8, exact solve) with `f64` reconstruction, and
/// Bunch–Kaufman on the indefinite fixture `[[1,2],[2,1]]` (spectrum
/// `{3, −1}`, which plain Cholesky rejects) with permutation validity
/// and `f64` reconstruction `P·A·Pᵀ = L·D·Lᵀ`. `D` storage is
/// discriminated by buffer length (dense `n×n` vs packed diagonal), the
/// same convention rule as QR's `R`.
pub(crate) fn symmetric_indefinite_factorizations_reconstruct<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let square = Layout::c_contiguous([2, 2]).expect("2x2 layout");

    // U·D·Uᵀ on the SPD fixture.
    let a_host = [4.0f32, 2.0, 2.0, 3.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let udu = ops
        .udu(device, StridedView::new(&a, &square))
        .expect("UDU factorization");
    assert_eq!(udu.order(), 2, "{name}: UDU order");
    assert!(
        (udu.det() - 8.0).abs() < DECOMP_BOUND,
        "{name}: UDU det {} != 8",
        udu.det()
    );
    let rhs = device.upload(&[6.0f32, 5.0]).expect("rhs upload");
    let x = udu.solve(device, &rhs).expect("UDU solve");
    let mut got = [0.0f32; 2];
    device.download(&x, &mut got).expect("solution download");
    for (index, value) in got.iter().enumerate() {
        assert!(
            (value - 1.0).abs() < DECOMP_BOUND,
            "{name}: UDU solution component {index} = {value}, expected 1"
        );
    }
    let mut u = [0.0f32; 4];
    device.download(udu.u_buffer(), &mut u).expect("U download");
    let d_values = download_diagonal(device, udu.d_buffer(), 2, name);
    let mut max_err = 0.0f64;
    for row in 0..2 {
        for col in 0..2 {
            let mut acc = 0.0f64;
            for k in 0..2 {
                let u_rk = unit_upper(&u, row, k);
                let u_ck = unit_upper(&u, col, k);
                acc += u_rk * d_values[k * 2 + k] * u_ck;
            }
            max_err = max_err.max((acc - f64::from(a_host[row * 2 + col])).abs());
        }
    }
    assert!(
        max_err < f64::from(DECOMP_BOUND),
        "{name}: UDU reconstruction error {max_err}"
    );

    // Bunch–Kaufman on the indefinite fixture Cholesky rejects.
    let ind_host = [1.0f32, 2.0, 2.0, 1.0];
    let indefinite = device.upload(&ind_host).expect("indefinite upload");
    let bk = ops
        .bunch_kaufman(device, StridedView::new(&indefinite, &square))
        .expect("Bunch-Kaufman of an indefinite matrix");
    assert_eq!(bk.order(), 2, "{name}: Bunch-Kaufman order");
    let permutation = bk.permutation();
    assert_eq!(permutation.len(), 2, "{name}: BK permutation length");
    let mut seen = [false; 2];
    for &p in permutation {
        assert!(
            p < 2 && !seen[p],
            "{name}: invalid BK permutation {permutation:?}"
        );
        seen[p] = true;
    }
    let mut l = [0.0f32; 4];
    device.download(bk.l_buffer(), &mut l).expect("L download");
    let d_values = download_diagonal(device, bk.d_buffer(), 2, name);
    let mut max_err = 0.0f64;
    for row in 0..2 {
        for col in 0..2 {
            let mut acc = 0.0f64;
            for j in 0..2 {
                for k in 0..2 {
                    let l_rj = unit_lower(&l, row, j);
                    let l_ck = unit_lower(&l, col, k);
                    acc += l_rj * d_values[j * 2 + k] * l_ck;
                }
            }
            let gathered = f64::from(ind_host[permutation[row] * 2 + permutation[col]]);
            max_err = max_err.max((acc - gathered).abs());
        }
    }
    assert!(
        max_err < f64::from(DECOMP_BOUND),
        "{name}: Bunch-Kaufman reconstruction error {max_err}"
    );
}

/// Read a `D` factor stored either dense (`n×n`) or as a packed diagonal
/// (`n`), returning dense `f64` storage.
fn download_diagonal<D: ComputeDevice>(
    device: &D,
    buffer: &D::Buffer<f32>,
    n: usize,
    name: &str,
) -> Vec<f64> {
    use hephaestus_core::DeviceBuffer;
    let len = buffer.len();
    let mut host = vec![0.0f32; len];
    device.download(buffer, &mut host).expect("D download");
    let mut dense = vec![0.0f64; n * n];
    if len == n {
        for (k, value) in host.iter().enumerate() {
            dense[k * n + k] = f64::from(*value);
        }
    } else if len == n * n {
        for (index, value) in host.iter().enumerate() {
            dense[index] = f64::from(*value);
        }
    } else {
        panic!("{name}: D buffer must be n or n*n elements, got {len}");
    }
    dense
}

/// Unit-triangular element access: implicit ones on the diagonal, zeros
/// beyond, stored values below (lower) in `f64`.
fn unit_lower(storage: &[f32; 4], row: usize, col: usize) -> f64 {
    match col.cmp(&row) {
        std::cmp::Ordering::Less => f64::from(storage[row * 2 + col]),
        std::cmp::Ordering::Equal => 1.0,
        std::cmp::Ordering::Greater => 0.0,
    }
}

/// Unit-upper access mirrors [`unit_lower`].
fn unit_upper(storage: &[f32; 4], row: usize, col: usize) -> f64 {
    match col.cmp(&row) {
        std::cmp::Ordering::Greater => f64::from(storage[row * 2 + col]),
        std::cmp::Ordering::Equal => 1.0,
        std::cmp::Ordering::Less => 0.0,
    }
}

/// UDUᵀ of the SPD matrix `[[4,12,-16],[12,37,-43],[-16,-43,98]]` agrees
/// with leto on the unit-upper factor, the diagonal, and the determinant.
pub(crate) fn udu_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [4.0f32, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([3, 3]).expect("matrix layout");
    let decomp = ops
        .udu(device, StridedView::new(&a, &layout))
        .expect("device UDU");

    let host_matrix = leto::Array::from_shape_vec([3, 3], a_host.to_vec()).expect("host matrix");
    let host_decomp = leto_ops::udu_decompose(&host_matrix.view()).expect("leto UDU");

    assert_eq!(decomp.order(), 3, "{name}: UDU order");

    let mut u = [0.0f32; 9];
    let mut d = [0.0f32; 3];
    device
        .download(decomp.u_buffer(), &mut u)
        .expect("U download");
    device
        .download(decomp.d_buffer(), &mut d)
        .expect("D download");
    let u_view = host_decomp.u();
    let expected_u = leto::Storage::as_slice(u_view.storage());
    for (index, (device_value, host_value)) in u.iter().zip(expected_u).enumerate() {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: UDU U factor {index}: device {device_value} vs leto {host_value}"
        );
    }
    for (index, (device_value, host_value)) in d.iter().zip(host_decomp.diagonal()).enumerate() {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: UDU diagonal {index}: device {device_value} vs leto {host_value}"
        );
    }
    assert!(
        (decomp.det() - host_decomp.det()).abs() < DECOMP_BOUND,
        "{name}: UDU det {} vs leto {}",
        decomp.det(),
        host_decomp.det()
    );
}

/// Bunch–Kaufman of the symmetric indefinite `[[1,2,3],[2,4,5],[3,5,6]]`
/// agrees with leto on the pivot choice and the unit-lower factor; the
/// block-diagonal factor is compared dense-to-dense when the backend
/// stores it densely, and against leto's diagonal when the backend packs
/// it (the packed form exists only when every 2×2 block is trivial, which
/// the dense-form comparison then also witnesses).
pub(crate) fn bunch_kaufman_matches_the_leto_reference<D, R>(device: &D, ops: &R)
where
    D: ComputeDevice,
    R: DecompositionOps<D>,
{
    let name = device.backend_name();
    let a_host = [1.0f32, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0];
    let a = device.upload(&a_host).expect("matrix upload");
    let layout = Layout::c_contiguous([3, 3]).expect("matrix layout");
    let decomp = ops
        .bunch_kaufman(device, StridedView::new(&a, &layout))
        .expect("device Bunch–Kaufman");

    let host_matrix = leto::Array::from_shape_vec([3, 3], a_host.to_vec()).expect("host matrix");
    let host_decomp = leto_ops::bunch_kaufman(&host_matrix.view()).expect("leto Bunch–Kaufman");

    assert_eq!(decomp.order(), 3, "{name}: Bunch–Kaufman order");
    assert_eq!(
        decomp.permutation(),
        host_decomp.permutation(),
        "{name}: Bunch–Kaufman pivot choice diverges from leto"
    );

    let mut l = [0.0f32; 9];
    device
        .download(decomp.l_buffer(), &mut l)
        .expect("L download");
    let l_view = host_decomp.l();
    let expected_l = leto::Storage::as_slice(l_view.storage());
    for (index, (device_value, host_value)) in l.iter().zip(expected_l).enumerate() {
        assert!(
            (device_value - host_value).abs() < DECOMP_BOUND,
            "{name}: Bunch–Kaufman L factor {index}: device {device_value} vs leto {host_value}"
        );
    }

    let d_view = host_decomp.d();
    let expected_d = leto::Storage::as_slice(d_view.storage());
    let d_len = hephaestus_core::DeviceBuffer::len(decomp.d_buffer());
    if d_len == 9 {
        let mut d = [0.0f32; 9];
        device
            .download(decomp.d_buffer(), &mut d)
            .expect("D download");
        for (index, (device_value, host_value)) in d.iter().zip(expected_d).enumerate() {
            assert!(
                (device_value - host_value).abs() < DECOMP_BOUND,
                "{name}: Bunch–Kaufman D factor {index}: device {device_value} vs leto {host_value}"
            );
        }
    } else {
        assert_eq!(d_len, 3, "{name}: Bunch–Kaufman D is dense n² or packed n");
        let mut d = [0.0f32; 3];
        device
            .download(decomp.d_buffer(), &mut d)
            .expect("D download");
        for (index, device_value) in d.iter().enumerate() {
            let host_value = expected_d[index * 3 + index];
            assert!(
                (device_value - host_value).abs() < DECOMP_BOUND,
                "{name}: Bunch–Kaufman packed D {index}: device {device_value} vs leto {host_value}"
            );
        }
    }
}
