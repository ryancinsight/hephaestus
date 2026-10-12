//! Contract clauses for the device-neutral dense vector seam.
//!
//! Each clause is generic over the device, the seam, and the fixture scalar,
//! so every backend runs the same assertions per scalar. All fixtures are
//! small integer-valued or dyadic vectors, so every product, sum, and
//! quotient is exactly representable in both `f32` and `f64`. The dot
//! fixture therefore uses exact equality, while the norm fixture permits
//! one output ULP for the backend square-root operation.

use core::fmt::Debug;

use eunomia::Pod;
use hephaestus_core::{ComputeDevice, DenseVectorOps};

/// Scalar a dense-vector conformance run can fixture.
///
/// Fixtures are built from exactly representable `f64` values, and norm
/// assertions compare through [`to_f64`](Self::to_f64) against a
/// scalar-width ULP bound.
pub trait ContractScalar: Pod + Copy + PartialEq + Debug {
    /// Exactly representable fixture value (small integers and dyadics only).
    fn from_fixture(value: f64) -> Self;
    /// Lossless widening for norm assertions.
    fn to_f64(self) -> f64;
    /// Machine epsilon of this scalar width.
    fn epsilon() -> f64;
}

impl ContractScalar for f32 {
    fn from_fixture(value: f64) -> Self {
        // Fixtures are small integers or dyadics: exactly representable.
        value as f32
    }

    fn to_f64(self) -> f64 {
        f64::from(self)
    }

    fn epsilon() -> f64 {
        f64::from(f32::EPSILON)
    }
}

impl ContractScalar for f64 {
    fn from_fixture(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn epsilon() -> f64 {
        f64::EPSILON
    }
}

/// Upload, run `op`, download `LEN` elements from `buffer`.
fn download<D: ComputeDevice, T: ContractScalar, const LEN: usize>(
    device: &D,
    buffer: &D::Buffer<T>,
) -> [T; LEN] {
    let mut got = [T::from_fixture(0.0); LEN];
    device.download(buffer, &mut got).expect("download");
    got
}

/// Fixture vector from exactly representable values.
pub(crate) fn fixture<T: ContractScalar, const LEN: usize>(values: [f64; LEN]) -> [T; LEN] {
    values.map(T::from_fixture)
}

/// Assert that one positive normal result is within one output ULP.
pub(crate) fn assert_one_ulp<T: ContractScalar>(
    actual: T,
    expected: f64,
    name: &str,
    clause: &str,
) {
    // For positive normal values, expected * EPSILON upper-bounds one ULP
    // throughout the expected value's binade.
    let one_ulp = expected * T::epsilon();
    let actual = actual.to_f64();
    assert!(
        (actual - expected).abs() <= one_ulp,
        "{name}: {clause}: expected {expected}, got {actual}, one-ULP bound {one_ulp}"
    );
}

/// Run every dense-vector clause against one backend.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// contract. Backends call this from a test that has already acquired a
/// device.
pub fn assert_dense_vector_contract<D, E, T>(device: &D, ops: &E)
where
    D: ComputeDevice,
    E: DenseVectorOps<D, T>,
    T: ContractScalar,
{
    let name = device.backend_name();

    // copy_vector: device-to-device copy is exact.
    let source = device
        .upload(&fixture([1.0, 2.0, 3.0, 4.0]))
        .expect("source");
    let target = device.alloc_zeroed::<T>(4).expect("target");
    ops.copy_vector(device, &source, &target).expect("copy");
    assert_eq!(
        download::<_, _, 4>(device, &target),
        fixture([1.0, 2.0, 3.0, 4.0]),
        "{name}: copy_vector"
    );

    // copy_vector: a length mismatch is rejected.
    let short = device.alloc_zeroed::<T>(3).expect("short target");
    assert!(
        ops.copy_vector(device, &source, &short).is_err(),
        "{name}: copy_vector must reject a length mismatch"
    );

    // scale_vector: in-place scale by a dyadic factor is exact.
    ops.scale_vector(device, &target, T::from_fixture(2.0))
        .expect("scale");
    assert_eq!(
        download::<_, _, 4>(device, &target),
        fixture([2.0, 4.0, 6.0, 8.0]),
        "{name}: scale_vector"
    );

    // axpy: target += factor * source.
    let y = device
        .upload(&fixture([4.0, 5.0, 6.0, 8.0]))
        .expect("axpy target");
    ops.axpy(device, &y, &source, T::from_fixture(2.0))
        .expect("axpy");
    assert_eq!(
        download::<_, _, 4>(device, &y),
        fixture([6.0, 9.0, 12.0, 16.0]),
        "{name}: axpy"
    );

    // xpay: target = source + factor * target, distinguishing the accumulator
    // scaling from axpy over the same operands.
    let y = device
        .upload(&fixture([4.0, 5.0, 6.0, 8.0]))
        .expect("xpay target");
    ops.xpay(device, &y, &source, T::from_fixture(2.0))
        .expect("xpay");
    assert_eq!(
        download::<_, _, 4>(device, &y),
        fixture([9.0, 12.0, 15.0, 20.0]),
        "{name}: xpay"
    );

    // Elementwise arithmetic into distinct storage.
    let left = device.upload(&fixture([8.0, 6.0, 4.0, 2.0])).expect("left");
    let right = device
        .upload(&fixture([2.0, 4.0, 8.0, 0.5]))
        .expect("right");
    let out = device.alloc_zeroed::<T>(4).expect("out");
    ops.subtract_into(device, &left, &right, &out).expect("sub");
    assert_eq!(
        download::<_, _, 4>(device, &out),
        fixture([6.0, 2.0, -4.0, 1.5]),
        "{name}: subtract_into"
    );
    ops.add_into(device, &left, &right, &out).expect("add");
    assert_eq!(
        download::<_, _, 4>(device, &out),
        fixture([10.0, 10.0, 12.0, 2.5]),
        "{name}: add_into"
    );
    ops.multiply_into(device, &left, &right, &out).expect("mul");
    assert_eq!(
        download::<_, _, 4>(device, &out),
        fixture([16.0, 24.0, 32.0, 1.0]),
        "{name}: multiply_into"
    );
    ops.divide_into(device, &left, &right, &out).expect("div");
    assert_eq!(
        download::<_, _, 4>(device, &out),
        fixture([4.0, 1.5, 0.5, 4.0]),
        "{name}: divide_into"
    );

    // Prepared dot: exact value, idempotent re-dispatch, rebind observation,
    // and mismatched-operand rejection.
    let a = device
        .upload(&fixture([1.0, 2.0, 3.0, 4.0]))
        .expect("dot lhs");
    let b = device
        .upload(&fixture([8.0, 4.0, 2.0, 1.0]))
        .expect("dot rhs");
    let prepared = ops.prepare_dot(device, &a, &b).expect("prepare dot");
    assert_eq!(
        ops.dot_prepared(device, &prepared, &a, &b).expect("dot"),
        T::from_fixture(26.0),
        "{name}: prepared dot"
    );
    assert_eq!(
        ops.dot_prepared(device, &prepared, &a, &b).expect("dot"),
        T::from_fixture(26.0),
        "{name}: prepared dot must be idempotent over unchanged operands"
    );
    device
        .write_buffer(&a, &fixture([2.0, 2.0, 2.0, 2.0]))
        .expect("lhs rewrite");
    assert_eq!(
        ops.dot_prepared(device, &prepared, &a, &b).expect("dot"),
        T::from_fixture(30.0),
        "{name}: prepared dot must observe writes to its bound operand"
    );
    let foreign = device
        .upload(&fixture([1.0, 1.0, 1.0, 1.0]))
        .expect("foreign");
    assert!(
        ops.dot_prepared(device, &prepared, &foreign, &b).is_err(),
        "{name}: prepared dot must reject an operand it was not prepared against"
    );

    // Prepared norm: the Pythagorean quadruple [2,3,6,0] has norm 7. The
    // integer sum of squares is exact; only the backend sqrt may round.
    let v = device
        .upload(&fixture([2.0, 3.0, 6.0, 0.0]))
        .expect("norm vector");
    assert_eq!(
        ops.norm_l1(device, &v).expect("norm_l1"),
        T::from_fixture(11.0),
        "{name}: L1 norm of the quadruple is the exact integer sum of magnitudes"
    );
    assert_eq!(
        ops.norm_max(device, &v).expect("norm_max"),
        T::from_fixture(6.0),
        "{name}: max norm of the quadruple is its largest magnitude"
    );
    let prepared = ops.prepare_norm_l2(device, &v).expect("prepare norm");
    assert_one_ulp::<T>(
        ops.norm_l2_prepared(device, &prepared, &v).expect("norm"),
        7.0,
        name,
        "prepared norm",
    );
    // Rebind: [9,12,20,0] is a Pythagorean quadruple with norm exactly 25.
    device
        .write_buffer(&v, &fixture([9.0, 12.0, 20.0, 0.0]))
        .expect("vector rewrite");
    assert_one_ulp::<T>(
        ops.norm_l2_prepared(device, &prepared, &v).expect("norm"),
        25.0,
        name,
        "prepared norm must observe writes to its bound operand",
    );
}
