//! Contract clauses for the device-neutral
//! [`hephaestus_core::CrossProductOps`] seam.
//!
//! Batched fixture of two triples: `i x j = k` and `j x k = i` (the standard
//! basis identities), each exact in `f32`.

use hephaestus_core::CrossProductOps;

/// Batched cross product against the standard basis identities.
///
/// # Panics
///
/// Panics with the violated clause when the backend disagrees.
pub fn assert_cross_product_contract<D, C>(device: &D, ops: &C)
where
    D: hephaestus_core::ComputeDevice,
    C: CrossProductOps<D, f32>,
{
    // i = (1,0,0), j = (0,1,0); k = (0,0,1), i = (1,0,0).
    let a = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 1.0];
    let expected = vec![0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0];

    let a_buf = device.upload(&a).expect("a upload");
    let b_buf = device.upload(&b).expect("b upload");
    let out = device.alloc_zeroed::<f32>(6).expect("out alloc");

    ops.cross_into(device, &a_buf, &b_buf, &out)
        .expect("cross product dispatch");

    let mut got = vec![0.0f32; 6];
    device.download(&out, &mut got).expect("download");
    assert_eq!(
        got,
        expected,
        "{}: batched cross product must match the standard basis identities",
        device.backend_name()
    );
}
