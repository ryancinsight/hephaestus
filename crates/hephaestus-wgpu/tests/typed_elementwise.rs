#![expect(
    clippy::unwrap_used,
    reason = "ratchet HEPH-UNWRAP-1: pre-existing debt"
)]

//! Contract tests for the scalar-typed elementwise dispatch entry points.
//!
//! `binary_elementwise_typed`, `binary_elementwise_typed_into`,
//! `binary_elementwise_strided_typed`, and `binary_elementwise_strided_typed_into`
//! are declared by every accelerator backend and were exercised by none
//! (`ATLAS-ARCH-001`; Atlas conformance triage 2026-07-28). These clauses migrate
//! into the shared conformance suite per ADR 0038; the oracles below are written
//! to survive that move unchanged.
//!
//! These entry points exist for the comparison operators, which are the only
//! `TypedBinaryExpr` implementors: a comparison needs per-scalar-type codegen
//! (`select(0.0, 1.0, ..)` for `f32`, `select(0u, 1u, ..)` for `u32`,
//! `select(0, 1, ..)` for `i32`), where an arithmetic operator shares one
//! expression across types. Each scalar instantiation therefore compiles a
//! *different* shader, so all three are exercised here rather than one standing
//! in for the others.
//!
//! The result is an indicator in the operand type — exactly `1` on true and `0`
//! on false — so every assertion is an exact equality. A tolerance would only be
//! able to hide a defect.

use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError};
use hephaestus_wgpu::{
    EqOp, GeOp, GtOp, LeOp, LtOp, NeOp, StridedOperand, WgpuBuffer, WgpuDevice,
    binary_elementwise_strided_typed, binary_elementwise_strided_typed_into,
    binary_elementwise_typed, binary_elementwise_typed_into,
};
use leto::Layout;

fn device_or_skip() -> Option<WgpuDevice> {
    static DEVICE: std::sync::OnceLock<Option<WgpuDevice>> = std::sync::OnceLock::new();
    DEVICE
        .get_or_init(
            || match WgpuDevice::try_default("hephaestus-typed-elementwise-test") {
                Ok(device) => Some(device),
                Err(e) => {
                    eprintln!("skipping wgpu typed-elementwise test: {e}");
                    None
                }
            },
        )
        .clone()
}

fn op<'a, T, const N: usize>(
    buffer: &'a WgpuBuffer<T>,
    layout: &'a Layout<N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand { buffer, layout }
}

/// `u32` comparisons: unsigned ordering with an equal pair at both ends.
#[test]
fn typed_comparisons_are_exact_indicators_for_unsigned_operands() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let a_host: [u32; 5] = [0, 1, 7, 7, 4_294_967_295];
    let b_host: [u32; 5] = [0, 2, 7, 3, 0];
    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();

    let mut got = [0u32; 5];
    for (label, expected) in [
        ("eq", [1u32, 0, 1, 0, 0]),
        ("ne", [0, 1, 0, 1, 1]),
        ("lt", [0, 1, 0, 0, 0]),
        ("le", [1, 1, 1, 0, 0]),
        ("gt", [0, 0, 0, 1, 1]),
        ("ge", [1, 0, 1, 1, 1]),
    ] {
        let out = match label {
            "eq" => binary_elementwise_typed::<EqOp, u32>(&device, &a, &b),
            "ne" => binary_elementwise_typed::<NeOp, u32>(&device, &a, &b),
            "lt" => binary_elementwise_typed::<LtOp, u32>(&device, &a, &b),
            "le" => binary_elementwise_typed::<LeOp, u32>(&device, &a, &b),
            "gt" => binary_elementwise_typed::<GtOp, u32>(&device, &a, &b),
            _ => binary_elementwise_typed::<GeOp, u32>(&device, &a, &b),
        }
        .unwrap();
        device.download(&out, &mut got).unwrap();
        assert_eq!(got, expected, "u32 {label} comparison");
    }
}

/// `i32` comparisons: the discriminating case is negative operands, which a
/// kernel that reused the unsigned expression would order incorrectly.
#[test]
fn typed_comparisons_order_signed_operands_by_sign() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let a_host: [i32; 5] = [-5, -1, 0, 3, i32::MIN];
    let b_host: [i32; 5] = [2, -1, -0, -3, i32::MAX];
    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();

    let mut got = [0i32; 5];

    let lt = binary_elementwise_typed::<LtOp, i32>(&device, &a, &b).unwrap();
    device.download(&lt, &mut got).unwrap();
    // -5 < 2, -1 !< -1, 0 !< 0, 3 !< -3, i32::MIN < i32::MAX.
    assert_eq!(got, [1, 0, 0, 0, 1]);

    let gt = binary_elementwise_typed::<GtOp, i32>(&device, &a, &b).unwrap();
    device.download(&gt, &mut got).unwrap();
    assert_eq!(got, [0, 0, 0, 1, 0]);

    let eq = binary_elementwise_typed::<EqOp, i32>(&device, &a, &b).unwrap();
    device.download(&eq, &mut got).unwrap();
    assert_eq!(got, [0, 1, 1, 0, 0]);
}

/// `f32` comparisons over finite operands, including the signed zero identity
/// `-0.0 == 0.0`.
///
/// NaN and infinity operands are deliberately absent. The WGSL specification
/// states that "implementations may assume that overflow, infinities, and NaNs
/// are not present during shader execution", and that a runtime expression
/// yielding one of them produces "an indeterminate value of the target type"
/// (W3C WGSL, floating point evaluation). Asserting IEEE-754 NaN ordering here
/// would therefore be asserting a guarantee this backend does not make: the
/// current device returns `NaN != NaN` as false, which the specification
/// permits.
///
/// This is a real contract boundary rather than a gap in this test. CUDA and HIP
/// backends do provide IEEE semantics, so NaN and infinity behaviour is
/// *capability-gated*, not a universal `ComputeBackend` clause — recorded for the
/// shared suite as `ATLAS-ARCH-010`.
#[test]
fn typed_comparisons_are_exact_indicators_for_finite_floats() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let a_host: [f32; 6] = [1.0, -0.0, 2.5, -3.0, 0.25, 16_777_216.0];
    let b_host: [f32; 6] = [1.0, 0.0, 2.0, -3.0, 0.5, 16_777_215.0];
    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();

    let mut got = [0.0f32; 6];

    let eq = binary_elementwise_typed::<EqOp, f32>(&device, &a, &b).unwrap();
    device.download(&eq, &mut got).unwrap();
    // -0.0 == 0.0 is true; the last pair is adjacent at the exact-integer limit.
    assert_eq!(got, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);

    let ne = binary_elementwise_typed::<NeOp, f32>(&device, &a, &b).unwrap();
    device.download(&ne, &mut got).unwrap();
    assert_eq!(got, [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]);

    let lt = binary_elementwise_typed::<LtOp, f32>(&device, &a, &b).unwrap();
    device.download(&lt, &mut got).unwrap();
    assert_eq!(got, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    let ge = binary_elementwise_typed::<GeOp, f32>(&device, &a, &b).unwrap();
    device.download(&ge, &mut got).unwrap();
    assert_eq!(got, [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]);
}

/// The `_into` form writes caller-owned storage, matches the allocating form,
/// and is stable across repeated dispatch.
#[test]
fn typed_comparison_into_writes_caller_storage_and_matches_allocating_form() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let a = device.upload(&[7u32, 6, 5, 4, 3, 2]).unwrap();
    let b = device.upload(&[1u32, 6, 9, 4, 5, 2]).unwrap();

    let out = device.alloc_zeroed::<u32>(6).unwrap();
    binary_elementwise_typed_into::<EqOp, u32>(&device, &a, &b, &out, BlockWidth::DEFAULT).unwrap();

    let mut got = [0u32; 6];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, [0, 1, 0, 1, 0, 1]);

    let allocated = binary_elementwise_typed::<EqOp, u32>(&device, &a, &b).unwrap();
    let mut allocated_got = [0u32; 6];
    device.download(&allocated, &mut allocated_got).unwrap();
    assert_eq!(got, allocated_got);

    // Re-dispatching over unchanged inputs is idempotent.
    binary_elementwise_typed_into::<EqOp, u32>(&device, &a, &b, &out, BlockWidth::DEFAULT).unwrap();
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, [0, 1, 0, 1, 0, 1]);
}

/// Operand length disagreement is a typed rejection, not a truncated dispatch.
#[test]
fn typed_comparison_rejects_length_mismatch() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let a = device.upload(&[1u32, 2, 3]).unwrap();
    let b = device.upload(&[1u32, 2]).unwrap();

    assert!(matches!(
        binary_elementwise_typed::<EqOp, u32>(&device, &a, &b),
        Err(HephaestusError::LengthMismatch { .. })
    ));

    let out = device.alloc_zeroed::<u32>(3).unwrap();
    assert!(matches!(
        binary_elementwise_typed_into::<EqOp, u32>(&device, &a, &b, &out, BlockWidth::DEFAULT),
        Err(HephaestusError::LengthMismatch { .. })
    ));
}

/// The strided `_into` form reads operands through their layouts rather than in
/// buffer order. A transposed view is the discriminating case: same bytes,
/// different traversal.
#[test]
fn typed_comparison_strided_into_respects_source_strides() {
    let Some(device) = device_or_skip() else {
        return;
    };

    // Buffer holds a 2x3 C-contiguous matrix [[1,2,3],[4,5,6]].
    let source = device.upload(&[1u32, 2, 3, 4, 5, 6]).unwrap();
    // Read as its 3x2 transpose [[1,4],[2,5],[3,6]] — same bytes, swapped strides.
    let transposed = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");

    // Compare against the transpose's own values with two deliberate mismatches.
    let probe = device.upload(&[1u32, 4, 9, 5, 3, 9]).unwrap();
    let dense = Layout::c_contiguous([3, 2]).unwrap();

    let out = device.alloc_zeroed::<u32>(6).unwrap();
    binary_elementwise_strided_typed_into::<EqOp, u32, 2>(
        &device,
        op(&source, &transposed),
        op(&probe, &dense),
        op(&out, &dense),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = [0u32; 6];
    device.download(&out, &mut got).unwrap();
    // Transpose reads [1,4,2,5,3,6]; probe is [1,4,9,5,3,9].
    assert_eq!(got, [1, 1, 0, 1, 1, 0]);
}

/// The allocating strided form broadcasts a zero-stride operand across the
/// output shape and returns dense storage.
#[test]
fn typed_comparison_strided_broadcasts_into_dense_output() {
    let Some(device) = device_or_skip() else {
        return;
    };

    // One row broadcast down two rows via a zero stride on axis 0.
    let row = device.upload(&[1u32, 2, 3]).unwrap();
    let broadcast = Layout::try_new([2, 3], [0, 1], 0).expect("valid test layout");

    let matrix = device.upload(&[1u32, 9, 3, 9, 2, 3]).unwrap();
    let dense = Layout::c_contiguous([2, 3]).unwrap();

    let out = binary_elementwise_strided_typed::<EqOp, u32, 2>(
        &device,
        op(&row, &broadcast),
        op(&matrix, &dense),
        [2, 3],
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = [0u32; 6];
    device.download(&out, &mut got).unwrap();
    // Row [1,2,3] compared against [[1,9,3],[9,2,3]].
    assert_eq!(got, [1, 0, 1, 0, 1, 1]);
}
