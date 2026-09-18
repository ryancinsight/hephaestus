//! Host instantiation of the shared scaled dot-product attention clause, plus
//! host-specific cases it does not reach.

use hephaestus_conformance::assert_attention_contract;
use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionGradientViews, AttentionMask,
    AttentionOps, ComputeDevice, StridedView,
};
use hephaestus_host::{HostAttentionOps, HostDevice};
use leto::Layout;

#[test]
fn host_satisfies_the_attention_contract() {
    assert_attention_contract(&HostDevice::new(), &HostAttentionOps);
}

/// A non-finite query and an out-of-range weight both fail at once; the
/// canonical priority (`NonFiniteQuery` before `InvalidWeights`, mirroring
/// `hephaestus-wgpu`'s `atomicMin`-combined status) must win, and no
/// destination may be written. This exercises the host's independent
/// per-check recomputation rather than leto-ops' own sequential
/// `validate_backward`, which checks `grad_output` (and here, transitively,
/// nothing about the query) before the probability row.
#[test]
fn nonfinite_query_precedes_invalid_weights() {
    let device = HostDevice::new();
    let layout = Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid fixture layout");
    let nonfinite = device.upload(&[f32::NAN]).expect("non-finite upload");
    let one = device.upload(&[1.0_f32]).expect("unit upload");
    let invalid_weights = device.upload(&[2.0_f32]).expect("invalid weights upload");
    let destination = device.upload(&[3.0_f32]).expect("gradient sentinel upload");

    let error = HostAttentionOps
        .attention_backward_accumulate(
            &device,
            AttentionBackwardOperands {
                grad_output: StridedView::new(&one, &layout),
                query: StridedView::new(&nonfinite, &layout),
                key: StridedView::new(&one, &layout),
                value: StridedView::new(&one, &layout),
                weights: StridedView::new(&invalid_weights, &layout),
                scale: 1.0,
                gradients: AttentionGradientViews {
                    query: None,
                    key: None,
                    value: Some(StridedView::new(&destination, &layout)),
                },
            },
        )
        .expect_err("combined semantic failures must use canonical priority");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention query contains a non-finite value"
    );
    let mut actual = [0.0_f32];
    device
        .download(&destination, &mut actual)
        .expect("gradient download");
    assert_eq!(actual, [3.0], "combined-failure atomicity: value gradient");
}

/// Self-attention over one shared buffer (`query == key == value`) must not
/// deadlock the host's `RwLock`-backed buffers and must still produce the
/// leto oracle's result; `hephaestus_host::operands::with_operand_reads`
/// exists specifically to take one guard for the aliased allocation.
#[test]
fn self_attention_over_one_shared_buffer_does_not_deadlock() {
    let device = HostDevice::new();
    let layout = Layout::try_new([1, 2, 2], [4, 2, 1], 0).expect("valid fixture layout");
    let shared = device
        .upload(&[1.0_f32, 0.0, 0.0, 1.0])
        .expect("shared upload");
    let output = device.upload(&[-3.0_f32; 4]).expect("output upload");
    let weights = device.upload(&[-3.0_f32; 4]).expect("weights upload");

    HostAttentionOps
        .attention_forward_into(
            &device,
            AttentionForwardOperands {
                query: StridedView::new(&shared, &layout),
                key: StridedView::new(&shared, &layout),
                value: StridedView::new(&shared, &layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &layout),
                weights: StridedView::new(&weights, &layout),
            },
        )
        .expect("self-attention over a shared buffer must not deadlock");

    let mut weights_actual = [0.0_f32; 4];
    device
        .download(&weights, &mut weights_actual)
        .expect("weights download");
    for row in weights_actual.chunks_exact(2) {
        let sum: f32 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "row must be a probability distribution: {row:?}"
        );
    }
}

/// `weights` naming the same allocation as a requested value-gradient
/// destination is illegal aliasing between a readable operand and a writable
/// target; the plan rejects it before any preflight or arithmetic runs.
#[test]
fn value_gradient_aliasing_weights_is_rejected() {
    let device = HostDevice::new();
    let layout = Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid fixture layout");
    let one = device.upload(&[1.0_f32]).expect("unit upload");

    let error = HostAttentionOps
        .attention_backward_accumulate(
            &device,
            AttentionBackwardOperands {
                grad_output: StridedView::new(&one, &layout),
                query: StridedView::new(&one, &layout),
                key: StridedView::new(&one, &layout),
                value: StridedView::new(&one, &layout),
                weights: StridedView::new(&one, &layout),
                scale: 1.0,
                gradients: AttentionGradientViews {
                    query: None,
                    key: None,
                    value: Some(StridedView::new(&one, &layout)),
                },
            },
        )
        .expect_err("value gradient aliasing weights must be rejected");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention writable buffers must not alias readable operands or each other"
    );
}
