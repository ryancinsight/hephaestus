//! Contract clauses for provider-owned scaled dot-product attention.
//!
//! The fixtures produce only zero, one, one-half, and integer arithmetic, all
//! exactly representable in `f32`. Backend reduction order therefore cannot
//! change the expected values and no empirical tolerance is applicable.

use core::num::NonZeroUsize;

use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionGradientViews, AttentionMask,
    AttentionOps, ComputeDevice, GroupedKeepMask, StridedView,
};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{
    AttentionGradients as LetoGradients, AttentionMask as LetoMask,
    scaled_dot_product_attention_backward_accumulate, scaled_dot_product_attention_into,
};

/// Run the shared forward, fully-masked, strided, and additive-backward clauses.
///
/// # Panics
///
/// Panics with the violated clause when the backend does not satisfy the
/// provider contract.
pub fn assert_attention_contract<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    unrestricted_and_causal_forward(device, operations);
    grouped_causal_forward_and_backward(device, operations);
    fully_masked_rows_are_zero(device, operations);
    finite_extreme_convex_output_is_preserved(device, operations);
    forward_semantic_failures_are_atomic(device, operations);
    backward_semantic_failures_are_atomic(device, operations);
}

fn unrestricted_and_causal_forward<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    let query_host = [0.0_f32; 4];
    let key_host = [0.0_f32; 4];
    let value_host = [2.0_f32, 4.0, 6.0, 10.0];
    let tensor_layout =
        Layout::try_new([1, 2, 2], [4, 2, 1], 0).expect("valid conformance fixture layout");

    let fixture = MaskPolicyFixture {
        query: &query_host,
        key: &key_host,
        value: &value_host,
        layout: tensor_layout,
    };
    assert_mask_policy(
        device,
        operations,
        fixture,
        AttentionMask::unrestricted(),
        LetoMask::Unmasked,
        "unrestricted attention",
    );
    assert_mask_policy(
        device,
        operations,
        fixture,
        AttentionMask::causal(),
        LetoMask::Causal,
        "causal attention",
    );
}

#[derive(Clone, Copy)]
struct MaskPolicyFixture<'a> {
    query: &'a [f32; 4],
    key: &'a [f32; 4],
    value: &'a [f32; 4],
    layout: Layout<3>,
}

fn assert_mask_policy<D, O>(
    device: &D,
    operations: &O,
    fixture: MaskPolicyFixture<'_>,
    mask: AttentionMask<'_, D::Buffer<f32>>,
    leto_mask: LetoMask<'_, f32>,
    clause: &str,
) where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    let mut expected_output = [-3.0_f32; 4];
    let mut expected_weights = [-3.0_f32; 4];
    scaled_dot_product_attention_into(
        &ArrayView::new(fixture.layout, fixture.query),
        &ArrayView::new(fixture.layout, fixture.key),
        &ArrayView::new(fixture.layout, fixture.value),
        leto_mask,
        1.0,
        &mut ArrayViewMut::new(fixture.layout, &mut expected_output),
        &mut ArrayViewMut::new(fixture.layout, &mut expected_weights),
    )
    .expect("Leto mask-policy oracle");

    let query = device.upload(fixture.query).expect("query upload");
    let key = device.upload(fixture.key).expect("key upload");
    let value = device.upload(fixture.value).expect("value upload");
    let output = device.upload(&[-3.0_f32; 4]).expect("output upload");
    let weights = device.upload(&[-3.0_f32; 4]).expect("weights upload");
    operations
        .attention_forward_into(
            device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &fixture.layout),
                key: StridedView::new(&key, &fixture.layout),
                value: StridedView::new(&value, &fixture.layout),
                mask,
                scale: 1.0,
                output: StridedView::new(&output, &fixture.layout),
                weights: StridedView::new(&weights, &fixture.layout),
            },
        )
        .expect(clause);
    assert_download_eq(device, &output, &expected_output, clause);
    assert_download_eq(device, &weights, &expected_weights, clause);
}

fn grouped_causal_forward_and_backward<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    let query_host = [9.0, 0.0, 0.0, 8.0, 0.0, 0.0, 7.0, 0.0, 0.0, 6.0, 0.0, 0.0];
    let key_host = query_host;
    let value_host = [9.0, 2.0, 4.0, 8.0, 6.0, 10.0, 7.0, 1.0, 3.0, 6.0, 5.0, 7.0];
    let mask_host = [1.0_f32, 0.0];
    let query_layout =
        Layout::try_new([2, 2, 2], [6, 3, 1], 1).expect("valid conformance fixture layout");
    let key_layout = query_layout;
    let value_layout = query_layout;
    let output_layout = query_layout;
    let weights_layout = query_layout;
    let mask_layout = Layout::try_new([1, 2], [2, 1], 0).expect("valid conformance fixture layout");
    let mask_layout_leto =
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout");
    let scale = 0.5_f32;

    let mut expected_output = [-3.0_f32; 12];
    let mut expected_weights = [-3.0_f32; 12];
    scaled_dot_product_attention_into(
        &ArrayView::new(query_layout, &query_host),
        &ArrayView::new(key_layout, &key_host),
        &ArrayView::new(value_layout, &value_host),
        LetoMask::CausalKeep(ArrayView::new(mask_layout_leto, &mask_host)),
        scale,
        &mut ArrayViewMut::new(output_layout, &mut expected_output),
        &mut ArrayViewMut::new(weights_layout, &mut expected_weights),
    )
    .expect("Leto attention forward oracle");

    let query = device.upload(&query_host).expect("query upload");
    let key = device.upload(&key_host).expect("key upload");
    let value = device.upload(&value_host).expect("value upload");
    let mask = device.upload(&mask_host).expect("mask upload");
    let output = device.upload(&[-3.0_f32; 12]).expect("output upload");
    let weights = device.upload(&[-3.0_f32; 12]).expect("weights upload");
    let grouped_mask = GroupedKeepMask::new(
        StridedView::new(&mask, &mask_layout),
        NonZeroUsize::new(2).expect("nonzero group width"),
    );
    operations
        .attention_forward_into(
            device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &query_layout),
                key: StridedView::new(&key, &key_layout),
                value: StridedView::new(&value, &value_layout),
                mask: AttentionMask::causal_keep(grouped_mask),
                scale,
                output: StridedView::new(&output, &output_layout),
                weights: StridedView::new(&weights, &weights_layout),
            },
        )
        .expect("attention forward dispatch");
    assert_download_eq(device, &output, &expected_output, "attention output");
    assert_download_eq(device, &weights, &expected_weights, "attention weights");

    let grad_output_host = [11.0, 1.0, 2.0, 10.0, 3.0, 4.0, 9.0, 5.0, 6.0, 8.0, 7.0, 8.0];
    let initial_gradient = [-2.0_f32; 12];
    let mut expected_query_gradient = initial_gradient;
    let mut expected_key_gradient = initial_gradient;
    let mut expected_value_gradient = initial_gradient;
    scaled_dot_product_attention_backward_accumulate(
        &ArrayView::new(output_layout, &grad_output_host),
        &ArrayView::new(query_layout, &query_host),
        &ArrayView::new(key_layout, &key_host),
        &ArrayView::new(value_layout, &value_host),
        &ArrayView::new(weights_layout, &expected_weights),
        scale,
        LetoGradients::new(
            Some(ArrayViewMut::new(
                query_layout,
                &mut expected_query_gradient,
            )),
            Some(ArrayViewMut::new(key_layout, &mut expected_key_gradient)),
            Some(ArrayViewMut::new(
                value_layout,
                &mut expected_value_gradient,
            )),
        ),
    )
    .expect("Leto attention backward oracle");

    let grad_output = device
        .upload(&grad_output_host)
        .expect("output gradient upload");
    let query_gradient = device
        .upload(&initial_gradient)
        .expect("query gradient upload");
    let key_gradient = device
        .upload(&initial_gradient)
        .expect("key gradient upload");
    let value_gradient = device
        .upload(&initial_gradient)
        .expect("value gradient upload");
    operations
        .attention_backward_accumulate(
            device,
            AttentionBackwardOperands {
                grad_output: StridedView::new(&grad_output, &output_layout),
                query: StridedView::new(&query, &query_layout),
                key: StridedView::new(&key, &key_layout),
                value: StridedView::new(&value, &value_layout),
                weights: StridedView::new(&weights, &weights_layout),
                scale,
                gradients: AttentionGradientViews {
                    query: Some(StridedView::new(&query_gradient, &query_layout)),
                    key: Some(StridedView::new(&key_gradient, &key_layout)),
                    value: Some(StridedView::new(&value_gradient, &value_layout)),
                },
            },
        )
        .expect("attention backward dispatch");
    assert_download_eq(
        device,
        &query_gradient,
        &expected_query_gradient,
        "attention query gradient",
    );
    assert_download_eq(
        device,
        &key_gradient,
        &expected_key_gradient,
        "attention key gradient",
    );
    assert_download_eq(
        device,
        &value_gradient,
        &expected_value_gradient,
        "attention value gradient",
    );
}

fn fully_masked_rows_are_zero<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    let input_host = [0.0_f32; 4];
    let mask_host = [0.0_f32, 0.0];
    let tensor_layout =
        Layout::try_new([1, 2, 2], [4, 2, 1], 0).expect("valid conformance fixture layout");
    let mask_layout = Layout::try_new([1, 2], [2, 1], 0).expect("valid conformance fixture layout");
    let input = device.upload(&input_host).expect("input upload");
    let mask = device.upload(&mask_host).expect("mask upload");
    let output = device
        .upload(&[5.0_f32; 4])
        .expect("fully masked output upload");
    let weights = device
        .upload(&[5.0_f32; 4])
        .expect("fully masked weights upload");
    let grouped_mask = GroupedKeepMask::new(
        StridedView::new(&mask, &mask_layout),
        NonZeroUsize::new(1).expect("nonzero group width"),
    );
    operations
        .attention_forward_into(
            device,
            AttentionForwardOperands {
                query: StridedView::new(&input, &tensor_layout),
                key: StridedView::new(&input, &tensor_layout),
                value: StridedView::new(&input, &tensor_layout),
                mask: AttentionMask::keep(grouped_mask),
                scale: 1.0,
                output: StridedView::new(&output, &tensor_layout),
                weights: StridedView::new(&weights, &tensor_layout),
            },
        )
        .expect("fully masked attention dispatch");
    assert_download_eq(device, &output, &[0.0; 4], "fully masked output");
    assert_download_eq(device, &weights, &[0.0; 4], "fully masked weights");
}

fn finite_extreme_convex_output_is_preserved<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    let query_layout =
        Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid conformance fixture layout");
    let key_layout =
        Layout::try_new([1, 2, 1], [2, 1, 1], 0).expect("valid conformance fixture layout");
    let weights_layout =
        Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid conformance fixture layout");
    let expected = 0.75 * f32::MAX;
    let query = device.upload(&[0.0_f32]).expect("query upload");
    let key = device.upload(&[0.0_f32; 2]).expect("key upload");
    let value = device.upload(&[expected; 2]).expect("value upload");
    let output = device.upload(&[0.0_f32]).expect("output upload");
    let weights = device.upload(&[0.0_f32; 2]).expect("weights upload");
    operations
        .attention_forward_into(
            device,
            AttentionForwardOperands {
                query: StridedView::new(&query, &query_layout),
                key: StridedView::new(&key, &key_layout),
                value: StridedView::new(&value, &key_layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &query_layout),
                weights: StridedView::new(&weights, &weights_layout),
            },
        )
        .expect("finite convex output must remain representable");
    assert_download_eq(device, &output, &[expected], "finite convex output");
    assert_download_eq(device, &weights, &[0.5, 0.5], "finite convex weights");
}

fn forward_semantic_failures_are_atomic<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    let scalar_layout =
        Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid conformance fixture layout");
    let mask_layout = Layout::try_new([1, 1], [1, 1], 0).expect("valid conformance fixture layout");
    let finite = device.upload(&[1.0_f32]).expect("finite input upload");
    let nonfinite = device.upload(&[f32::NAN]).expect("non-finite input upload");
    let output = device.upload(&[7.0_f32]).expect("sentinel output upload");
    let weights = device.upload(&[8.0_f32]).expect("sentinel weights upload");

    let error = operations
        .attention_forward_into(
            device,
            AttentionForwardOperands {
                query: StridedView::new(&nonfinite, &scalar_layout),
                key: StridedView::new(&finite, &scalar_layout),
                value: StridedView::new(&finite, &scalar_layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &scalar_layout),
                weights: StridedView::new(&weights, &scalar_layout),
            },
        )
        .expect_err("non-finite query must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention query contains a non-finite value"
    );
    assert_download_eq(device, &output, &[7.0], "non-finite query output atomicity");
    assert_download_eq(
        device,
        &weights,
        &[8.0],
        "non-finite query weight atomicity",
    );

    let grouped_mask = GroupedKeepMask::new(
        StridedView::new(&nonfinite, &mask_layout),
        NonZeroUsize::new(1).expect("nonzero group width"),
    );
    let error = operations
        .attention_forward_into(
            device,
            AttentionForwardOperands {
                query: StridedView::new(&finite, &scalar_layout),
                key: StridedView::new(&finite, &scalar_layout),
                value: StridedView::new(&finite, &scalar_layout),
                mask: AttentionMask::keep(grouped_mask),
                scale: 1.0,
                output: StridedView::new(&output, &scalar_layout),
                weights: StridedView::new(&weights, &scalar_layout),
            },
        )
        .expect_err("non-finite keep mask must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention keep mask contains a non-finite value"
    );
    assert_download_eq(device, &output, &[7.0], "non-finite mask output atomicity");
    assert_download_eq(device, &weights, &[8.0], "non-finite mask weight atomicity");

    let maximum = device.upload(&[f32::MAX]).expect("maximum input upload");
    let error = operations
        .attention_forward_into(
            device,
            AttentionForwardOperands {
                query: StridedView::new(&maximum, &scalar_layout),
                key: StridedView::new(&maximum, &scalar_layout),
                value: StridedView::new(&finite, &scalar_layout),
                mask: AttentionMask::unrestricted(),
                scale: 1.0,
                output: StridedView::new(&output, &scalar_layout),
                weights: StridedView::new(&weights, &scalar_layout),
            },
        )
        .expect_err("non-finite score arithmetic must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention weight arithmetic produced a non-finite value"
    );
    assert_download_eq(device, &output, &[7.0], "score overflow output atomicity");
    assert_download_eq(device, &weights, &[8.0], "score overflow weight atomicity");
}

fn backward_semantic_failures_are_atomic<D, O>(device: &D, operations: &O)
where
    D: ComputeDevice,
    O: AttentionOps<D, f32>,
{
    let layout =
        Layout::try_new([1, 1, 1], [1, 1, 1], 0).expect("valid conformance fixture layout");
    let one = device.upload(&[1.0_f32]).expect("unit input upload");
    let invalid_weights = device.upload(&[-0.25_f32]).expect("invalid weights upload");
    let destination = device.upload(&[3.0_f32]).expect("gradient sentinel upload");
    let error = operations
        .attention_backward_accumulate(
            device,
            AttentionBackwardOperands {
                grad_output: StridedView::new(&one, &layout),
                query: StridedView::new(&one, &layout),
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
        .expect_err("invalid probability row must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention weights do not form a probability row"
    );
    assert_download_eq(device, &destination, &[3.0], "invalid weights atomicity");

    let maximum_output_gradient = device.upload(&[f32::MAX]).expect("maximum gradient upload");
    let maximum_destination = device
        .upload(&[f32::MAX])
        .expect("maximum destination upload");
    let error = operations
        .attention_backward_accumulate(
            device,
            AttentionBackwardOperands {
                grad_output: StridedView::new(&maximum_output_gradient, &layout),
                query: StridedView::new(&one, &layout),
                key: StridedView::new(&one, &layout),
                value: StridedView::new(&one, &layout),
                weights: StridedView::new(&one, &layout),
                scale: 1.0,
                gradients: AttentionGradientViews {
                    query: None,
                    key: None,
                    value: Some(StridedView::new(&maximum_destination, &layout)),
                },
            },
        )
        .expect_err("additive gradient overflow must fail");
    assert_eq!(
        error.to_string(),
        "invalid configuration: attention value-gradient arithmetic produced a non-finite value"
    );
    assert_download_eq(
        device,
        &maximum_destination,
        &[f32::MAX],
        "gradient overflow atomicity",
    );
}

fn assert_download_eq<D, T>(device: &D, buffer: &D::Buffer<T>, expected: &[T], clause: &str)
where
    D: ComputeDevice,
    T: bytemuck::Pod + Default + Copy + PartialEq + core::fmt::Debug,
{
    let mut actual = vec![T::default(); expected.len()];
    device.download(buffer, &mut actual).expect(clause);
    assert_eq!(actual, expected, "{clause}");
}
