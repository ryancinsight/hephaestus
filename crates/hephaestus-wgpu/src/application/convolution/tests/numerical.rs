use hephaestus_core::{ComputeDevice, ConvolutionForwardOperands, ConvolutionOps, StridedView};
use leto::{ArrayView, ArrayViewMut, ConvolutionParameters, Layout};
use leto_ops::convolution_forward_into;

use super::{WgpuConvolutionOps, device_or_skip};
use crate::WgpuDevice;

#[test]
fn reordered_forward_matches_leto_with_derived_roundoff_bound() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let input_host = [0.1_f32, -0.2, 0.3, 0.4];
    let weight_host = [0.15_f32, -0.35];
    let bias_host = [0.05_f32];
    let input_layout = Layout::try_new([1, 1, 4], [4, 4, 1], 0).expect("valid test layout");
    let weight_layout = Layout::try_new([1, 1, 2], [2, 2, 1], 0).expect("valid test layout");
    let bias_layout = Layout::try_new([1], [1], 0).expect("valid test layout");
    let output_layout = Layout::try_new([1, 1, 3], [3, 3, 1], 0).expect("valid test layout");
    let parameters = ConvolutionParameters::new([1], [0], [1]).expect("valid parameters");
    let mut expected = [0.0_f32; 3];
    convolution_forward_into(
        &ArrayView::new(input_layout, &input_host),
        &ArrayView::new(weight_layout, &weight_host),
        Some(&ArrayView::new(bias_layout, &bias_host)),
        parameters,
        &mut ArrayViewMut::new(output_layout, &mut expected),
    )
    .expect("Leto forward");

    let input = device.upload(&input_host).expect("input upload");
    let weight = device.upload(&weight_host).expect("weight upload");
    let bias = device.upload(&bias_host).expect("bias upload");
    let output = device
        .alloc_zeroed::<f32>(expected.len())
        .expect("output allocation");
    let operations = WgpuConvolutionOps;
    let prepared = operations
        .prepare_convolution_forward(
            &device,
            ConvolutionForwardOperands {
                input: StridedView::new(&input, &input_layout),
                weight: StridedView::new(&weight, &weight_layout),
                bias: Some(StridedView::new(&bias, &bias_layout)),
                output: StridedView::new(&output, &output_layout),
            },
            parameters,
        )
        .expect("prepare forward");
    <WgpuConvolutionOps as ConvolutionOps<WgpuDevice, f32>>::dispatch_convolution_forward::<3, 1>(
        &operations,
        &device,
        &prepared,
    )
    .expect("dispatch forward");
    let mut actual = [0.0_f32; 3];
    device
        .download(&output, &mut actual)
        .expect("output download");

    let products = weight_host.len();
    let rounded_operations = products
        .checked_mul(2)
        .and_then(|count| count.checked_add(1))
        .expect("invariant: fixed test operation count fits usize");
    let rounded_operations =
        u16::try_from(rounded_operations).expect("invariant: fixed test operation count fits u16");
    let products = u16::try_from(products).expect("invariant: fixed kernel size fits u16");
    let roundoff = f32::from(rounded_operations) * f32::EPSILON;
    let gamma = roundoff / (1.0 - roundoff);
    let max_input = input_host.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let max_weight = weight_host
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max);
    let sum_magnitude = bias_host[0].abs() + f32::from(products) * max_input * max_weight;
    // Each evaluation order is bounded by gamma_n * sum(|terms|), so their
    // pairwise difference is bounded by twice that forward-error bound.
    let bound = 2.0 * gamma * sum_magnitude;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let difference = (actual - expected).abs();
        assert!(
            difference <= bound,
            "output {index} differs by {difference}, derived bound {bound}"
        );
    }
}
