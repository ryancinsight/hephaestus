//! Value-semantic contracts for provider-owned CUDA runtime-rank fusion.

use std::borrow::Cow;

use hephaestus_core::{
    ComputeDevice, CudaC, DeviceBuffer, DynamicStridedView, FusedElementwiseOps, FusedExpression,
    FusedReduction, FusedReductionOps, HephaestusError,
};
use hephaestus_cuda::{CudaDevice, CudaFusionOps};
use leto::LayoutDyn;

struct AddExpression;

impl FusedExpression<CudaC> for AddExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0 + input_1")
    }
}

struct InputExpression;

impl FusedExpression<CudaC> for InputExpression {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed("input_0")
    }
}

fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(device) => Some(device),
        Err(error) => {
            if std::env::var_os("HEPHAESTUS_CUDA_REQUIRE_DEVICE").is_some() {
                panic!("CUDA device required for {test}, but acquisition failed: {error}");
            }
            eprintln!("skip {test}: CUDA device unavailable ({error})");
            None
        }
    }
}

fn layout(shape: &[usize], strides: &[isize], offset: usize) -> LayoutDyn {
    LayoutDyn::new(
        shape.to_vec().into_boxed_slice(),
        strides.to_vec().into_boxed_slice(),
        offset,
    )
    .expect("fusion test layout ranks match")
}

fn download(device: &CudaDevice, buffer: &hephaestus_cuda::CudaBuffer<f32>) -> Vec<f32> {
    let mut values = vec![0.0; buffer.len()];
    device
        .download(buffer, &mut values)
        .expect("fusion download");
    values
}

#[test]
fn runtime_rank_elementwise_broadcast_matches_reference() {
    let Some(device) = device("runtime_rank_elementwise_broadcast_matches_reference") else {
        return;
    };
    let first = device.upload(&[1.0_f32, 2.0]).expect("first input");
    let second = device
        .upload(&[10.0_f32, 20.0, 30.0])
        .expect("second input");
    let output = device.alloc_zeroed::<f32>(6).expect("output allocation");
    let first_layout = layout(&[2, 1], &[1, 1], 0);
    let second_layout = layout(&[1, 3], &[3, 1], 0);
    let output_layout = layout(&[2, 3], &[3, 1], 0);

    CudaFusionOps
        .fused_elementwise_into(
            &device,
            &AddExpression,
            &[
                DynamicStridedView::new(&first, &first_layout),
                DynamicStridedView::new(&second, &second_layout),
            ],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("runtime-rank elementwise dispatch");

    assert_eq!(
        download(&device, &output),
        [11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );
}

#[test]
fn runtime_rank_reduction_matches_reference() {
    let Some(device) = device("runtime_rank_reduction_matches_reference") else {
        return;
    };
    let input = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("reduction input");
    let output = device.alloc_zeroed::<f32>(2).expect("reduction output");
    let input_layout = layout(&[2, 3], &[3, 1], 0);
    let output_layout = layout(&[2, 1], &[1, 1], 0);

    CudaFusionOps
        .fused_reduce_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            FusedReduction::Sum,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("runtime-rank reduction dispatch");

    assert_eq!(download(&device, &output), [6.0, 15.0]);
}

#[test]
fn runtime_rank_reduction_preserves_signed_strides() {
    let Some(device) = device("runtime_rank_reduction_preserves_signed_strides") else {
        return;
    };
    let input = device.upload(&[3.0_f32, 2.0, 1.0]).expect("input");
    let output = device.alloc_zeroed::<f32>(3).expect("output");
    let input_layout = layout(&[3], &[-1], 2);
    let output_layout = layout(&[3], &[1], 0);

    CudaFusionOps
        .fused_elementwise_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("signed-stride elementwise dispatch");

    assert_eq!(download(&device, &output), [1.0, 2.0, 3.0]);
}

#[test]
fn runtime_rank_empty_sum_uses_identity_and_mean_rejects_empty_axis() {
    let Some(device) = device("runtime_rank_empty_sum_uses_identity_and_mean_rejects_empty_axis")
    else {
        return;
    };
    let input = device
        .alloc_uninitialized::<f32>(0)
        .expect("empty reduction input");
    let output = device.upload(&[7.0_f32, 11.0]).expect("reduction output");
    let input_layout = layout(&[2, 0], &[0, 1], 0);
    let output_layout = layout(&[2, 1], &[1, 1], 0);

    CudaFusionOps
        .fused_reduce_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            FusedReduction::Sum,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("empty sum identity dispatch");
    assert_eq!(download(&device, &output), [0.0, 0.0]);

    let error = CudaFusionOps
        .fused_reduce_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            FusedReduction::Mean,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect_err("empty mean must be rejected");
    assert!(
        matches!(error, HephaestusError::InvalidConfiguration { .. }),
        "unexpected error: {error}"
    );
    assert_eq!(download(&device, &output), [0.0, 0.0]);
}

#[test]
fn fusion_rejects_noninjective_output() {
    let Some(device) = device("fusion_rejects_noninjective_output") else {
        return;
    };
    let input = device.upload(&[1.0_f32, 2.0]).expect("input");
    let output = device.upload(&[7.0_f32]).expect("output");
    let input_layout = layout(&[2], &[1], 0);
    let output_layout = layout(&[2], &[0], 0);

    let error = CudaFusionOps
        .fused_elementwise_into(
            &device,
            &InputExpression,
            &[DynamicStridedView::new(&input, &input_layout)],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect_err("non-injective fusion output must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "unexpected error: {error}"
    );
    assert_eq!(download(&device, &output), [7.0]);
}
