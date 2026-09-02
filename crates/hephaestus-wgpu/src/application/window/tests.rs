use hephaestus_core::{
    ComputeDevice, PoolingBackwardOperands, PoolingForwardOperands, PoolingMode, PoolingOps,
    SlidingWindowFoldOperands, SlidingWindowOps, SlidingWindowUnfoldOperands, StridedView,
};
use leto::{Layout, WindowParameters};

use super::{WgpuPoolingOps, WgpuSlidingWindowOps};
use crate::WgpuDevice;

#[test]
fn pooling_and_sliding_window_kernels_match_reference_values() {
    let Some(device) = device_or_skip() else {
        return;
    };

    let parameters = WindowParameters::new([2], [1], [0], [1]).expect("valid window parameters");
    let input_layout = Layout::c_contiguous([1, 1, 4]).expect("input layout");
    let output_layout = Layout::c_contiguous([1, 1, 3]).expect("output layout");
    let input_host = [1.0_f32, 2.0, 3.0, 4.0];
    let input = device.upload(&input_host).expect("input upload");

    let max_output = device
        .alloc_zeroed::<f32>(3)
        .expect("max output allocation");
    WgpuPoolingOps
        .pooling_forward_into(
            &device,
            PoolingForwardOperands {
                input: StridedView::new(&input, &input_layout),
                output: StridedView::new(&max_output, &output_layout),
            },
            parameters,
            PoolingMode::Maximum,
        )
        .expect("maximum pooling dispatch");
    assert_download_eq(&device, &max_output, &[2.0, 3.0, 4.0]);

    let average_output = device
        .alloc_zeroed::<f32>(3)
        .expect("average output allocation");
    WgpuPoolingOps
        .pooling_forward_into(
            &device,
            PoolingForwardOperands {
                input: StridedView::new(&input, &input_layout),
                output: StridedView::new(&average_output, &output_layout),
            },
            parameters,
            PoolingMode::Average,
        )
        .expect("average pooling dispatch");
    assert_download_eq(&device, &average_output, &[1.5, 2.5, 3.5]);

    let grad_output = device
        .upload(&[1.0_f32, 2.0, 3.0])
        .expect("gradient upload");
    let max_grad_input = device
        .alloc_zeroed::<f32>(4)
        .expect("max gradient allocation");
    WgpuPoolingOps
        .pooling_backward_accumulate(
            &device,
            PoolingBackwardOperands {
                input: Some(StridedView::new(&input, &input_layout)),
                grad_output: StridedView::new(&grad_output, &output_layout),
                grad_input: StridedView::new(&max_grad_input, &input_layout),
            },
            parameters,
            PoolingMode::Maximum,
        )
        .expect("maximum pooling backward dispatch");
    assert_download_eq(&device, &max_grad_input, &[0.0, 1.0, 2.0, 3.0]);

    let average_grad_input = device
        .alloc_zeroed::<f32>(4)
        .expect("average gradient allocation");
    WgpuPoolingOps
        .pooling_backward_accumulate(
            &device,
            PoolingBackwardOperands {
                input: None,
                grad_output: StridedView::new(&grad_output, &output_layout),
                grad_input: StridedView::new(&average_grad_input, &input_layout),
            },
            parameters,
            PoolingMode::Average,
        )
        .expect("average pooling backward dispatch");
    assert_download_eq(&device, &average_grad_input, &[0.5, 1.5, 2.5, 1.5]);

    let unfold_output_layout = Layout::c_contiguous([1, 2, 3]).expect("unfold output layout");
    let unfold_output = device
        .alloc_zeroed::<f32>(6)
        .expect("unfold output allocation");
    WgpuSlidingWindowOps
        .unfold_into(
            &device,
            SlidingWindowUnfoldOperands {
                input: StridedView::new(&input, &input_layout),
                output: StridedView::new(&unfold_output, &unfold_output_layout),
            },
            parameters,
        )
        .expect("unfold dispatch");
    assert_download_eq(&device, &unfold_output, &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);

    let fold_output = device
        .alloc_zeroed::<f32>(4)
        .expect("fold output allocation");
    WgpuSlidingWindowOps
        .fold_into(
            &device,
            SlidingWindowFoldOperands {
                input: StridedView::new(&unfold_output, &unfold_output_layout),
                output: StridedView::new(&fold_output, &input_layout),
            },
            [4],
            parameters,
        )
        .expect("fold dispatch");
    assert_download_eq(&device, &fold_output, &[1.0, 4.0, 6.0, 4.0]);
}

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-window-test") {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping WGPU window test: {error}");
            None
        }
    }
}

fn assert_download_eq(device: &WgpuDevice, buffer: &crate::WgpuBuffer<f32>, expected: &[f32]) {
    let actual = device.download_owned(buffer).expect("device download");
    assert_eq!(actual, expected);
}
