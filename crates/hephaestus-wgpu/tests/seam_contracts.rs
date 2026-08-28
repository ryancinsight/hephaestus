use hephaestus_core::{
    AddOp, CombineExpr, ComputeDevice, DeviceBuffer, ElementwiseOps, FullReductionOps, IdentityOp,
    IdentityToken, OpIdentity, ProdOp, ScanDirection, ScanOps, StridedView, SumOp, UnaryExpr, Wgsl,
};
use hephaestus_wgpu::{WgpuDevice, WgpuElementwiseOps, WgpuFullReductionOps, WgpuScanOps};
use leto::Layout;

fn device() -> WgpuDevice {
    super::device_or_skip().expect("WGPU seam contract tests require a device")
}

fn download(device: &WgpuDevice, buffer: &hephaestus_wgpu::WgpuBuffer<f32>) -> Vec<f32> {
    let mut values = vec![0.0; buffer.len()];
    device
        .download(buffer, &mut values)
        .expect("download seam result");
    values
}

pub(super) fn full_reduction_honors_output_offset() {
    let device = device();
    let input = device.upload(&[1.0_f32, 2.0, 3.0]).expect("input");
    let output = device.upload(&[17.0_f32, 19.0, 23.0]).expect("output");
    let input_layout = Layout::c_contiguous([3]).expect("input layout");
    let output_layout = Layout::try_new([1], [1], 1).expect("valid test layout");

    WgpuFullReductionOps
        .reduce_full_into::<SumOp, 1>(
            &device,
            StridedView::new(&input, &input_layout),
            StridedView::new(&output, &output_layout),
        )
        .expect("offset full reduction");

    assert_eq!(download(&device, &output), [17.0, 6.0, 23.0]);
}

pub(super) fn empty_full_reductions_write_operator_identities() {
    let device = device();
    let input = device.alloc_uninitialized::<f32>(0).expect("empty input");
    let output = device.upload(&[7.0_f32, 11.0]).expect("output");
    let input_layout = Layout::c_contiguous([0]).expect("empty layout");
    let sum_layout = Layout::try_new([1], [1], 0).expect("valid test layout");
    let product_layout = Layout::try_new([1], [1], 1).expect("valid test layout");

    WgpuFullReductionOps
        .reduce_full_into::<SumOp, 1>(
            &device,
            StridedView::new(&input, &input_layout),
            StridedView::new(&output, &sum_layout),
        )
        .expect("empty sum");
    WgpuFullReductionOps
        .reduce_full_into::<ProdOp, 1>(
            &device,
            StridedView::new(&input, &input_layout),
            StridedView::new(&output, &product_layout),
        )
        .expect("empty product");

    assert_eq!(download(&device, &output), [0.0, 1.0]);
}

pub(super) fn prepared_elementwise_rejects_cross_device_dispatch() {
    let source_device = device();
    let other_device = WgpuDevice::try_default("hephaestus-seam-contracts-other")
        .expect("cross-device contract requires a second logical WGPU device");
    let input = source_device.upload(&[2.0_f32, 3.0]).expect("input");
    let output = source_device.upload(&[13.0_f32, 17.0]).expect("output");
    let layout = Layout::c_contiguous([2]).expect("layout");
    let prepared = WgpuElementwiseOps
        .prepare_unary_into::<IdentityOp, 1>(
            &source_device,
            StridedView::new(&input, &layout),
            StridedView::new(&output, &layout),
        )
        .expect("prepare elementwise");

    let error = <WgpuElementwiseOps as ElementwiseOps<WgpuDevice, f32>>::dispatch_unary::<1>(
        &WgpuElementwiseOps,
        &other_device,
        &prepared,
    )
    .expect_err("cross-device dispatch must fail");

    assert!(
        error.to_string().contains("belongs to a different device"),
        "unexpected error: {error}"
    );
    assert_eq!(download(&source_device, &output), [13.0, 17.0]);
}

pub(super) fn prepared_scan_and_full_reduction_reject_cross_device_dispatch() {
    let source_device = device();
    let other_device = WgpuDevice::try_default("hephaestus-seam-contracts-other-prepared")
        .expect("cross-device contract requires a second logical WGPU device");
    let input = source_device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0])
        .expect("input");
    let scan_output = source_device
        .upload(&[13.0_f32, 17.0, 19.0, 23.0])
        .expect("scan output");
    let reduction_output = source_device.upload(&[29.0_f32]).expect("reduction output");
    let matrix_layout = Layout::c_contiguous([2, 2]).expect("matrix layout");
    let scalar_layout = Layout::c_contiguous([1]).expect("scalar layout");
    let scan = WgpuScanOps
        .prepare_scan_axis::<SumOp, 2>(
            &source_device,
            StridedView::new(&input, &matrix_layout),
            1,
            ScanDirection::Forward,
            StridedView::new(&scan_output, &matrix_layout),
        )
        .expect("prepare scan");
    let reduction = WgpuFullReductionOps
        .prepare_reduce_full::<SumOp, 2>(
            &source_device,
            StridedView::new(&input, &matrix_layout),
            StridedView::new(&reduction_output, &scalar_layout),
        )
        .expect("prepare reduction");

    let scan_error = <WgpuScanOps as ScanOps<WgpuDevice, f32>>::dispatch_scan::<2>(
        &WgpuScanOps,
        &other_device,
        &scan,
    )
    .expect_err("cross-device scan dispatch must fail");
    let reduction_error =
        <WgpuFullReductionOps as FullReductionOps<WgpuDevice, f32>>::dispatch_full::<2>(
            &WgpuFullReductionOps,
            &other_device,
            &reduction,
        )
        .expect_err("cross-device reduction dispatch must fail");

    assert!(scan_error.to_string().contains("different device"));
    assert!(reduction_error.to_string().contains("different device"));
    assert_eq!(
        download(&source_device, &scan_output),
        [13.0, 17.0, 19.0, 23.0]
    );
    assert_eq!(download(&source_device, &reduction_output), [29.0]);
}

pub(super) fn overlapping_writable_layouts_fail_before_mutation() {
    let device = device();
    let input = device.upload(&[1.0_f32, 2.0, 3.0, 4.0]).expect("input");
    let output = device.upload(&[29.0_f32, 31.0, 37.0]).expect("output");
    let input_layout = Layout::c_contiguous([2, 2]).expect("input layout");
    let overlapping = Layout::try_new([2, 2], [1, 1], 0).expect("valid test layout");

    let elementwise_error = WgpuElementwiseOps
        .unary_into::<IdentityOp, 2>(
            &device,
            StridedView::new(&input, &input_layout),
            StridedView::new(&output, &overlapping),
        )
        .expect_err("overlapping elementwise output");
    assert!(
        elementwise_error.to_string().contains("non-overlapping"),
        "unexpected error: {elementwise_error}"
    );

    let scan_error = WgpuScanOps
        .scan_axis_into::<SumOp, 2>(
            &device,
            StridedView::new(&input, &input_layout),
            1,
            ScanDirection::Forward,
            StridedView::new(&output, &overlapping),
        )
        .expect_err("overlapping scan output");
    assert!(
        scan_error.to_string().contains("non-overlapping"),
        "unexpected error: {scan_error}"
    );
    assert_eq!(download(&device, &output), [29.0, 31.0, 37.0]);
}

pub(super) fn elementwise_and_scan_match_value_oracles() {
    let device = device();
    let lhs = device
        .upload(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("left input");
    let rhs = device
        .upload(&[6.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0])
        .expect("right input");
    let binary_output = device.alloc_zeroed::<f32>(6).expect("binary output");
    let forward_output = device.alloc_zeroed::<f32>(6).expect("forward scan");
    let reverse_output = device.alloc_zeroed::<f32>(6).expect("reverse scan");
    let layout = Layout::c_contiguous([2, 3]).expect("layout");

    WgpuElementwiseOps
        .binary_into::<AddOp, 2>(
            &device,
            StridedView::new(&lhs, &layout),
            StridedView::new(&rhs, &layout),
            StridedView::new(&binary_output, &layout),
        )
        .expect("binary addition");
    WgpuScanOps
        .scan_axis_into::<SumOp, 2>(
            &device,
            StridedView::new(&lhs, &layout),
            1,
            ScanDirection::Forward,
            StridedView::new(&forward_output, &layout),
        )
        .expect("forward row scan");
    WgpuScanOps
        .scan_axis_into::<SumOp, 2>(
            &device,
            StridedView::new(&lhs, &layout),
            0,
            ScanDirection::Reverse,
            StridedView::new(&reverse_output, &layout),
        )
        .expect("reverse column scan");

    assert_eq!(
        download(&device, &binary_output),
        [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
    );
    assert_eq!(
        download(&device, &forward_output),
        [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
    );
    assert_eq!(
        download(&device, &reverse_output),
        [5.0, 7.0, 9.0, 4.0, 5.0, 6.0]
    );
}

#[derive(Clone, Copy)]
struct InvalidWgsl;

impl UnaryExpr<Wgsl> for InvalidWgsl {
    const EXPR: &'static str = "this is not a WGSL expression";
}

pub(super) fn invalid_external_expression_is_a_typed_preparation_error() {
    let device = device();
    let input = device.upload(&[1.0_f32]).expect("input");
    let output = device.upload(&[43.0_f32]).expect("output");
    let layout = Layout::c_contiguous([1]).expect("layout");

    let result = WgpuElementwiseOps.prepare_unary_into::<InvalidWgsl, 1>(
        &device,
        StridedView::new(&input, &layout),
        StridedView::new(&output, &layout),
    );
    let error = match result {
        Ok(_) => panic!("invalid WGSL must fail preparation"),
        Err(error) => error,
    };

    assert!(
        error.to_string().contains("compilation failed"),
        "unexpected error: {error}"
    );
    assert_eq!(download(&device, &output), [43.0]);
}

#[derive(Clone, Copy)]
struct InvalidCombine;

impl CombineExpr<Wgsl> for InvalidCombine {
    const EXPR: &'static str = "this is not a WGSL expression";
}

impl OpIdentity<InvalidCombine> for f32 {
    const IDENTITY: Self = 0.0;
}

impl IdentityToken<InvalidCombine, Wgsl> for f32 {
    const TOKEN: &'static str = "0.0";
}

pub(super) fn invalid_external_combine_is_a_typed_preparation_error() {
    let device = device();
    let input = device.upload(&[1.0_f32, 2.0]).expect("input");
    let output = device.upload(&[47.0_f32]).expect("output");
    let input_layout = Layout::c_contiguous([2]).expect("input layout");
    let output_layout = Layout::c_contiguous([1]).expect("output layout");

    let result = WgpuFullReductionOps.prepare_reduce_full::<InvalidCombine, 1>(
        &device,
        StridedView::new(&input, &input_layout),
        StridedView::new(&output, &output_layout),
    );
    let error = match result {
        Ok(_) => panic!("invalid WGSL must fail preparation"),
        Err(error) => error,
    };

    assert!(
        error.to_string().contains("compilation failed"),
        "unexpected error: {error}"
    );
    assert_eq!(download(&device, &output), [47.0]);
}

pub(super) fn full_reduction_rejects_foreign_buffers_before_mutation() {
    let source_device = device();
    let other_device = WgpuDevice::try_default("hephaestus-seam-contracts-foreign-buffers")
        .expect("foreign-buffer contract requires a second logical WGPU device");
    let source_input = source_device.upload(&[1.0_f32, 2.0]).expect("input");
    let source_output = source_device.upload(&[53.0_f32]).expect("output");
    let foreign_input = other_device.upload(&[3.0_f32, 5.0]).expect("foreign input");
    let foreign_output = other_device.upload(&[59.0_f32]).expect("foreign output");
    let input_layout = Layout::c_contiguous([2]).expect("input layout");
    let output_layout = Layout::c_contiguous([1]).expect("output layout");

    let input_error = match WgpuFullReductionOps.prepare_reduce_full::<SumOp, 1>(
        &source_device,
        StridedView::new(&foreign_input, &input_layout),
        StridedView::new(&source_output, &output_layout),
    ) {
        Ok(_) => panic!("foreign input must fail preparation"),
        Err(error) => error,
    };
    let output_error = match WgpuFullReductionOps.prepare_reduce_full::<SumOp, 1>(
        &source_device,
        StridedView::new(&source_input, &input_layout),
        StridedView::new(&foreign_output, &output_layout),
    ) {
        Ok(_) => panic!("foreign output must fail preparation"),
        Err(error) => error,
    };

    assert!(input_error.to_string().contains("different device"));
    assert!(output_error.to_string().contains("different device"));
    assert_eq!(download(&source_device, &source_output), [53.0]);
    assert_eq!(download(&other_device, &foreign_output), [59.0]);
}
