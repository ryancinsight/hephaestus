//! Contract clauses for provider-owned runtime-rank expression fusion.
//!
//! Every GPU provider monomorphizes the same
//! [`FusedElementwiseOps`](hephaestus_core::FusedElementwiseOps) and
//! [`FusedReductionOps`](hephaestus_core::FusedReductionOps) seams over its
//! own device, scalar, and dialect. This
//! suite pins the shared value semantics — broadcast elementwise evaluation,
//! axis reduction, signed strides, empty-axis identities, and the rejection
//! clauses — so backend choice never changes which expressions fuse or what
//! they compute.

use eunomia::Pod;
use hephaestus_core::{
    ComputeDevice, DynamicStridedView, FusedElementwiseOps, FusedExpression, FusedReduction,
    FusedReductionOps, HephaestusError, KernelDialect,
};
use leto::LayoutDyn;

fn layout(shape: &[usize], strides: &[isize], offset: usize) -> LayoutDyn {
    LayoutDyn::new(
        shape.to_vec().into_boxed_slice(),
        strides.to_vec().into_boxed_slice(),
        offset,
    )
    .expect("fusion test layout ranks match")
}

fn lift<T: From<f32>>(values: &[f32]) -> Vec<T> {
    values.iter().map(|&value| T::from(value)).collect()
}

fn download<D: ComputeDevice, T: Pod + Copy + From<f32>>(
    device: &D,
    buffer: &D::Buffer<T>,
    len: usize,
) -> Vec<T> {
    let mut values = vec![T::from(0.0); len];
    device
        .download(buffer, &mut values)
        .expect("fusion download");
    values
}

/// Assert the shared runtime-rank fusion contract for one provider.
///
/// `add` evaluates `input_0 + input_1` and `input` evaluates `input_0` in the
/// provider's dialect; both fragments are valid WGSL, CUDA C++, and HIP C++,
/// so every backend instantiates this suite with the same expression text.
///
/// # Panics
///
/// Panics with the backend and violated broadcast, reduction, striding,
/// empty-axis, or rejection clause when the provider diverges from the
/// shared contract.
pub fn assert_fusion_contract<D, O, L, T, A, I>(device: &D, operations: &O, add: &A, input: &I)
where
    D: ComputeDevice,
    L: KernelDialect,
    O: FusedElementwiseOps<D, T, Dialect = L> + FusedReductionOps<D, T, Dialect = L>,
    T: Pod + Copy + PartialEq + core::fmt::Debug + From<f32>,
    A: FusedExpression<L>,
    I: FusedExpression<L>,
{
    assert_broadcast_elementwise(device, operations, add);
    assert_reduction(device, operations, input);
    assert_signed_strides(device, operations, input);
    assert_empty_axis(device, operations, input);
    assert_rejections(device, operations, input);
}

fn assert_broadcast_elementwise<D, O, L, T, A>(device: &D, operations: &O, add: &A)
where
    D: ComputeDevice,
    L: KernelDialect,
    O: FusedElementwiseOps<D, T, Dialect = L>,
    T: Pod + Copy + PartialEq + core::fmt::Debug + From<f32>,
    A: FusedExpression<L>,
{
    let first = device.upload(&lift::<T>(&[1.0, 2.0])).expect("first input");
    let second = device
        .upload(&lift::<T>(&[10.0, 20.0, 30.0]))
        .expect("second input");
    let output = device.alloc_zeroed::<T>(6).expect("output allocation");
    let first_layout = layout(&[2, 1], &[1, 1], 0);
    let second_layout = layout(&[1, 3], &[3, 1], 0);
    let output_layout = layout(&[2, 3], &[3, 1], 0);

    operations
        .fused_elementwise_into(
            device,
            add,
            &[
                DynamicStridedView::new(&first, &first_layout),
                DynamicStridedView::new(&second, &second_layout),
            ],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("runtime-rank elementwise dispatch");

    assert_eq!(
        download(device, &output, 6),
        lift::<T>(&[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]),
    );
}

fn assert_reduction<D, O, L, T, I>(device: &D, operations: &O, input: &I)
where
    D: ComputeDevice,
    L: KernelDialect,
    O: FusedReductionOps<D, T, Dialect = L>,
    T: Pod + Copy + PartialEq + core::fmt::Debug + From<f32>,
    I: FusedExpression<L>,
{
    let values = device
        .upload(&lift::<T>(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        .expect("reduction input");
    let output = device.alloc_zeroed::<T>(2).expect("reduction output");
    let input_layout = layout(&[2, 3], &[3, 1], 0);
    let output_layout = layout(&[2, 1], &[1, 1], 0);

    operations
        .fused_reduce_into(
            device,
            input,
            &[DynamicStridedView::new(&values, &input_layout)],
            FusedReduction::Sum,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("runtime-rank reduction dispatch");

    assert_eq!(download(device, &output, 2), lift::<T>(&[6.0, 15.0]));
}

fn assert_signed_strides<D, O, L, T, I>(device: &D, operations: &O, input: &I)
where
    D: ComputeDevice,
    L: KernelDialect,
    O: FusedElementwiseOps<D, T, Dialect = L>,
    T: Pod + Copy + PartialEq + core::fmt::Debug + From<f32>,
    I: FusedExpression<L>,
{
    let values = device.upload(&lift::<T>(&[3.0, 2.0, 1.0])).expect("input");
    let output = device.alloc_zeroed::<T>(3).expect("output");
    let input_layout = layout(&[3], &[-1], 2);
    let output_layout = layout(&[3], &[1], 0);

    operations
        .fused_elementwise_into(
            device,
            input,
            &[DynamicStridedView::new(&values, &input_layout)],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("signed-stride elementwise dispatch");

    assert_eq!(download(device, &output, 3), lift::<T>(&[1.0, 2.0, 3.0]));
}

fn assert_empty_axis<D, O, L, T, I>(device: &D, operations: &O, input: &I)
where
    D: ComputeDevice,
    L: KernelDialect,
    O: FusedReductionOps<D, T, Dialect = L>,
    T: Pod + Copy + PartialEq + core::fmt::Debug + From<f32>,
    I: FusedExpression<L>,
{
    let values = device
        .alloc_uninitialized::<T>(0)
        .expect("empty reduction input");
    let output = device
        .upload(&lift::<T>(&[7.0, 11.0]))
        .expect("reduction output");
    let input_layout = layout(&[2, 0], &[0, 1], 0);
    let output_layout = layout(&[2, 1], &[1, 1], 0);

    operations
        .fused_reduce_into(
            device,
            input,
            &[DynamicStridedView::new(&values, &input_layout)],
            FusedReduction::Sum,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect("empty sum identity dispatch");
    assert_eq!(download(device, &output, 2), lift::<T>(&[0.0, 0.0]));

    let error = operations
        .fused_reduce_into(
            device,
            input,
            &[DynamicStridedView::new(&values, &input_layout)],
            FusedReduction::Mean,
            1,
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect_err("empty mean must be rejected");
    assert!(
        matches!(error, HephaestusError::InvalidConfiguration { .. }),
        "unexpected error: {error}"
    );
    assert_eq!(download(device, &output, 2), lift::<T>(&[0.0, 0.0]));
}

fn assert_rejections<D, O, L, T, I>(device: &D, operations: &O, input: &I)
where
    D: ComputeDevice,
    L: KernelDialect,
    O: FusedElementwiseOps<D, T, Dialect = L>,
    T: Pod + Copy + PartialEq + core::fmt::Debug + From<f32>,
    I: FusedExpression<L>,
{
    let values = device.upload(&lift::<T>(&[1.0, 2.0])).expect("input");
    let output = device.upload(&lift::<T>(&[7.0])).expect("output");
    let input_layout = layout(&[2], &[1], 0);
    let output_layout = layout(&[2], &[0], 0);

    let error = operations
        .fused_elementwise_into(
            device,
            input,
            &[DynamicStridedView::new(&values, &input_layout)],
            DynamicStridedView::new(&output, &output_layout),
        )
        .expect_err("non-injective fusion output must be rejected");
    assert!(
        matches!(error, HephaestusError::DispatchFailed { .. }),
        "unexpected error: {error}"
    );
    assert_eq!(download(device, &output, 1), lift::<T>(&[7.0]));
}
