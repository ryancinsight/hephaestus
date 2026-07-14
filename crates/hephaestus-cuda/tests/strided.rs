//! Differential contract tests for CUDA strided-layout dispatch: device results vs
//! a CPU reference computed over the same leto layout metadata.

use hephaestus_core::{BlockWidth, ComputeDevice, HephaestusError};
use hephaestus_cuda::{
    AddOp, CudaBuffer, CudaDevice, MulOp, NegOp, SqrtOp, StridedLayout, StridedOperand,
    StridedOperandDyn, binary_elementwise_strided_dyn_into, binary_elementwise_strided_into,
    scalar_elementwise_strided_into, unary_elementwise_strided_dyn_into,
    unary_elementwise_strided_into,
};
use leto::Layout;

fn op<'a, T, const N: usize>(
    buffer: &'a CudaBuffer<T>,
    layout: &'a Layout<N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand { buffer, layout }
}

fn dyn_op<'a, T>(
    buffer: &'a CudaBuffer<T>,
    shape: &'a [usize],
    strides: &'a [usize],
    offset: usize,
) -> StridedOperandDyn<'a, T> {
    StridedOperandDyn {
        buffer,
        layout: StridedLayout {
            shape,
            strides,
            offset,
        },
    }
}

fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skip {test}: CUDA device unavailable ({e})");
            None
        }
    }
}

fn assert_dispatch_message<T>(result: hephaestus_cuda::Result<T>, expected: &'static str) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => assert_eq!(message, expected),
        Err(error) => panic!("expected dispatch failure {expected:?}, got {error:?}"),
        Ok(_) => panic!("expected dispatch failure {expected:?}, got success"),
    }
}

/// CPU reference: out[idx] = op(a[idx], b[idx]) by logical row-major index
/// over the given layouts (a/b already broadcast-compatible with out shape).
fn cpu_reference<const N: usize>(
    a: &[f32],
    a_layout: &Layout<N>,
    b: &[f32],
    b_layout: &Layout<N>,
    out: &mut [f32],
    out_layout: &Layout<N>,
    op: impl Fn(f32, f32) -> f32,
) {
    let a_l = a_layout.broadcast(out_layout.shape).unwrap();
    let b_l = b_layout.broadcast(out_layout.shape).unwrap();
    let shape = out_layout.shape;
    let size: usize = shape.iter().product();
    for flat in 0..size {
        let mut index = [0usize; N];
        let mut rem = flat;
        for d in (0..N).rev() {
            index[d] = rem % shape[d];
            rem /= shape[d];
        }
        let ao = a_l.offset_of(index).unwrap();
        let bo = b_l.offset_of(index).unwrap();
        let oo = out_layout.offset_of(index).unwrap();
        out[oo] = op(a[ao], b[bo]);
    }
}

#[test]
fn strided_add_transposed_input_matches_cpu() {
    let Some(dev) = device("strided_add_transposed_input_matches_cpu") else {
        return;
    };
    let a_host: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a_layout = Layout::new([2, 3], [1, 2], 0);
    let b_host: Vec<f32> = (0..6).map(|i| 10.0 * i as f32).collect();
    let b_layout = Layout::c_contiguous([2, 3]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let mut expected = vec![0.0f32; 6];
    cpu_reference(
        &a_host,
        &a_layout,
        &b_host,
        &b_layout,
        &mut expected,
        &out_layout,
        |x, y| x + y,
    );

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn strided_broadcast_inputs_match_cpu() {
    let Some(dev) = device("strided_broadcast_inputs_match_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 2.0];
    let a_layout = Layout::c_contiguous([2, 1]).unwrap();
    let b_host = vec![10.0f32, 20.0, 30.0];
    let b_layout = Layout::c_contiguous([1, 3]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let mut expected = vec![0.0f32; 6];
    cpu_reference(
        &a_host,
        &a_layout,
        &b_host,
        &b_layout,
        &mut expected,
        &out_layout,
        |x, y| x + y,
    );

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
    assert_eq!(got, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
}

#[test]
fn dynamic_strided_broadcast_inputs_match_cpu() {
    let Some(dev) = device("dynamic_strided_broadcast_inputs_match_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 2.0];
    let b_host = vec![10.0f32, 20.0, 30.0];

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();

    binary_elementwise_strided_dyn_into::<AddOp, f32>(
        &dev,
        dyn_op(&a, &[2, 1], &[1, 1], 0),
        dyn_op(&b, &[1, 3], &[3, 1], 0),
        dyn_op(&out, &[2, 3], &[3, 1], 0),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
}

#[test]
fn strided_offset_output_writes_only_selected_region() {
    let Some(dev) = device("strided_offset_output_writes_only_selected_region") else {
        return;
    };
    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let a_layout = Layout::c_contiguous([2, 2]).unwrap();
    let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
    let b_layout = Layout::c_contiguous([2, 2]).unwrap();
    let out_layout = Layout::new([2, 2], [3, 1], 4);

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(9).unwrap();
    binary_elementwise_strided_into::<MulOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 9];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 0.0, 21.0, 32.0]);
}

#[test]
fn strided_rejects_aliasing_output_and_short_buffers() {
    let Some(dev) = device("strided_rejects_aliasing_output_and_short_buffers") else {
        return;
    };
    let a = dev.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let b = dev.upload(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let out = dev.alloc_zeroed::<f32>(4).unwrap();

    let aliasing = Layout::new([2, 2], [0, 1], 0);
    let flat = Layout::c_contiguous([2, 2]).unwrap();
    assert_dispatch_message(
        binary_elementwise_strided_into::<AddOp, f32, 2>(
            &dev,
            op(&a, &flat),
            op(&b, &flat),
            op(&out, &aliasing),
            BlockWidth::DEFAULT,
        ),
        "output layout must not contain zero-stride aliasing",
    );
}

#[test]
fn strided_rank3_batched_matches_cpu() {
    let Some(dev) = device("strided_rank3_batched_matches_cpu") else {
        return;
    };
    let a_host: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let a_layout = Layout::new([2, 3, 4], [12, 1, 3], 0);
    let b_host: Vec<f32> = (0..24).map(|i| 100.0 - i as f32).collect();
    let b_layout = Layout::c_contiguous([2, 3, 4]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3, 4]).unwrap();

    let mut expected = vec![0.0f32; 24];
    cpu_reference(
        &a_host,
        &a_layout,
        &b_host,
        &b_layout,
        &mut expected,
        &out_layout,
        |x, y| x * y,
    );

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(24).unwrap();
    binary_elementwise_strided_into::<MulOp, f32, 3>(
        &dev,
        op(&a, &a_layout),
        op(&b, &b_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 24];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn strided_unary_transposed_matches_cpu() {
    let Some(dev) = device("strided_unary_transposed_matches_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 9.0, 25.0, 4.0, 16.0, 36.0];
    let a_layout = Layout::new([2, 3], [1, 2], 0);
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    unary_elementwise_strided_into::<SqrtOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![1.0, 5.0, 4.0, 3.0, 2.0, 6.0]);
}

#[test]
fn strided_unary_broadcasts_input_to_output_shape() {
    let Some(dev) = device("strided_unary_broadcasts_input_to_output_shape") else {
        return;
    };
    let a_host = vec![1.0f32, -2.0, 3.0];
    let a_layout = Layout::c_contiguous([1, 3]).unwrap();
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    unary_elementwise_strided_into::<NegOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![-1.0, 2.0, -3.0, -1.0, 2.0, -3.0]);
}

#[test]
fn dynamic_strided_unary_transposed_matches_cpu() {
    let Some(dev) = device("dynamic_strided_unary_transposed_matches_cpu") else {
        return;
    };
    let a_host = vec![1.0f32, 9.0, 25.0, 4.0, 16.0, 36.0];

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    unary_elementwise_strided_dyn_into::<SqrtOp, f32>(
        &dev,
        dyn_op(&a, &[2, 3], &[1, 2], 0),
        dyn_op(&out, &[2, 3], &[3, 1], 0),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![1.0, 5.0, 4.0, 3.0, 2.0, 6.0]);
}

#[test]
fn strided_scalar_matches_binary_broadcast_semantics() {
    let Some(dev) = device("strided_scalar_matches_binary_broadcast_semantics") else {
        return;
    };
    let a_host = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    let a_layout = Layout::new([2, 3], [1, 2], 0);
    let out_layout = Layout::c_contiguous([2, 3]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let out = dev.alloc_zeroed::<f32>(6).unwrap();
    scalar_elementwise_strided_into::<AddOp, f32, 2>(
        &dev,
        op(&a, &a_layout),
        100.0,
        op(&out, &out_layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    dev.download(&out, &mut got).unwrap();
    assert_eq!(got, vec![101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
}

#[test]
fn non_default_block_width_produces_identical_results() {
    let Some(dev) = device("non_default_block_width_produces_identical_results") else {
        return;
    };
    let len = 1027usize;
    let a_host: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
    let b_host: Vec<f32> = (0..len).map(|i| 1000.0 - i as f32).collect();
    let layout = Layout::c_contiguous([len]).unwrap();

    let a = dev.upload(&a_host).unwrap();
    let b = dev.upload(&b_host).unwrap();

    let narrow = BlockWidth::new(128).unwrap();
    let out_narrow = dev.alloc_zeroed::<f32>(len).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 1>(
        &dev,
        op(&a, &layout),
        op(&b, &layout),
        op(&out_narrow, &layout),
        narrow,
    )
    .unwrap();

    let out_default = dev.alloc_zeroed::<f32>(len).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 1>(
        &dev,
        op(&a, &layout),
        op(&b, &layout),
        op(&out_default, &layout),
        BlockWidth::DEFAULT,
    )
    .unwrap();

    let mut got_narrow = vec![0.0f32; len];
    let mut got_default = vec![0.0f32; len];
    dev.download(&out_narrow, &mut got_narrow).unwrap();
    dev.download(&out_default, &mut got_default).unwrap();
    let expected: Vec<f32> = a_host.iter().zip(&b_host).map(|(x, y)| x + y).collect();
    assert_eq!(got_narrow, expected);
    assert_eq!(got_default, expected);
}
