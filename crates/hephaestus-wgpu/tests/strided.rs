//! Differential contract tests for strided-layout dispatch: device results vs
//! a CPU reference computed over the same leto layout metadata.

use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{binary_elementwise_strided_into, AddOp, MulOp, WgpuDevice};
use leto::Layout;

fn device_or_skip() -> Option<WgpuDevice> {
    match WgpuDevice::try_default("hephaestus-strided-test") {
        Ok(device) => Some(device),
        Err(e) => {
            eprintln!("skipping wgpu strided test: {e}");
            None
        }
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
    let Some(device) = device_or_skip() else {
        return;
    };
    // a is a 3x2 buffer viewed transposed as 2x3; b and out are C-contiguous 2x3.
    let a_host: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // physical [3,2]
    let a_layout = Layout::new([2, 3], [1, 2], 0); // transposed view
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

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(6).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 2>(
        &device,
        &a,
        &a_layout,
        &b,
        &b_layout,
        &out,
        &out_layout,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}

#[test]
fn strided_broadcast_inputs_match_cpu() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // [2,1] + [1,3] -> [2,3]: both inputs broadcast on the device via zero
    // strides; no materialized operands.
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

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(6).unwrap();
    binary_elementwise_strided_into::<AddOp, f32, 2>(
        &device,
        &a,
        &a_layout,
        &b,
        &b_layout,
        &out,
        &out_layout,
    )
    .unwrap();

    let mut got = vec![0.0f32; 6];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
    assert_eq!(got, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
}

#[test]
fn strided_offset_output_writes_only_selected_region() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // Write a 2x2 product into the bottom-right 2x2 block of a zeroed 3x3
    // output buffer: out layout has offset 4 (row 1, col 1) over C [3,3].
    let a_host = vec![1.0f32, 2.0, 3.0, 4.0];
    let a_layout = Layout::c_contiguous([2, 2]).unwrap();
    let b_host = vec![5.0f32, 6.0, 7.0, 8.0];
    let b_layout = Layout::c_contiguous([2, 2]).unwrap();
    // Sub-block of [3,3]: shape [2,2], strides [3,1], offset 4.
    let out_layout = Layout::new([2, 2], [3, 1], 4);

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(9).unwrap();
    binary_elementwise_strided_into::<MulOp, f32, 2>(
        &device,
        &a,
        &a_layout,
        &b,
        &b_layout,
        &out,
        &out_layout,
    )
    .unwrap();

    let mut got = vec![0.0f32; 9];
    device.download(&out, &mut got).unwrap();
    // products: [5, 12, 21, 32] at physical 4,5,7,8; everything else untouched.
    assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 0.0, 21.0, 32.0]);
}

#[test]
fn strided_rejects_aliasing_output_and_short_buffers() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let a = device.upload(&[1.0f32, 2.0]).unwrap();
    let b = device.upload(&[1.0f32, 2.0]).unwrap();
    let out = device.alloc_zeroed::<f32>(2).unwrap();

    // Zero-stride aliasing output is rejected.
    let aliasing = Layout::new([2, 2], [0, 1], 0);
    let flat = Layout::c_contiguous([2, 2]).unwrap();
    assert!(binary_elementwise_strided_into::<AddOp, f32, 2>(
        &device, &a, &flat, &b, &flat, &out, &aliasing
    )
    .is_err());

    // Layout exceeding the backing buffer is rejected before dispatch.
    let too_big = Layout::c_contiguous([4]).unwrap();
    let small = Layout::c_contiguous([2]).unwrap();
    let a1 = device.upload(&[1.0f32, 2.0]).unwrap();
    let out1 = device.alloc_zeroed::<f32>(4).unwrap();
    assert!(binary_elementwise_strided_into::<AddOp, f32, 1>(
        &device, &a1, &too_big, &a1, &small, &out1, &too_big
    )
    .is_err());
}

#[test]
fn strided_rank3_batched_matches_cpu() {
    let Some(device) = device_or_skip() else {
        return;
    };
    // Rank-3 with a transposed inner pair on `a`: [2,3,4] logical, a stored
    // as [2,4,3] and viewed with swapped inner strides.
    let a_host: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
    let a_layout = Layout::new([2, 3, 4], [12, 1, 3], 0); // inner transpose view
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

    let a = device.upload(&a_host).unwrap();
    let b = device.upload(&b_host).unwrap();
    let out = device.alloc_zeroed::<f32>(24).unwrap();
    binary_elementwise_strided_into::<MulOp, f32, 3>(
        &device,
        &a,
        &a_layout,
        &b,
        &b_layout,
        &out,
        &out_layout,
    )
    .unwrap();

    let mut got = vec![0.0f32; 24];
    device.download(&out, &mut got).unwrap();
    assert_eq!(got, expected);
}
