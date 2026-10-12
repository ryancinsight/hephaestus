use hephaestus_core::HephaestusError;
use hephaestus_cuda::{CudaBuffer, CudaDevice, StridedLayout, StridedOperand, StridedOperandDyn};
use leto::Layout;

pub(crate) fn op<'a, T, const N: usize>(
    buffer: &'a CudaBuffer<T>,
    layout: &'a Layout<N>,
) -> StridedOperand<'a, T, N> {
    StridedOperand { buffer, layout }
}

pub(crate) fn dyn_op<'a, T>(
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

pub(crate) fn device(test: &str) -> Option<CudaDevice> {
    match CudaDevice::try_default() {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skip {test}: CUDA device unavailable ({e})");
            None
        }
    }
}

pub(crate) fn assert_dispatch_message<T>(
    result: hephaestus_cuda::Result<T>,
    expected: &'static str,
) {
    match result {
        Err(HephaestusError::DispatchFailed { message }) => assert_eq!(message, expected),
        Err(error) => panic!("expected dispatch failure {expected:?}, got {error:?}"),
        Ok(_) => panic!("expected dispatch failure {expected:?}, got success"),
    }
}

/// CPU reference: out[idx] = op(a[idx], b[idx]) by logical row-major index
/// over the given layouts (a/b already broadcast-compatible with out shape).
pub(crate) fn cpu_reference<const N: usize>(
    a: &[f32],
    a_layout: &Layout<N>,
    b: &[f32],
    b_layout: &Layout<N>,
    out: &mut [f32],
    out_layout: &Layout<N>,
    op: impl Fn(f32, f32) -> f32,
) {
    let a_l = a_layout.broadcast(out_layout.shape()).unwrap();
    let b_l = b_layout.broadcast(out_layout.shape()).unwrap();
    let shape = out_layout.shape();
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
