//! Strided operand types and decode snippets.

use crate::infrastructure::buffer::CudaBuffer;

/// A device buffer paired with the leto layout describing its logical view.
///
/// This device's instantiation of the device-neutral
/// [`StridedView`](hephaestus_core::StridedView), so an operand built here
/// passes straight into the generic accelerator layer without conversion.
pub type StridedOperand<'a, T, const N: usize> =
    hephaestus_core::StridedOperand<'a, CudaBuffer<T>, N>;

/// Borrowed dynamic-rank layout metadata for runtime-shaped consumers.
///
/// This is the layout-neutral counterpart to [`leto::Layout<N>`]. It lets
/// consumers pass runtime tensor layouts without materializing a fixed-rank Leto
/// layout. `shape` and `strides` must have equal length, and rank must be at
/// most [`MAX_STRIDED_RANK`](super::MAX_STRIDED_RANK).
#[derive(Clone, Copy)]
pub struct StridedLayout<'a> {
    /// Logical shape, right-aligned into the rank-eight CUDA metadata.
    pub shape: &'a [usize],
    /// Physical element strides, one per logical dimension.
    pub strides: &'a [usize],
    /// Base element offset into the storage buffer.
    pub offset: usize,
}

/// A device buffer paired with dynamic-rank layout metadata.
#[derive(Clone, Copy)]
pub struct StridedOperandDyn<'a, T> {
    /// The device buffer.
    pub buffer: &'a CudaBuffer<T>,
    /// The logical layout over that buffer.
    pub layout: StridedLayout<'a>,
}

pub(crate) const CUDA_META: &str = r#"

struct Meta {
    unsigned int shape[8];
    int a_strides[8];
    int b_strides[8];
    int out_strides[8];
    unsigned int offsets[4];
};
"#;

pub(crate) const CUDA_DECODE: &str = r#"
    unsigned int rem = i;
    int a_off = lmeta.offsets[0];
    int b_off = lmeta.offsets[1];
    int o_off = lmeta.offsets[2];
    for (int d = 7; d >= 0; d--) {
        unsigned int dim = lmeta.shape[d];
        int idx = (int)(rem % dim);
        rem = rem / dim;
        a_off += idx * lmeta.a_strides[d];
        b_off += idx * lmeta.b_strides[d];
        o_off += idx * lmeta.out_strides[d];
    }
"#;
