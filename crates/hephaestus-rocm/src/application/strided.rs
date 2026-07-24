//! Layout-aware operands shared by ROCm strided operator families.

use leto::Layout;

use crate::RocmBuffer;

/// A typed ROCm buffer paired with a rank-specific logical layout.
#[derive(Clone, Copy)]
pub struct StridedOperand<'a, T, const N: usize> {
    /// The device buffer containing the view's storage.
    pub buffer: &'a RocmBuffer<T>,
    /// The logical shape, strides, and offset within `buffer`.
    pub layout: &'a Layout<N>,
}
