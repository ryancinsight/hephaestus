//! Device-neutral strided views over backend buffers.
//!
//! Every backend already pairs a device buffer with a [`leto::Layout`] to
//! describe a strided operand, but each declares its own operand struct over its
//! own buffer type. A consumer holding such an operand is therefore bound to one
//! device API, which is the same coupling [`crate::SparseOperatorOps`] and
//! [`crate::DenseVectorOps`] removed for their families.
//!
//! [`crate::StridedView`] is that pairing with the buffer type left open, so an
//! accelerator seam can accept a strided operand without naming a backend. It is
//! deliberately a plain borrowed pair: the layout carries the shape, strides, and
//! offset, and the view adds no invariant of its own beyond holding both for the
//! same borrow. Validation stays with the operation that consumes it, because
//! what counts as a legal layout depends on the operation, not on the pairing.

use leto::{Layout, LayoutDyn};

/// A device buffer of type `B` interpreted through an `N`-dimensional layout.
///
/// `B` is the backend's buffer handle — typically `D::Buffer<T>` for a
/// [`ComputeDevice`](crate::ComputeDevice) `D` — so one seam signature serves
/// every backend without a `dyn` boundary or a per-device operand type.
///
/// The view borrows both parts, so it never extends the lifetime of the buffer
/// or the layout and costs nothing beyond the two references it holds.
#[derive(Debug)]
pub struct StridedView<'a, B, const N: usize> {
    /// The device-resident buffer supplying elements.
    pub buffer: &'a B,
    /// Shape, strides, and offset applied to `buffer`.
    pub layout: &'a Layout<N>,
}

impl<'a, B, const N: usize> StridedView<'a, B, N> {
    /// Pair a buffer with the layout through which it is read.
    #[must_use]
    #[inline]
    pub const fn new(buffer: &'a B, layout: &'a Layout<N>) -> Self {
        Self { buffer, layout }
    }
}

// `Clone`/`Copy` are implemented by hand rather than derived: a derive would
// bound them on `B: Clone`/`B: Copy`, but a view only ever copies its two
// references, so the bound would be wrong and would exclude every backend
// buffer type.
impl<B, const N: usize> Clone for StridedView<'_, B, N> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<B, const N: usize> Copy for StridedView<'_, B, N> {}

/// A device buffer of type `B` interpreted through a runtime-rank layout.
///
/// This is the boundary carrier for expression graphs and other operations
/// whose rank is data-dependent. It is a pair of borrowed references, so it
/// never copies storage or allocates while crossing the provider seam.
#[derive(Debug)]
pub struct DynamicStridedView<'a, B> {
    /// The device-resident buffer supplying elements.
    pub buffer: &'a B,
    /// Runtime-rank shape, strides, and offset applied to `buffer`.
    pub layout: &'a LayoutDyn,
}

impl<'a, B> DynamicStridedView<'a, B> {
    /// Pair a buffer with its runtime-rank layout.
    #[must_use]
    #[inline]
    pub const fn new(buffer: &'a B, layout: &'a LayoutDyn) -> Self {
        Self { buffer, layout }
    }
}

impl<B> Clone for DynamicStridedView<'_, B> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for DynamicStridedView<'_, B> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_is_copy_regardless_of_buffer_type() {
        // A deliberately non-Copy, non-Clone stand-in for a backend buffer.
        struct OpaqueBuffer(#[allow(dead_code)] Vec<u8>);

        let buffer = OpaqueBuffer(vec![1, 2, 3]);
        let layout = Layout::c_contiguous([3]).expect("rank-1 layout");
        let view = StridedView::new(&buffer, &layout);

        // Copying the view must not require anything of the buffer type.
        let copied = view;
        assert_eq!(copied.layout.shape(), view.layout.shape());
        assert!(core::ptr::eq(copied.buffer, view.buffer));
    }

    #[test]
    fn view_preserves_layout_metadata() {
        let buffer = [0u8; 6];
        let transposed = Layout::try_new([3, 2], [1, 3], 0).expect("valid test layout");
        let view = StridedView::new(&buffer, &transposed);

        assert_eq!(view.layout.shape(), [3, 2]);
        assert_eq!(view.layout.strides(), [1, 3]);
    }

    #[test]
    fn dynamic_view_is_copy_regardless_of_buffer_type() {
        struct OpaqueBuffer(Vec<u8>);

        let buffer = OpaqueBuffer(vec![1, 2, 3]);
        let layout = LayoutDyn::new(Box::from([3usize]), Box::from([1isize]), 0)
            .expect("rank-one dynamic layout");
        let view = DynamicStridedView::new(&buffer, &layout);
        let copied = view;

        assert_eq!(copied.layout.shape.as_ref(), &[3]);
        assert!(core::ptr::eq(copied.buffer, view.buffer));
        assert_eq!(buffer.0, [1, 2, 3]);
    }
}
