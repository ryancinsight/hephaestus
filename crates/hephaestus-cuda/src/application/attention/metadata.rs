use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, HephaestusError, Result,
};
use leto::Layout;

use crate::infrastructure::buffer::CudaBuffer;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LayoutMeta {
    pub(super) shape: [i64; 3],
    pub(super) strides: [i64; 3],
    pub(super) offset: i64,
}

impl LayoutMeta {
    fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > 3 {
            return Err(invalid(format!(
                "attention metadata rank {R} exceeds CUDA rank 3"
            )));
        }
        let mut shape = [1_i64; 3];
        let mut strides = [0_i64; 3];
        for axis in 0..R {
            shape[axis] = i64::try_from(layout.shape[axis]).map_err(|_| {
                invalid(format!(
                    "attention extent {} on axis {axis} exceeds i64",
                    layout.shape[axis]
                ))
            })?;
            strides[axis] = i64::try_from(layout.strides[axis]).map_err(|_| {
                invalid(format!(
                    "attention stride {} on axis {axis} exceeds i64",
                    layout.strides[axis]
                ))
            })?;
        }
        Ok(Self {
            shape,
            strides,
            offset: i64::try_from(layout.offset)
                .map_err(|_| invalid("attention layout offset exceeds i64"))?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct ForwardMeta {
    pub(super) query: LayoutMeta,
    pub(super) key: LayoutMeta,
    pub(super) value: LayoutMeta,
    pub(super) output: LayoutMeta,
    pub(super) weights: LayoutMeta,
    pub(super) keep: LayoutMeta,
    pub(super) heads_per_batch: i64,
    pub(super) causal: i32,
    pub(super) keep_present: i32,
}

impl ForwardMeta {
    pub(super) fn new<T>(
        operands: &AttentionForwardOperands<'_, CudaBuffer<T>, T>,
    ) -> Result<Self> {
        let keep = operands.mask.grouped_keep();
        Ok(Self {
            query: LayoutMeta::new(operands.query.layout)?,
            key: LayoutMeta::new(operands.key.layout)?,
            value: LayoutMeta::new(operands.value.layout)?,
            output: LayoutMeta::new(operands.output.layout)?,
            weights: LayoutMeta::new(operands.weights.layout)?,
            keep: keep.map_or_else(
                || Ok(LayoutMeta::empty()),
                |grouped| LayoutMeta::new(grouped.view().layout),
            )?,
            heads_per_batch: keep.map_or(1, |grouped| {
                i64::try_from(grouped.heads_per_batch().get())
                    .expect("invariant: attention plan restricts CUDA addresses to i32")
            }),
            causal: i32::from(matches!(
                operands.mask.causality(),
                hephaestus_core::AttentionCausality::Causal
            )),
            keep_present: i32::from(keep.is_some()),
        })
    }
}

impl LayoutMeta {
    const fn empty() -> Self {
        Self {
            shape: [1; 3],
            strides: [0; 3],
            offset: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct BackwardMeta {
    pub(super) grad_output: LayoutMeta,
    pub(super) query: LayoutMeta,
    pub(super) key: LayoutMeta,
    pub(super) value: LayoutMeta,
    pub(super) weights: LayoutMeta,
    pub(super) target: LayoutMeta,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct BackwardPreflightMeta {
    pub(super) grad_output: LayoutMeta,
    pub(super) query: LayoutMeta,
    pub(super) key: LayoutMeta,
    pub(super) value: LayoutMeta,
    pub(super) weights: LayoutMeta,
    pub(super) query_gradient: LayoutMeta,
    pub(super) key_gradient: LayoutMeta,
    pub(super) value_gradient: LayoutMeta,
    pub(super) query_selected: i32,
    pub(super) key_selected: i32,
    pub(super) value_selected: i32,
}

impl BackwardPreflightMeta {
    pub(super) fn new<T>(
        operands: &AttentionBackwardOperands<'_, CudaBuffer<T>, T>,
    ) -> Result<Self> {
        Ok(Self {
            grad_output: LayoutMeta::new(operands.grad_output.layout)?,
            query: LayoutMeta::new(operands.query.layout)?,
            key: LayoutMeta::new(operands.key.layout)?,
            value: LayoutMeta::new(operands.value.layout)?,
            weights: LayoutMeta::new(operands.weights.layout)?,
            query_gradient: operands.gradients.query.map_or_else(
                || Ok(LayoutMeta::empty()),
                |gradient| LayoutMeta::new(gradient.layout),
            )?,
            key_gradient: operands.gradients.key.map_or_else(
                || Ok(LayoutMeta::empty()),
                |gradient| LayoutMeta::new(gradient.layout),
            )?,
            value_gradient: operands.gradients.value.map_or_else(
                || Ok(LayoutMeta::empty()),
                |gradient| LayoutMeta::new(gradient.layout),
            )?,
            query_selected: i32::from(operands.gradients.query.is_some()),
            key_selected: i32::from(operands.gradients.key.is_some()),
            value_selected: i32::from(operands.gradients.value.is_some()),
        })
    }
}

impl BackwardMeta {
    pub(super) fn new<T>(
        operands: &AttentionBackwardOperands<'_, CudaBuffer<T>, T>,
        target: &Layout<3>,
    ) -> Result<Self> {
        Ok(Self {
            grad_output: LayoutMeta::new(operands.grad_output.layout)?,
            query: LayoutMeta::new(operands.query.layout)?,
            key: LayoutMeta::new(operands.key.layout)?,
            value: LayoutMeta::new(operands.value.layout)?,
            weights: LayoutMeta::new(operands.weights.layout)?,
            target: LayoutMeta::new(target)?,
        })
    }
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

const _: () = assert!(core::mem::size_of::<LayoutMeta>() == 56);
const _: () = assert!(core::mem::size_of::<ForwardMeta>() == 352);
const _: () = assert!(core::mem::size_of::<BackwardMeta>() == 336);
const _: () = assert!(core::mem::size_of::<BackwardPreflightMeta>() == 464);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_metadata_preserves_signed_strides_and_offset() {
        let layout = Layout::new([2, 3, 4], [20, -4, 1], 11);
        let metadata = LayoutMeta::new(&layout).expect("rank-3 metadata");
        assert_eq!(metadata.shape, [2, 3, 4]);
        assert_eq!(metadata.strides, [20, -4, 1]);
        assert_eq!(metadata.offset, 11);
    }

    #[test]
    fn rank_two_metadata_pads_the_unused_axis() {
        let layout = Layout::new([2, 3], [5, 1], 2);
        let metadata = LayoutMeta::new(&layout).expect("rank-2 metadata");
        assert_eq!(metadata.shape, [2, 3, 1]);
        assert_eq!(metadata.strides, [5, 1, 0]);
    }
}
