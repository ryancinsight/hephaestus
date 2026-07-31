use hephaestus_core::{
    AttentionBackwardOperands, AttentionForwardOperands, AttentionPlan, HephaestusError, Result,
};
use leto::Layout;

use crate::RocmBuffer;

const TENSOR_RANK: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LayoutMeta {
    shape: [i32; TENSOR_RANK],
    strides: [i32; TENSOR_RANK],
    offset: i32,
    rank: i32,
}

impl LayoutMeta {
    fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > TENSOR_RANK {
            return Err(invalid(format!(
                "attention layout rank {R} exceeds ROCm metadata rank {TENSOR_RANK}"
            )));
        }
        let (minimum, maximum) = layout.checked_min_max_offsets().map_err(|error| {
            invalid(format!("attention layout address range rejected: {error}"))
        })?;
        for (name, value) in [("minimum", minimum), ("maximum", maximum)] {
            i32::try_from(value).map_err(|_| {
                invalid(format!(
                    "attention layout {name} offset {value} exceeds signed ROCm address range"
                ))
            })?;
        }

        let mut shape = [0; TENSOR_RANK];
        let mut strides = [0; TENSOR_RANK];
        for axis in 0..R {
            shape[axis] = i32::try_from(layout.shape[axis]).map_err(|_| {
                invalid(format!(
                    "attention extent {} on axis {axis} exceeds i32 range",
                    layout.shape[axis]
                ))
            })?;
            strides[axis] = i32::try_from(layout.strides[axis]).map_err(|_| {
                invalid(format!(
                    "attention stride {} on axis {axis} exceeds i32 range",
                    layout.strides[axis]
                ))
            })?;
        }
        Ok(Self {
            shape,
            strides,
            offset: i32::try_from(layout.offset).map_err(|_| {
                invalid(format!(
                    "attention layout offset {} exceeds signed ROCm address range",
                    layout.offset
                ))
            })?,
            rank: i32::try_from(R).expect("invariant: attention rank fits i32"),
        })
    }

    const fn empty() -> Self {
        Self {
            shape: [0; TENSOR_RANK],
            strides: [0; TENSOR_RANK],
            offset: 0,
            rank: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct AttentionMeta {
    query: LayoutMeta,
    key: LayoutMeta,
    value: LayoutMeta,
    output: LayoutMeta,
    weights: LayoutMeta,
    grad_output: LayoutMeta,
    destination: LayoutMeta,
    keep: LayoutMeta,
    heads_per_batch: i32,
    causal: i32,
    has_keep: i32,
}

impl AttentionMeta {
    pub(super) fn forward<T>(
        operands: &AttentionForwardOperands<'_, RocmBuffer<T>, T>,
    ) -> Result<Self> {
        let keep = operands.mask.grouped_keep();
        let heads_per_batch = keep.map_or(Ok(1), |mask| {
            i32::try_from(mask.heads_per_batch().get()).map_err(|_| {
                invalid(format!(
                    "attention mask group width {} exceeds signed ROCm range",
                    mask.heads_per_batch()
                ))
            })
        })?;
        Ok(Self {
            query: LayoutMeta::new(operands.query.layout)?,
            key: LayoutMeta::new(operands.key.layout)?,
            value: LayoutMeta::new(operands.value.layout)?,
            output: LayoutMeta::new(operands.output.layout)?,
            weights: LayoutMeta::new(operands.weights.layout)?,
            grad_output: LayoutMeta::empty(),
            destination: LayoutMeta::empty(),
            keep: keep.map_or_else(
                || Ok(LayoutMeta::empty()),
                |mask| LayoutMeta::new(mask.view().layout),
            )?,
            heads_per_batch,
            causal: i32::from(matches!(
                operands.mask.causality(),
                hephaestus_core::AttentionCausality::Causal
            )),
            has_keep: i32::from(keep.is_some()),
        })
    }

    pub(super) fn backward<T>(
        operands: &AttentionBackwardOperands<'_, RocmBuffer<T>, T>,
    ) -> Result<Self> {
        Ok(Self {
            query: LayoutMeta::new(operands.query.layout)?,
            key: LayoutMeta::new(operands.key.layout)?,
            value: LayoutMeta::new(operands.value.layout)?,
            output: LayoutMeta::empty(),
            weights: LayoutMeta::new(operands.weights.layout)?,
            grad_output: LayoutMeta::new(operands.grad_output.layout)?,
            destination: LayoutMeta::empty(),
            keep: LayoutMeta::empty(),
            heads_per_batch: 1,
            causal: 0,
            has_keep: 0,
        })
    }

    pub(super) fn with_destination<const R: usize>(mut self, layout: &Layout<R>) -> Result<Self> {
        self.destination = LayoutMeta::new(layout)?;
        Ok(self)
    }
}

pub(super) fn validate_plan(plan: AttentionPlan) -> Result<()> {
    plan.validate_address_limit(i32::MAX as usize)?;
    for (name, value) in [
        ("batch", plan.batch),
        ("query sequence", plan.query_sequence),
        ("key sequence", plan.key_sequence),
        ("key feature", plan.key_feature),
        ("value feature", plan.value_feature),
        ("score elements", plan.score_elements),
    ] {
        i32::try_from(value).map_err(|_| {
            invalid(format!(
                "attention {name} extent {value} exceeds signed ROCm range"
            ))
        })?;
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

const _: () = assert!(core::mem::size_of::<LayoutMeta>() == 32);
const _: () = assert!(core::mem::size_of::<AttentionMeta>() == 268);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_metadata_preserves_negative_strides_and_offsets() {
        let layout = Layout::new([2, 3, 4], [20, -4, 1], 8);
        let metadata = LayoutMeta::new(&layout).expect("valid reversed attention view");

        assert_eq!(metadata.shape, [2, 3, 4]);
        assert_eq!(metadata.strides, [20, -4, 1]);
        assert_eq!(metadata.offset, 8);
        assert_eq!(metadata.rank, 3);
    }
}
