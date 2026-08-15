use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyPlan, HephaestusError,
    Result,
};
use leto::Layout;

use crate::RocmBuffer;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Layout1Meta {
    shape: i32,
    stride: i32,
    offset: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Layout2Meta {
    rows: i32,
    columns: i32,
    row_stride: i32,
    column_stride: i32,
    offset: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CrossEntropyMeta {
    logits: Layout2Meta,
    targets: Layout1Meta,
    loss: Layout1Meta,
    probabilities: Layout2Meta,
    output_gradient: Layout1Meta,
    logit_gradient: Layout2Meta,
    batch: i32,
    classes: i32,
    probability_tolerance: f32,
}

impl CrossEntropyMeta {
    pub(super) fn forward(
        operands: &CrossEntropyForwardOperands<'_, RocmBuffer<f32>, RocmBuffer<u32>>,
        plan: CrossEntropyPlan,
    ) -> Result<Self> {
        Ok(Self {
            logits: Layout2Meta::new(operands.logits.layout)?,
            targets: Layout1Meta::new(operands.targets.layout)?,
            loss: Layout1Meta::new(operands.loss.layout)?,
            probabilities: Layout2Meta::new(operands.probabilities.layout)?,
            output_gradient: Layout1Meta::empty(),
            logit_gradient: Layout2Meta::empty(),
            batch: narrow(plan.batch, "batch")?,
            classes: narrow(plan.classes, "class count")?,
            probability_tolerance: plan.probability_tolerance,
        })
    }

    pub(super) fn backward(
        operands: &CrossEntropyBackwardOperands<'_, RocmBuffer<f32>, RocmBuffer<u32>>,
        plan: CrossEntropyPlan,
    ) -> Result<Self> {
        Ok(Self {
            logits: Layout2Meta::empty(),
            targets: Layout1Meta::new(operands.targets.layout)?,
            loss: Layout1Meta::empty(),
            probabilities: Layout2Meta::new(operands.probabilities.layout)?,
            output_gradient: Layout1Meta::new(operands.output_gradient.layout)?,
            logit_gradient: Layout2Meta::new(operands.logit_gradient.layout)?,
            batch: narrow(plan.batch, "batch")?,
            classes: narrow(plan.classes, "class count")?,
            probability_tolerance: plan.probability_tolerance,
        })
    }
}

impl Layout1Meta {
    fn new(layout: &Layout<1>) -> Result<Self> {
        Ok(Self {
            shape: narrow(layout.shape()[0], "rank-1 extent")?,
            stride: narrow_stride(layout.strides()[0], "rank-1 stride")?,
            offset: narrow(layout.offset(), "rank-1 offset")?,
        })
    }

    const fn empty() -> Self {
        Self {
            shape: 0,
            stride: 0,
            offset: 0,
        }
    }
}

impl Layout2Meta {
    fn new(layout: &Layout<2>) -> Result<Self> {
        Ok(Self {
            rows: narrow(layout.shape()[0], "row extent")?,
            columns: narrow(layout.shape()[1], "column extent")?,
            row_stride: narrow_stride(layout.strides()[0], "row stride")?,
            column_stride: narrow_stride(layout.strides()[1], "column stride")?,
            offset: narrow(layout.offset(), "rank-2 offset")?,
        })
    }

    const fn empty() -> Self {
        Self {
            rows: 0,
            columns: 0,
            row_stride: 0,
            column_stride: 0,
            offset: 0,
        }
    }
}

fn narrow(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        invalid(format!(
            "cross-entropy {name} {value} exceeds signed ROCm address range"
        ))
    })
}

fn narrow_stride(value: isize, name: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        invalid(format!(
            "cross-entropy {name} {value} exceeds signed ROCm address range"
        ))
    })
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_preserves_permuted_and_negative_strides() {
        let permuted = Layout::try_new([2, 3], [1, 2], 0).expect("valid test layout");
        let reversed = Layout::try_new([2, 3], [3, -1], 2).expect("valid test layout");

        let permuted = Layout2Meta::new(&permuted).expect("permuted layout metadata");
        let reversed = Layout2Meta::new(&reversed).expect("reversed layout metadata");

        assert_eq!(permuted.row_stride, 1);
        assert_eq!(permuted.column_stride, 2);
        assert_eq!(reversed.row_stride, 3);
        assert_eq!(reversed.column_stride, -1);
        assert_eq!(reversed.offset, 2);
    }
}
