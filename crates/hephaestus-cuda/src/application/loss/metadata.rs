use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, HephaestusError, Result,
};
use leto::Layout;

use crate::infrastructure::buffer::CudaBuffer;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LayoutMeta {
    shape: [i64; 2],
    strides: [i64; 2],
    offset: i64,
}

impl LayoutMeta {
    fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > 2 {
            return Err(invalid(format!(
                "cross-entropy metadata rank {R} exceeds CUDA rank 2"
            )));
        }
        let mut shape = [1_i64; 2];
        let mut strides = [0_i64; 2];
        for axis in 0..R {
            shape[axis] = i64::try_from(layout.shape[axis])
                .map_err(|_| invalid("cross-entropy extent exceeds i64"))?;
            strides[axis] = i64::try_from(layout.strides[axis])
                .map_err(|_| invalid("cross-entropy stride exceeds i64"))?;
        }
        Ok(Self {
            shape,
            strides,
            offset: i64::try_from(layout.offset)
                .map_err(|_| invalid("cross-entropy offset exceeds i64"))?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct ForwardMeta {
    logits: LayoutMeta,
    targets: LayoutMeta,
    loss: LayoutMeta,
    probabilities: LayoutMeta,
}

impl ForwardMeta {
    pub(super) fn new(
        operands: &CrossEntropyForwardOperands<'_, CudaBuffer<f32>, CudaBuffer<u32>>,
    ) -> Result<Self> {
        Ok(Self {
            logits: LayoutMeta::new(operands.logits.layout)?,
            targets: LayoutMeta::new(operands.targets.layout)?,
            loss: LayoutMeta::new(operands.loss.layout)?,
            probabilities: LayoutMeta::new(operands.probabilities.layout)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct BackwardMeta {
    output_gradient: LayoutMeta,
    probabilities: LayoutMeta,
    targets: LayoutMeta,
    logit_gradient: LayoutMeta,
    tolerance: f32,
}

impl BackwardMeta {
    pub(super) fn new(
        operands: &CrossEntropyBackwardOperands<'_, CudaBuffer<f32>, CudaBuffer<u32>>,
        tolerance: f32,
    ) -> Result<Self> {
        Ok(Self {
            output_gradient: LayoutMeta::new(operands.output_gradient.layout)?,
            probabilities: LayoutMeta::new(operands.probabilities.layout)?,
            targets: LayoutMeta::new(operands.targets.layout)?,
            logit_gradient: LayoutMeta::new(operands.logit_gradient.layout)?,
            tolerance,
        })
    }
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}
