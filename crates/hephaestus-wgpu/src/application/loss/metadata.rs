use bytemuck::{Pod, Zeroable};
use hephaestus_core::{CrossEntropyPlan, HephaestusError, Result};
use leto::Layout;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct LayoutMeta {
    shape: [u32; 4],
    address: [i32; 4],
}

impl LayoutMeta {
    fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > 2 {
            return Err(invalid(format!(
                "cross-entropy WGSL metadata supports rank at most 2, got {R}"
            )));
        }
        let mut shape = [1_u32; 4];
        let mut address = [0_i32; 4];
        for axis in 0..R {
            shape[axis] = u32::try_from(layout.shape()[axis]).map_err(|_| {
                invalid(format!(
                    "cross-entropy extent {} on axis {axis} exceeds u32 range",
                    layout.shape()[axis]
                ))
            })?;
            address[axis] = i32::try_from(layout.strides()[axis]).map_err(|_| {
                invalid(format!(
                    "cross-entropy stride {} on axis {axis} exceeds i32 range",
                    layout.strides()[axis]
                ))
            })?;
        }
        address[2] = i32::try_from(layout.offset()).map_err(|_| {
            invalid(format!(
                "cross-entropy offset {} exceeds signed WGSL address range",
                layout.offset()
            ))
        })?;
        let (minimum, maximum) = layout.checked_min_max_offsets().map_err(|error| {
            invalid(format!(
                "cross-entropy layout address range rejected: {error}"
            ))
        })?;
        for (name, value) in [("minimum", minimum), ("maximum", maximum)] {
            i32::try_from(value).map_err(|_| {
                invalid(format!(
                    "cross-entropy {name} offset {value} exceeds signed WGSL address range"
                ))
            })?;
        }
        Ok(Self { shape, address })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct CrossEntropyMeta {
    logits: LayoutMeta,
    targets: LayoutMeta,
    loss: LayoutMeta,
    probabilities: LayoutMeta,
    output_gradient: LayoutMeta,
    logit_gradient: LayoutMeta,
    dimensions: [u32; 4],
}

impl CrossEntropyMeta {
    pub(super) fn forward(
        plan: CrossEntropyPlan,
        logits: &Layout<2>,
        targets: &Layout<1>,
        loss: &Layout<1>,
        probabilities: &Layout<2>,
    ) -> Result<Self> {
        let empty_one = Layout::c_contiguous([1]).map_err(layout_error)?;
        let empty_two = Layout::c_contiguous([1, 1]).map_err(layout_error)?;
        Self::new(
            plan,
            logits,
            targets,
            loss,
            probabilities,
            &empty_one,
            &empty_two,
        )
    }

    pub(super) fn backward(
        plan: CrossEntropyPlan,
        output_gradient: &Layout<1>,
        probabilities: &Layout<2>,
        targets: &Layout<1>,
        logit_gradient: &Layout<2>,
    ) -> Result<Self> {
        let empty_two = Layout::c_contiguous([1, 1]).map_err(layout_error)?;
        let empty_one = Layout::c_contiguous([1]).map_err(layout_error)?;
        Self::new(
            plan,
            &empty_two,
            targets,
            &empty_one,
            probabilities,
            output_gradient,
            logit_gradient,
        )
    }

    fn new(
        plan: CrossEntropyPlan,
        logits: &Layout<2>,
        targets: &Layout<1>,
        loss: &Layout<1>,
        probabilities: &Layout<2>,
        output_gradient: &Layout<1>,
        logit_gradient: &Layout<2>,
    ) -> Result<Self> {
        Ok(Self {
            logits: LayoutMeta::new(logits)?,
            targets: LayoutMeta::new(targets)?,
            loss: LayoutMeta::new(loss)?,
            probabilities: LayoutMeta::new(probabilities)?,
            output_gradient: LayoutMeta::new(output_gradient)?,
            logit_gradient: LayoutMeta::new(logit_gradient)?,
            dimensions: [
                narrow(plan.batch, "batch")?,
                narrow(plan.classes, "classes")?,
                plan.probability_tolerance.to_bits(),
                0,
            ],
        })
    }
}

fn narrow(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid(format!("cross-entropy {name} {value} exceeds u32 range")))
}

fn layout_error(error: leto::LetoError) -> HephaestusError {
    invalid(format!(
        "cross-entropy placeholder layout rejected: {error}"
    ))
}

fn invalid(message: String) -> HephaestusError {
    HephaestusError::InvalidConfiguration { message }
}
