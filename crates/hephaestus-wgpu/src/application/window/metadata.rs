use bytemuck::{Pod, Zeroable};
use hephaestus_core::{HephaestusError, Result, WindowPlan};
use leto::Layout;

const MAX_RANK: usize = 5;
const VECTOR_WIDTH: usize = 4;

/// WGSL-compatible metadata for one strided layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct WindowLayoutMeta {
    pub(super) shape: [[u32; 4]; 2],
    pub(super) strides: [[i32; 4]; 2],
    pub(super) offset_and_rank: [u32; VECTOR_WIDTH],
}

impl WindowLayoutMeta {
    pub(super) fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > MAX_RANK {
            return Err(invalid(format!(
                "window rank {R} exceeds WGPU metadata rank {MAX_RANK}"
            )));
        }
        let offset = u32::try_from(layout.offset()).map_err(|_| {
            invalid(format!(
                "window offset {} exceeds u32 range",
                layout.offset()
            ))
        })?;
        i32::try_from(layout.offset()).map_err(|_| {
            invalid(format!(
                "window offset {} exceeds signed WGSL address range",
                layout.offset()
            ))
        })?;
        let (_, maximum_offset) = layout
            .checked_min_max_offsets()
            .map_err(|error| invalid(format!("window layout address range rejected: {error}")))?;
        i32::try_from(maximum_offset).map_err(|_| {
            invalid(format!(
                "window maximum offset {maximum_offset} exceeds signed WGSL address range"
            ))
        })?;

        let mut shape = [[0_u32; 4]; 2];
        let mut strides = [[0_i32; 4]; 2];
        for axis in 0..R {
            shape[axis / 4][axis % 4] = u32::try_from(layout.shape()[axis]).map_err(|_| {
                invalid(format!(
                    "window extent {} on axis {axis} exceeds u32 range",
                    layout.shape()[axis]
                ))
            })?;
            strides[axis / 4][axis % 4] = i32::try_from(layout.strides()[axis]).map_err(|_| {
                invalid(format!(
                    "window stride {} on axis {axis} exceeds i32 range",
                    layout.strides()[axis]
                ))
            })?;
            validate_axis_span(layout.shape()[axis], layout.strides()[axis], axis)?;
        }
        validate_product(&shape, R)?;

        Ok(Self {
            shape,
            strides,
            offset_and_rank: [
                offset,
                u32::try_from(R).expect("invariant: rank fits u32"),
                0,
                0,
            ],
        })
    }

    pub(super) const fn empty() -> Self {
        Self {
            shape: [[0; 4]; 2],
            strides: [[0; 4]; 2],
            offset_and_rank: [0; VECTOR_WIDTH],
        }
    }
}

/// Fixed WGSL uniform for pooling and sliding-window kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct WindowMeta {
    pub(super) source: WindowLayoutMeta,
    pub(super) target: WindowLayoutMeta,
    pub(super) destination: WindowLayoutMeta,
    pub(super) kernel: [u32; VECTOR_WIDTH],
    pub(super) stride: [u32; VECTOR_WIDTH],
    pub(super) padding_values: [u32; VECTOR_WIDTH],
    pub(super) dilation: [u32; VECTOR_WIDTH],
    pub(super) output_spatial: [u32; VECTOR_WIDTH],
    /// `[spatial_rank, kernel_volume, output_locations, batch]`.
    pub(super) geometry: [u32; VECTOR_WIDTH],
}

impl WindowMeta {
    pub(super) fn new<const S: usize>(
        source: WindowLayoutMeta,
        target: WindowLayoutMeta,
        destination: WindowLayoutMeta,
        plan: &WindowPlan<S>,
    ) -> Result<Self> {
        Ok(Self {
            source,
            target,
            destination,
            kernel: parameters(plan.parameters.kernel(), "kernel")?,
            stride: parameters(plan.parameters.stride(), "stride")?,
            padding_values: parameters(plan.parameters.padding(), "padding")?,
            dilation: parameters(plan.parameters.dilation(), "dilation")?,
            output_spatial: parameters(&plan.output_spatial, "output spatial")?,
            geometry: [
                u32::try_from(S).map_err(|_| invalid("window spatial rank exceeds u32 range"))?,
                u32::try_from(plan.kernel_volume)
                    .map_err(|_| invalid("window kernel volume exceeds u32 range"))?,
                u32::try_from(plan.output_locations)
                    .map_err(|_| invalid("window output location count exceeds u32 range"))?,
                u32::try_from(plan.batch)
                    .map_err(|_| invalid("window batch extent exceeds u32 range"))?,
            ],
        })
    }
}

fn parameters<const S: usize>(values: &[usize; S], name: &str) -> Result<[u32; VECTOR_WIDTH]> {
    if S > VECTOR_WIDTH {
        return Err(invalid(format!(
            "window spatial rank {S} exceeds metadata rank {VECTOR_WIDTH}"
        )));
    }
    let mut output = [0_u32; VECTOR_WIDTH];
    for (axis, &value) in values.iter().enumerate() {
        output[axis] = u32::try_from(value)
            .map_err(|_| invalid(format!("window {name}[{axis}] exceeds u32 range")))?;
    }
    Ok(output)
}

fn validate_product(shape: &[[u32; 4]; 2], rank: usize) -> Result<()> {
    let mut product = 1_u32;
    for axis in 0..rank {
        let extent = shape[axis / 4][axis % 4];
        product = product.checked_mul(extent).ok_or_else(|| {
            invalid(format!(
                "window element product overflows WGSL u32 at axis {axis}"
            ))
        })?;
        if product > i32::MAX as u32 {
            return Err(invalid(format!(
                "window element product {product} exceeds signed WGSL address range"
            )));
        }
    }
    Ok(())
}

fn validate_axis_span(extent: usize, stride: isize, axis: usize) -> Result<()> {
    let steps = extent.saturating_sub(1);
    let span = steps.checked_mul(stride.unsigned_abs()).ok_or_else(|| {
        invalid(format!(
            "window address span overflows on axis {axis}: extent {extent}, stride {stride}"
        ))
    })?;
    if span > i32::MAX as usize {
        return Err(invalid(format!(
            "window address span {span} on axis {axis} exceeds signed WGSL range"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

const _: () = assert!(core::mem::size_of::<WindowLayoutMeta>().is_multiple_of(16));
const _: () = assert!(core::mem::size_of::<WindowMeta>().is_multiple_of(16));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_layout_is_uniform_aligned() {
        assert_eq!(core::mem::size_of::<WindowLayoutMeta>(), 80);
        assert_eq!(core::mem::size_of::<WindowMeta>(), 336);
    }

    #[test]
    fn rejects_signed_stride_span_overflow() {
        let stride = isize::try_from(i32::MAX).expect("invariant: 64-bit isize represents i32");
        let layout = Layout::try_new([2], [stride + 1], 0).expect("valid test layout");
        let error = WindowLayoutMeta::new(&layout).expect_err("span exceeds WGSL range");
        assert!(error.to_string().contains("signed WGSL address range"));
    }
}
