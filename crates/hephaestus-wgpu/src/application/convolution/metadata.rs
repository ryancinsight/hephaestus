use bytemuck::{Pod, Zeroable};
use hephaestus_core::{HephaestusError, Result};
use leto::{ConvolutionParameters, Layout, TransposedConvolutionParameters};

const METADATA_AXES: usize = 8;
const VECTOR_AXES: usize = 4;

/// WGSL-compatible layout metadata.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct LayoutMeta {
    shape_low: [u32; VECTOR_AXES],
    shape_high: [u32; VECTOR_AXES],
    strides_low: [i32; VECTOR_AXES],
    strides_high: [i32; VECTOR_AXES],
    offset_and_rank: [u32; VECTOR_AXES],
}

impl LayoutMeta {
    fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > METADATA_AXES {
            return Err(invalid(format!(
                "convolution rank {R} exceeds WGPU metadata rank {METADATA_AXES}"
            )));
        }

        let offset = u32::try_from(layout.offset)
            .map_err(|_| invalid(format!("layout offset {} exceeds u32 range", layout.offset)))?;
        i32::try_from(layout.offset).map_err(|_| {
            invalid(format!(
                "layout offset {} exceeds signed WGSL address range",
                layout.offset
            ))
        })?;
        let (_, maximum_offset) = layout.checked_min_max_offsets().map_err(|error| {
            invalid(format!(
                "convolution layout address range rejected: {error}"
            ))
        })?;
        i32::try_from(maximum_offset).map_err(|_| {
            invalid(format!(
                "layout maximum offset {maximum_offset} exceeds signed WGSL address range"
            ))
        })?;

        let mut shape = [0_u32; METADATA_AXES];
        let mut strides = [0_i32; METADATA_AXES];
        for axis in 0..R {
            shape[axis] = u32::try_from(layout.shape[axis]).map_err(|_| {
                invalid(format!(
                    "layout extent {} on axis {axis} exceeds u32 range",
                    layout.shape[axis]
                ))
            })?;
            strides[axis] = i32::try_from(layout.strides[axis]).map_err(|_| {
                invalid(format!(
                    "layout stride {} on axis {axis} exceeds i32 range",
                    layout.strides[axis]
                ))
            })?;
            validate_axis_span(layout.shape[axis], layout.strides[axis], axis)?;
        }
        validate_shader_products(&shape[..R])?;

        Ok(Self {
            shape_low: shape[..VECTOR_AXES]
                .try_into()
                .expect("invariant: low shape slice has four elements"),
            shape_high: shape[VECTOR_AXES..]
                .try_into()
                .expect("invariant: high shape slice has four elements"),
            strides_low: strides[..VECTOR_AXES]
                .try_into()
                .expect("invariant: low stride slice has four elements"),
            strides_high: strides[VECTOR_AXES..]
                .try_into()
                .expect("invariant: high stride slice has four elements"),
            offset_and_rank: [
                offset,
                u32::try_from(R).expect("invariant: metadata rank fits u32"),
                0,
                0,
            ],
        })
    }

    const fn empty() -> Self {
        Self {
            shape_low: [0; VECTOR_AXES],
            shape_high: [0; VECTOR_AXES],
            strides_low: [0; VECTOR_AXES],
            strides_high: [0; VECTOR_AXES],
            offset_and_rank: [0; VECTOR_AXES],
        }
    }
}

fn validate_shader_products(shape: &[u32]) -> Result<()> {
    validate_shader_product(shape, "logical")?;
    match shape {
        [_, _, spatial @ ..] => validate_shader_product(spatial, "spatial"),
        _ => Ok(()),
    }
}

fn validate_shader_product(extents: &[u32], name: &str) -> Result<()> {
    let mut product = 1_u32;
    for (axis, &extent) in extents.iter().enumerate() {
        product = product.checked_mul(extent).ok_or_else(|| {
            invalid(format!(
                "layout {name} element product overflows WGSL u32 at axis {axis}"
            ))
        })?;
        if product > i32::MAX as u32 {
            return Err(invalid(format!(
                "layout {name} element product {product} exceeds signed WGSL address range"
            )));
        }
    }
    Ok(())
}

fn validate_axis_span(extent: usize, stride: isize, axis: usize) -> Result<()> {
    let steps = extent.saturating_sub(1);
    let magnitude = stride.unsigned_abs();
    let span = steps.checked_mul(magnitude).ok_or_else(|| {
        invalid(format!(
            "layout address span overflows on axis {axis}: extent {extent}, stride {stride}"
        ))
    })?;
    if span > i32::MAX as usize {
        return Err(invalid(format!(
            "layout address span {span} on axis {axis} exceeds signed WGSL range"
        )));
    }
    Ok(())
}

/// Fixed-size WGSL uniform shared by every convolution kernel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct ConvolutionMeta {
    input: LayoutMeta,
    weight: LayoutMeta,
    output: LayoutMeta,
    destination: LayoutMeta,
    stride_low: [u32; VECTOR_AXES],
    stride_high: [u32; VECTOR_AXES],
    padding_low: [u32; VECTOR_AXES],
    padding_high: [u32; VECTOR_AXES],
    dilation_low: [u32; VECTOR_AXES],
    dilation_high: [u32; VECTOR_AXES],
    output_padding_low: [u32; VECTOR_AXES],
    output_padding_high: [u32; VECTOR_AXES],
    spatial_rank_and_flags: [u32; VECTOR_AXES],
}

impl ConvolutionMeta {
    pub(super) fn regular_forward<const R: usize, const S: usize>(
        input: &Layout<R>,
        weight: &Layout<R>,
        output: &Layout<R>,
        bias: Option<&Layout<1>>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self> {
        Self::new(
            input,
            weight,
            output,
            bias.map_or_else(|| Ok(LayoutMeta::empty()), LayoutMeta::new),
            parameters.stride(),
            parameters.padding(),
            parameters.dilation(),
            &[0; S],
            S,
        )
    }

    pub(super) fn transposed_forward<const R: usize, const S: usize>(
        input: &Layout<R>,
        weight: &Layout<R>,
        output: &Layout<R>,
        bias: Option<&Layout<1>>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self> {
        Self::new(
            input,
            weight,
            output,
            bias.map_or_else(|| Ok(LayoutMeta::empty()), LayoutMeta::new),
            parameters.stride(),
            parameters.padding(),
            parameters.dilation(),
            parameters.output_padding(),
            S,
        )
    }

    pub(super) fn regular_backward<const R: usize, const S: usize>(
        input: &Layout<R>,
        weight: &Layout<R>,
        grad_output: &Layout<R>,
        parameters: ConvolutionParameters<S>,
    ) -> Result<Self> {
        Self::new(
            input,
            weight,
            grad_output,
            Ok(LayoutMeta::empty()),
            parameters.stride(),
            parameters.padding(),
            parameters.dilation(),
            &[0; S],
            S,
        )
    }

    pub(super) fn transposed_backward<const R: usize, const S: usize>(
        input: &Layout<R>,
        weight: &Layout<R>,
        grad_output: &Layout<R>,
        parameters: TransposedConvolutionParameters<S>,
    ) -> Result<Self> {
        Self::new(
            input,
            weight,
            grad_output,
            Ok(LayoutMeta::empty()),
            parameters.stride(),
            parameters.padding(),
            parameters.dilation(),
            parameters.output_padding(),
            S,
        )
    }

    pub(super) fn with_target<const R: usize>(mut self, target: &Layout<R>) -> Result<Self> {
        self.destination = LayoutMeta::new(target)?;
        Ok(self)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the metadata constructor enumerates the complete convolution ABI"
    )]
    fn new<const R: usize, const S: usize>(
        input: &Layout<R>,
        weight: &Layout<R>,
        output: &Layout<R>,
        destination: Result<LayoutMeta>,
        stride: &[usize; S],
        padding: &[usize; S],
        dilation: &[usize; S],
        output_padding: &[usize; S],
        spatial_rank: usize,
    ) -> Result<Self> {
        let stride = parameter_vectors(stride, "stride")?;
        let padding = parameter_vectors(padding, "padding")?;
        let dilation = parameter_vectors(dilation, "dilation")?;
        let output_padding = parameter_vectors(output_padding, "output padding")?;
        Ok(Self {
            input: LayoutMeta::new(input)?,
            weight: LayoutMeta::new(weight)?,
            output: LayoutMeta::new(output)?,
            destination: destination?,
            stride_low: stride.0,
            stride_high: stride.1,
            padding_low: padding.0,
            padding_high: padding.1,
            dilation_low: dilation.0,
            dilation_high: dilation.1,
            output_padding_low: output_padding.0,
            output_padding_high: output_padding.1,
            spatial_rank_and_flags: [
                u32::try_from(spatial_rank)
                    .map_err(|_| invalid("spatial rank exceeds u32 range"))?,
                0,
                0,
                0,
            ],
        })
    }
}

fn parameter_vectors<const S: usize>(
    values: &[usize; S],
    name: &str,
) -> Result<([u32; VECTOR_AXES], [u32; VECTOR_AXES])> {
    if S > METADATA_AXES {
        return Err(invalid(format!(
            "convolution spatial rank {S} exceeds metadata rank {METADATA_AXES}"
        )));
    }
    let mut padded = [0_u32; METADATA_AXES];
    for (axis, &value) in values.iter().enumerate() {
        padded[axis] = u32::try_from(value)
            .map_err(|_| invalid(format!("convolution {name}[{axis}] exceeds u32 range")))?;
    }
    Ok((
        padded[..VECTOR_AXES]
            .try_into()
            .expect("invariant: low parameter slice has four elements"),
        padded[VECTOR_AXES..]
            .try_into()
            .expect("invariant: high parameter slice has four elements"),
    ))
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

const _: () = assert!(core::mem::size_of::<LayoutMeta>().is_multiple_of(16));
const _: () = assert!(core::mem::size_of::<ConvolutionMeta>().is_multiple_of(16));

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_stride_spans_outside_signed_wgsl_addressing() {
        let maximum_stride =
            isize::try_from(i32::MAX).expect("invariant: 64-bit isize represents i32");
        let layout = Layout::new([2], [maximum_stride + 1], 0);
        let error = LayoutMeta::new(&layout).expect_err("stride exceeds i32");
        assert_eq!(
            error.to_string(),
            format!(
                "invalid configuration: layout maximum offset {} exceeds signed WGSL address range",
                maximum_stride + 1
            )
        );
    }

    #[test]
    fn metadata_layout_matches_wgsl_alignment() {
        assert_eq!(core::mem::size_of::<LayoutMeta>(), 80);
        assert_eq!(core::mem::size_of::<ConvolutionMeta>(), 464);
    }

    #[test]
    fn rejects_overlapping_reads_with_unaddressable_logical_products() {
        let layout = Layout::new([46_341, 46_341, 1], [0, 0, 0], 0);
        let error = LayoutMeta::new(&layout).expect_err("logical product exceeds i32");
        assert_eq!(
            error.to_string(),
            "invalid configuration: layout logical element product 2147488281 exceeds signed WGSL address range"
        );
    }
}
