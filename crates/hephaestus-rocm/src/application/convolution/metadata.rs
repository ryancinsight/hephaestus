use hephaestus_core::{HephaestusError, Result};
use leto::{ConvolutionParameters, Layout, TransposedConvolutionParameters};

const TENSOR_AXES: usize = 5;
const SPATIAL_AXES: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct LayoutMeta {
    shape: [i32; TENSOR_AXES],
    strides: [i32; TENSOR_AXES],
    offset: i32,
    rank: i32,
}

impl LayoutMeta {
    fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > TENSOR_AXES {
            return Err(invalid(format!(
                "convolution rank {R} exceeds ROCm metadata rank {TENSOR_AXES}"
            )));
        }
        let (minimum, maximum) = layout.checked_min_max_offsets().map_err(|error| {
            invalid(format!(
                "convolution layout address range rejected: {error}"
            ))
        })?;
        i32::try_from(minimum).map_err(|_| {
            invalid(format!(
                "layout minimum offset {minimum} exceeds signed ROCm address range"
            ))
        })?;
        i32::try_from(maximum).map_err(|_| {
            invalid(format!(
                "layout maximum offset {maximum} exceeds signed ROCm address range"
            ))
        })?;

        let mut shape = [0; TENSOR_AXES];
        let mut strides = [0; TENSOR_AXES];
        for axis in 0..R {
            shape[axis] = i32::try_from(layout.shape[axis]).map_err(|_| {
                invalid(format!(
                    "layout extent {} on axis {axis} exceeds i32 range",
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
        validate_products(&shape[..R])?;

        Ok(Self {
            shape,
            strides,
            offset: i32::try_from(layout.offset).map_err(|_| {
                invalid(format!(
                    "layout offset {} exceeds signed ROCm address range",
                    layout.offset
                ))
            })?,
            rank: i32::try_from(R).expect("invariant: ROCm metadata rank fits i32"),
        })
    }

    const fn empty() -> Self {
        Self {
            shape: [0; TENSOR_AXES],
            strides: [0; TENSOR_AXES],
            offset: 0,
            rank: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct ConvolutionMeta {
    input: LayoutMeta,
    weight: LayoutMeta,
    output: LayoutMeta,
    destination: LayoutMeta,
    stride: [i32; SPATIAL_AXES],
    padding: [i32; SPATIAL_AXES],
    dilation: [i32; SPATIAL_AXES],
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
        )
    }

    pub(super) fn with_target<const R: usize>(mut self, target: &Layout<R>) -> Result<Self> {
        self.destination = LayoutMeta::new(target)?;
        Ok(self)
    }

    fn new<const R: usize, const S: usize>(
        input: &Layout<R>,
        weight: &Layout<R>,
        output: &Layout<R>,
        destination: Result<LayoutMeta>,
        stride: &[usize; S],
        padding: &[usize; S],
        dilation: &[usize; S],
    ) -> Result<Self> {
        Ok(Self {
            input: LayoutMeta::new(input)?,
            weight: LayoutMeta::new(weight)?,
            output: LayoutMeta::new(output)?,
            destination: destination?,
            stride: parameters(stride, "stride")?,
            padding: parameters(padding, "padding")?,
            dilation: parameters(dilation, "dilation")?,
        })
    }
}

fn parameters<const S: usize>(values: &[usize; S], name: &str) -> Result<[i32; SPATIAL_AXES]> {
    if S > SPATIAL_AXES {
        return Err(invalid(format!(
            "convolution spatial rank {S} exceeds ROCm metadata rank {SPATIAL_AXES}"
        )));
    }
    let mut result = [0; SPATIAL_AXES];
    for (axis, &value) in values.iter().enumerate() {
        result[axis] = i32::try_from(value)
            .map_err(|_| invalid(format!("convolution {name}[{axis}] exceeds i32 range")))?;
    }
    Ok(result)
}

fn validate_products(shape: &[i32]) -> Result<()> {
    validate_product(shape, "logical")?;
    validate_product(shape.get(2..).unwrap_or_default(), "spatial")
}

fn validate_product(extents: &[i32], name: &str) -> Result<()> {
    let mut product = 1_i32;
    for (axis, &extent) in extents.iter().enumerate() {
        product = product.checked_mul(extent).ok_or_else(|| {
            invalid(format!(
                "layout {name} element product overflows ROCm i32 at axis {axis}"
            ))
        })?;
    }
    Ok(())
}

fn validate_axis_span(extent: usize, stride: isize, axis: usize) -> Result<()> {
    let span = extent
        .saturating_sub(1)
        .checked_mul(stride.unsigned_abs())
        .ok_or_else(|| {
            invalid(format!(
                "layout address span overflows on axis {axis}: extent {extent}, stride {stride}"
            ))
        })?;
    if span > i32::MAX as usize {
        return Err(invalid(format!(
            "layout address span {span} on axis {axis} exceeds signed ROCm range"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

const _: () = assert!(core::mem::size_of::<LayoutMeta>() == 48);
const _: () = assert!(core::mem::size_of::<ConvolutionMeta>() == 228);
