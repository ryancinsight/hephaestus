use crate::domain::error::{HephaestusError, Result};
use crate::domain::launch::BlockWidth;
use crate::domain::planning::{map_layout_err, to_i32, to_u32};
use eunomia::{Pod, Zeroable};
use leto::Layout;

/// Validate that a reduction block width can be halved into a workgroup tree.
///
/// Both WGSL and CUDA scalar reductions reduce by repeatedly halving active
/// lanes. A non-power-of-two width would drop lanes, so it is rejected before
/// backend dispatch.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when `width` is not a power of
/// two.
pub fn validate_reduction_width(width: BlockWidth) -> Result<()> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "reduction block width {} must be a power of two",
                width.get()
            ),
        });
    }
    Ok(())
}

/// Return the number of tree-reduction passes needed to reduce `len` elements.
///
/// The value is backend-neutral host planning: each pass emits
/// `ceil(current_len / width)` partials until one element remains.
#[must_use]
pub fn reduction_pass_count(mut len: usize, width: BlockWidth) -> usize {
    // `width.get()` is `u32`; this is a lossless widening on all supported
    // Rust targets (`usize >= 32` bits).
    let width = width.get() as usize;
    let mut passes = 0;
    while len > 1 {
        len = len.div_ceil(width);
        passes += 1;
    }
    passes
}

/// Launch metadata for the axis-reduction kernel.
///
/// `#[repr(C)]` so it is bitwise-compatible with the WGSL/CUDA uniform struct
/// the shaders declare. `offsets` packs, in order: input element offset, output
/// element offset, the reduced axis, and the output element count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AxisReductionMeta {
    /// Input shape `[rows, cols]`.
    pub input_shape: [u32; 2],
    /// Input strides `[row, col]` in elements.
    pub input_strides: [i32; 2],
    /// Output strides `[row, col]` in elements.
    pub output_strides: [i32; 2],
    /// Padding keeping `offsets` 16-byte aligned in the uniform block.
    pub _pre_offsets_pad: [u32; 2],
    /// `[input_offset, output_offset, axis, output_len]`.
    pub offsets: [u32; 4],
}

/// A validated axis-reduction dispatch: launch metadata plus the workgroup count.
#[derive(Clone, Copy, Debug)]
pub struct AxisReductionDispatch {
    /// Launch metadata for the shader uniform.
    pub meta: AxisReductionMeta,
    /// Workgroup (block) count covering the output length.
    pub groups: u32,
}

/// Validate a rank-2 axis reduction and build its dispatch plan.
///
/// Returns `Ok(None)` when the output is empty (a no-op dispatch). The output
/// shape must equal the input shape with the reduced `axis` collapsed to 1.
/// The caller supplies `buffers_alias` (the backend's device-pointer identity
/// check) and the two buffer element counts.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when the block width is not a
/// power of two, the axis is out of range, the output shape does not match the
/// reduced input shape, the buffers alias, the output layout has zero-stride
/// aliasing, a layout does not fit its buffer, or an extent/stride exceeds the
/// shader's `u32`/`i32` range.
pub fn plan_axis_reduction(
    input_layout: &Layout<2>,
    input_buf_len: usize,
    output_layout: &Layout<2>,
    output_buf_len: usize,
    axis: usize,
    width: BlockWidth,
    buffers_alias: bool,
) -> Result<Option<AxisReductionDispatch>> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "reduction block width {} must be a power of two",
                width.get()
            ),
        });
    }
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("axis {axis} is out of bounds for rank-2 reduction"),
        });
    }

    let mut expected_shape = input_layout.shape();
    expected_shape[axis] = 1;
    if output_layout.shape() != expected_shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "axis reduction output shape mismatch: input {:?}, axis {axis}, out {:?}",
                input_layout.shape(),
                output_layout.shape()
            ),
        });
    }
    if buffers_alias {
        return Err(HephaestusError::DispatchFailed {
            message: "axis reduction output buffer must not alias input buffer".to_string(),
        });
    }
    if output_layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "axis reduction output layout must not contain zero-stride aliasing"
                .to_string(),
        });
    }

    input_layout
        .validate_storage_len(input_buf_len)
        .map_err(map_layout_err)?;
    output_layout
        .validate_storage_len(output_buf_len)
        .map_err(map_layout_err)?;
    let output_len = output_layout.checked_size().map_err(map_layout_err)?;
    if output_len == 0 {
        return Ok(None);
    }

    let output_len_u64 =
        u64::try_from(output_len).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("output length {output_len} exceeds u64 range"),
        })?;
    let groups = width
        .checked_covering_blocks(output_len_u64)
        .ok_or_else(|| HephaestusError::DispatchFailed {
            message: format!("output length {output_len} exceeds u32 workgroup range"),
        })?;

    let meta = AxisReductionMeta {
        input_shape: [
            to_u32(input_layout.shape()[0], "input rows")?,
            to_u32(input_layout.shape()[1], "input columns")?,
        ],
        input_strides: [
            to_i32(input_layout.strides()[0], "input row stride")?,
            to_i32(input_layout.strides()[1], "input column stride")?,
        ],
        output_strides: [
            to_i32(output_layout.strides()[0], "output row stride")?,
            to_i32(output_layout.strides()[1], "output column stride")?,
        ],
        _pre_offsets_pad: [0; 2],
        offsets: [
            to_u32(input_layout.offset(), "input offset")?,
            to_u32(output_layout.offset(), "output offset")?,
            to_u32(axis, "axis")?,
            to_u32(output_len, "output length")?,
        ],
    };

    Ok(Some(AxisReductionDispatch { meta, groups }))
}
