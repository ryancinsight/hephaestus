//! Backend-neutral axis-scan planning.
//!
//! Both GPU backends dispatch one workgroup/block per scan line. Each lane
//! owns a contiguous chunk, and the chunk totals are combined in logical
//! order, exposing lane-level parallel work. For associative operations this
//! preserves the sequential mathematical fold; floating-point addition and
//! multiplication are reassociated and therefore use a derived error bound.
//! The host-side contract is identical: validate the
//! shapes/strides, pack the launch metadata into a `#[repr(C)]` block that
//! mirrors the shader's uniform struct, and derive the workgroup count. That
//! logic is dialect-independent — only the shader source and the raw dispatch
//! differ — so it lives here once and both backends call
//! [`plan_axis_scan`](crate::plan_axis_scan).
//!
//! The one backend-specific input is whether the input and output buffers
//! alias (a device-pointer identity check); callers compute it and pass it in.

use crate::domain::error::{HephaestusError, Result};
use crate::domain::launch::BlockWidth;
use crate::domain::planning::{map_layout_err, to_i32, to_u32};
use bytemuck::{Pod, Zeroable};
use leto::Layout;

/// Direction of a scan along an axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScanDirection {
    /// Accumulate from index 0 upward.
    Forward,
    /// Accumulate from the last index downward.
    Reverse,
}

/// Launch metadata for the axis-scan kernel.
///
/// `#[repr(C)]` so it is bitwise-compatible with the WGSL/CUDA uniform struct
/// the shaders declare. `offsets` packs, in order: input element offset, output
/// element offset, `axis | direction_bit` (bit 1 set for [`ScanDirection::Reverse`]),
/// and the scan-line count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AxisScanMeta {
    /// Input shape `[rows, cols]`.
    pub input_shape: [u32; 2],
    /// Input strides `[row, col]` in elements.
    pub input_strides: [i32; 2],
    /// Output strides `[row, col]` in elements.
    pub output_strides: [i32; 2],
    /// Padding keeping `offsets` 16-byte aligned in the uniform block.
    pub _pre_offsets_pad: [u32; 2],
    /// `[input_offset, output_offset, axis | direction_bit, line_count]`.
    pub offsets: [u32; 4],
}

/// A validated axis-scan dispatch: launch metadata plus the line count.
#[derive(Clone, Copy, Debug)]
pub struct AxisScanDispatch {
    /// Launch metadata for the shader uniform.
    pub meta: AxisScanMeta,
    /// Workgroup (block) count, exactly one workgroup/block per scan line.
    pub groups: u32,
}

/// Validate a rank-2 axis scan and build its dispatch plan.
///
/// Returns `Ok(None)` when the output is empty (a no-op dispatch). The caller
/// supplies `buffers_alias` (the backend's device-pointer identity check
/// between the input and output buffers) and the two buffer element counts.
///
/// # Errors
/// Returns [`HephaestusError::DispatchFailed`] when the block width is not a
/// power of two, the axis is out of range, shapes mismatch, the buffers alias,
/// the output layout has zero-stride aliasing, a layout does not fit its
/// buffer, or an extent/stride exceeds the shader's `u32`/`i32` range.
#[allow(clippy::too_many_arguments)]
pub fn plan_axis_scan(
    input_layout: &Layout<2>,
    input_buf_len: usize,
    output_layout: &Layout<2>,
    output_buf_len: usize,
    axis: usize,
    direction: ScanDirection,
    width: BlockWidth,
    buffers_alias: bool,
) -> Result<Option<AxisScanDispatch>> {
    if !width.get().is_power_of_two() {
        return Err(HephaestusError::DispatchFailed {
            message: format!("scan block width {} must be a power of two", width.get()),
        });
    }
    if axis >= 2 {
        return Err(HephaestusError::DispatchFailed {
            message: format!("scan axis {axis} is out of bounds for rank-2 scan"),
        });
    }
    if input_layout.shape != output_layout.shape {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "scan output shape mismatch: input {:?}, out {:?}",
                input_layout.shape, output_layout.shape
            ),
        });
    }
    if buffers_alias {
        return Err(HephaestusError::DispatchFailed {
            message: "scan output buffer must not alias input buffer".to_string(),
        });
    }
    if output_layout.has_zero_stride_aliasing() {
        return Err(HephaestusError::DispatchFailed {
            message: "scan output layout must not contain zero-stride aliasing".to_string(),
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

    let direction_bit = match direction {
        ScanDirection::Forward => 0usize,
        ScanDirection::Reverse => 2usize,
    };
    // Lines run along `axis`, so their count is the orthogonal extent. The
    // tiled kernel launches one workgroup/block per line; every lane then
    // owns a contiguous chunk of that line. Non-zero here because output_len
    // > 0.
    let line_count = input_layout.shape[1 - axis];
    let meta = AxisScanMeta {
        input_shape: [
            to_u32(input_layout.shape[0], "input rows")?,
            to_u32(input_layout.shape[1], "input columns")?,
        ],
        input_strides: [
            to_i32(input_layout.strides[0], "input row stride")?,
            to_i32(input_layout.strides[1], "input column stride")?,
        ],
        output_strides: [
            to_i32(output_layout.strides[0], "output row stride")?,
            to_i32(output_layout.strides[1], "output column stride")?,
        ],
        _pre_offsets_pad: [0; 2],
        offsets: [
            to_u32(input_layout.offset, "input offset")?,
            to_u32(output_layout.offset, "output offset")?,
            to_u32(axis | direction_bit, "axis and direction")?,
            to_u32(line_count, "scan line count")?,
        ],
    };

    let line_count_u64 =
        u64::try_from(line_count).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("scan line count {line_count} exceeds u64 range"),
        })?;
    let groups = u32::try_from(line_count_u64).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("scan line count {line_count} exceeds u32 workgroup range"),
    })?;

    Ok(Some(AxisScanDispatch { meta, groups }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: [usize; 2]) -> Layout<2> {
        Layout::c_contiguous(shape).expect("valid contiguous layout")
    }

    fn width() -> BlockWidth {
        BlockWidth::new(256).expect("non-zero width")
    }

    #[test]
    fn plans_a_dense_forward_scan() {
        let l = layout([3, 4]);
        let plan = plan_axis_scan(&l, 12, &l, 12, 1, ScanDirection::Forward, width(), false)
            .expect("valid")
            .expect("non-empty");
        // axis 1, forward → offsets[2] = 1 | 0 = 1; line_count = rows = 3.
        assert_eq!(plan.meta.offsets[2], 1);
        assert_eq!(plan.meta.offsets[3], 3);
        assert_eq!(plan.groups, 3);
        assert_eq!(plan.meta.input_shape, [3, 4]);
    }

    #[test]
    fn reverse_sets_the_direction_bit() {
        let l = layout([2, 5]);
        let plan = plan_axis_scan(&l, 10, &l, 10, 0, ScanDirection::Reverse, width(), false)
            .expect("valid")
            .expect("non-empty");
        // axis 0, reverse → offsets[2] = 0 | 2 = 2; line_count = cols = 5.
        assert_eq!(plan.meta.offsets[2], 2);
        assert_eq!(plan.meta.offsets[3], 5);
    }

    #[test]
    fn aliasing_is_rejected() {
        let l = layout([2, 2]);
        let err =
            plan_axis_scan(&l, 4, &l, 4, 0, ScanDirection::Forward, width(), true).unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn empty_output_is_a_noop() {
        let l = layout([0, 4]);
        let plan =
            plan_axis_scan(&l, 0, &l, 0, 1, ScanDirection::Forward, width(), false).expect("valid");
        assert!(plan.is_none());
    }

    #[test]
    fn non_power_of_two_width_is_rejected() {
        let l = layout([2, 2]);
        let w = BlockWidth::new(3).expect("non-zero");
        let err = plan_axis_scan(&l, 4, &l, 4, 0, ScanDirection::Forward, w, false).unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }

    #[test]
    fn plans_one_workgroup_per_line_for_long_lines() {
        let l = layout([3, 1_024]);
        let plan = plan_axis_scan(
            &l,
            3 * 1_024,
            &l,
            3 * 1_024,
            1,
            ScanDirection::Forward,
            width(),
            false,
        )
        .expect("valid")
        .expect("non-empty");
        assert_eq!(plan.meta.offsets[3], 3);
        assert_eq!(plan.groups, 3);
    }
}
