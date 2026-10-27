//! Device-side rank-2 matrix layout metadata, signed 32-bit address range
//! validation, and width conversions for CUDA.

use eunomia::{Pod, Zeroable};
use hephaestus_core::{HephaestusError, Result, map_layout_err};
use leto::Layout;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuMatrixLayout {
    pub(super) shape: [u32; 2],
    pub(super) strides: [i32; 2],
    pub(super) offset: u32,
    pub(super) _pad: [u32; 3],
}

/// Require every address of a rank-2 layout to fit the signed 32-bit range
/// the `matmul`/`kron` kernels compute in.
///
/// Those kernels form each element address from the converted origin plus
/// the converted row coordinate times the row stride plus the converted
/// column coordinate times the column stride, all in 32-bit signed
/// arithmetic. Addressing is affine over the index rectangle, so every
/// addressed cell lies between the
/// minimum and maximum corner addresses reported by
/// [`Layout::checked_min_max_offsets`]. Leto's `Layout` invariant pins the
/// minimum at zero or above (its only extra-crate construction paths
/// validate); requiring the maximum at or below `i32::MAX` therefore keeps
/// every offset conversion, coordinate product, and left-to-right partial
/// sum the kernels evaluate inside the signed 32-bit range as well:
/// single-axis corners isolate each product, and each partial sum is bounded
/// by a corner on the rectangle edge being swept. Layouts with an empty
/// shape address no cells and always pass; the dispatchers additionally
/// short-circuit those before launch.
///
/// The batched kernel is intentionally outside this bound: it widens every
/// address term to `long long`, so origins above `i32::MAX` remain
/// representable there.
pub(super) fn validate_signed_address_range(layout: &Layout<2>, what: &str) -> Result<()> {
    let [rows, cols] = layout.shape();
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let (_min, max) = layout.checked_min_max_offsets().map_err(map_layout_err)?;
    if i32::try_from(max).is_err() {
        return Err(HephaestusError::DispatchFailed {
            message: format!(
                "{what} maximum address {max} exceeds the signed 32-bit device address range"
            ),
        });
    }
    Ok(())
}

#[inline]
pub(super) fn map_layout(layout: &Layout<2>) -> Result<GpuMatrixLayout> {
    Ok(GpuMatrixLayout {
        shape: [
            to_u32(layout.shape()[0], "dimension")?,
            to_u32(layout.shape()[1], "dimension")?,
        ],
        strides: [
            to_i32(layout.strides()[0], "stride")?,
            to_i32(layout.strides()[1], "stride")?,
        ],
        offset: to_u32(layout.offset(), "offset")?,
        _pad: [0; 3],
    })
}

/// Convert a `usize` extent to the `u32` the device-side layout struct uses.
#[inline]
pub(crate) fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds u32 range"),
    })
}

/// Convert an `isize` stride to the `i32` the device-side layout struct uses.
#[inline]
pub(crate) fn to_i32(value: isize, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i32 range"),
    })
}

/// Convert an `isize` batch stride to the `long long` (`i64`) a batched
/// kernel argument uses. Unlike the per-matrix `i32` strides, a batch stride
/// is multiplied by up to the batch count on the device side, so it needs
/// the wider range even though `isize` is losslessly representable on every
/// 64-bit target this crate builds for.
#[inline]
pub(crate) fn to_i64(value: isize, what: &str) -> Result<i64> {
    i64::try_from(value).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("{what} {value} exceeds i64 range"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: [usize; 2], strides: [isize; 2], offset: usize) -> Layout<2> {
        Layout::try_new(shape, strides, offset).expect("invariant: metadata fixture constructs")
    }

    #[test]
    fn ordinary_contiguous_layouts_pass() {
        let contiguous = Layout::c_contiguous([64, 64]).expect("invariant: fixture constructs");
        validate_signed_address_range(&contiguous, "output").expect("ordinary layout passes");
    }

    #[test]
    fn negative_strides_in_range_pass() {
        // Corners {10, 9, 12, 11}: negative stride, every address in range.
        let view = layout([2, 2], [2, -1], 10);
        validate_signed_address_range(&view, "output").expect("in-range negative strides pass");
    }

    #[test]
    fn origin_at_i32_max_boundary_passes() {
        let edge = layout(
            [1, 1],
            [1, 1],
            usize::try_from(i32::MAX).expect("i32 fits usize"),
        );
        validate_signed_address_range(&edge, "output").expect("max-boundary address passes");
    }

    #[test]
    fn origin_past_i32_max_fails() {
        let origin = layout([1, 1], [1, 1], 1usize << 31);
        let error =
            validate_signed_address_range(&origin, "output").expect_err("origin at 2^31 must fail");
        assert!(
            error.to_string().contains("signed 32-bit"),
            "unexpected rejection message: {error}"
        );
    }

    #[test]
    fn boundary_overflow_by_one_cell_fails() {
        let edge = layout(
            [1, 2],
            [0, 1],
            usize::try_from(i32::MAX).expect("i32 fits usize"),
        );
        validate_signed_address_range(&edge, "output").expect_err("max address 2^31 must fail");
    }

    #[test]
    fn coordinate_products_beyond_i32_fail() {
        // Max corner (2^20 - 1) * 2^20: representable metadata, unrepresentable address.
        let wide = layout([1, 1 << 20], [0, 1 << 20], 0);
        validate_signed_address_range(&wide, "output").expect_err("2^40-scale product must fail");
    }

    #[test]
    fn leto_construction_pins_nonnegative_minima() {
        // Soundness pin: `validate_signed_address_range` only bounds the
        // maximum because leto's only extra-crate construction path rejects
        // negative minima itself. If that ever relaxes, this fails loudly.
        assert!(
            Layout::try_new([2, 2], [2, -2], 1).is_err(),
            "leto must keep rejecting negative-minimum layouts"
        );
    }

    #[test]
    fn empty_shapes_pass_regardless_of_metadata() {
        let empty = layout([0, 5], [isize::MIN, isize::MAX], usize::MAX);
        validate_signed_address_range(&empty, "output").expect("empty shape addresses nothing");
    }

    #[test]
    fn huge_zero_stride_extent_passes() {
        // Every cell addresses the base: `(int)row` may wrap, but the
        // product with a zero stride is still zero.
        let broadcast = layout(
            [
                usize::try_from(u32::MAX)
                    .expect("invariant: supported targets have a usize of at least 32 bits"),
                1,
            ],
            [0, 1],
            0,
        );
        validate_signed_address_range(&broadcast, "output").expect("zero-stride extent passes");
    }
}
