use eunomia::{Pod, Zeroable};
use hephaestus_core::{AttentionPlan, HephaestusError, Result};
use leto::Layout;

/// One rank-three strided view in the WGSL metadata ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct LayoutMeta {
    shape: [u32; 4],
    strides: [i32; 4],
    offset: [i32; 4],
}

impl LayoutMeta {
    pub(super) fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > 3 {
            return Err(invalid(format!(
                "attention WGSL metadata supports rank at most 3, got {R}"
            )));
        }
        let mut shape = [1_u32; 4];
        let mut strides = [0_i32; 4];
        for axis in 0..R {
            shape[axis] = u32::try_from(layout.shape()[axis]).map_err(|_| {
                invalid(format!(
                    "attention extent {} on axis {axis} exceeds u32 range",
                    layout.shape()[axis]
                ))
            })?;
            strides[axis] = i32::try_from(layout.strides()[axis]).map_err(|_| {
                invalid(format!(
                    "attention stride {} on axis {axis} exceeds i32 range",
                    layout.strides()[axis]
                ))
            })?;
        }
        let offset = i32::try_from(layout.offset()).map_err(|_| {
            invalid(format!(
                "attention offset {} exceeds signed WGSL address range",
                layout.offset()
            ))
        })?;
        let (minimum, maximum) = layout.checked_min_max_offsets().map_err(|error| {
            invalid(format!("attention layout address range rejected: {error}"))
        })?;
        i32::try_from(minimum).map_err(|_| {
            invalid(format!(
                "attention minimum offset {minimum} exceeds signed WGSL address range"
            ))
        })?;
        i32::try_from(maximum).map_err(|_| {
            invalid(format!(
                "attention maximum offset {maximum} exceeds signed WGSL address range"
            ))
        })?;
        Ok(Self {
            shape,
            strides,
            offset: [offset, 0, 0, 0],
        })
    }

    pub(super) const fn empty() -> Self {
        Self {
            shape: [1; 4],
            strides: [0; 4],
            offset: [0; 4],
        }
    }
}

/// Fixed-size uniform shared by forward and all backward kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(super) struct AttentionMeta<T: super::WgslAttentionScalar> {
    query: LayoutMeta,
    key: LayoutMeta,
    value: LayoutMeta,
    weights: LayoutMeta,
    grad_output: LayoutMeta,
    destination: LayoutMeta,
    keep_mask: LayoutMeta,
    dimensions: [u32; 4],
    value_and_flags: [u32; 4],
    scale_and_padding: T::ScaleRepr,
}

impl<T: super::WgslAttentionScalar> AttentionMeta<T> {
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor enumerates the complete attention kernel ABI"
    )]
    pub(super) fn new(
        plan: AttentionPlan,
        query: &Layout<3>,
        key: &Layout<3>,
        value: &Layout<3>,
        weights: &Layout<3>,
        grad_output: Option<&Layout<3>>,
        destination: &Layout<3>,
        keep_mask: Option<&Layout<2>>,
        heads_per_batch: usize,
        causal: bool,
        scale: T,
    ) -> Result<Self> {
        let dimensions = [
            narrow(plan.batch, "batch")?,
            narrow(plan.query_sequence, "query sequence")?,
            narrow(plan.key_sequence, "key sequence")?,
            narrow(plan.key_feature, "key feature")?,
        ];
        let value_and_flags = [
            narrow(plan.value_feature, "value feature")?,
            u32::from(causal),
            u32::from(keep_mask.is_some()),
            narrow(heads_per_batch, "heads per batch")?,
        ];
        Ok(Self {
            query: LayoutMeta::new(query)?,
            key: LayoutMeta::new(key)?,
            value: LayoutMeta::new(value)?,
            weights: LayoutMeta::new(weights)?,
            grad_output: grad_output.map_or_else(|| Ok(LayoutMeta::empty()), LayoutMeta::new)?,
            destination: LayoutMeta::new(destination)?,
            keep_mask: keep_mask.map_or_else(|| Ok(LayoutMeta::empty()), LayoutMeta::new)?,
            dimensions,
            value_and_flags,
            scale_and_padding: T::scale_repr(scale),
        })
    }
}

// SAFETY: `repr(C)` over `Pod` fields with no padding. Seven 48-byte
// `LayoutMeta`s plus two 16-byte `u32` arrays end at offset 368, which is a
// multiple of 8, so both 16-byte scale tails (`[f32; 4]`, `[f64; 2]`) sit
// naturally aligned; total size is 384 with no trailing padding. Every
// field type admits all bit patterns, and all-zeroes is a valid value.
unsafe impl<T: super::WgslAttentionScalar> Zeroable for AttentionMeta<T> {}
// SAFETY: see the `Zeroable` impl; additionally `Copy + 'static`, and the
// struct contains no interior mutability, pointers, or references.
unsafe impl<T: super::WgslAttentionScalar> Pod for AttentionMeta<T> {}

fn narrow(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("attention {name} {value} exceeds u32 range")))
}

fn invalid(message: String) -> HephaestusError {
    HephaestusError::InvalidConfiguration { message }
}

#[cfg(test)]
mod tests {
    use super::super::WgslAttentionScalar;
    use super::*;

    #[test]
    fn uniform_tail_keeps_one_shape_for_both_scalars() {
        // The manual `Pod` impl promises 384 padding-free bytes for both.
        assert_eq!(core::mem::size_of::<AttentionMeta<f32>>(), 384);
        assert_eq!(core::mem::size_of::<AttentionMeta<f64>>(), 384);
        assert_eq!(
            <f32 as WgslAttentionScalar>::scale_repr(0.5),
            [0.5, 0.0, 0.0, 0.0]
        );
        assert_eq!(<f64 as WgslAttentionScalar>::scale_repr(0.5), [0.5, 0.0]);
    }
}
