use bytemuck::{Pod, Zeroable};
use leto::Layout;

use crate::domain::error::{HephaestusError, Result};
use crate::domain::window::WindowPlan;

const MAX_RANK: usize = 5;
const SPATIAL_RANK: usize = 3;

/// C-compatible metadata for one strided tensor layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct WindowLayoutMeta {
    pub(super) shape: [u32; MAX_RANK],
    pub(super) strides: [i32; MAX_RANK],
    pub(super) offset: u32,
    pub(super) rank: u32,
}

impl WindowLayoutMeta {
    pub(super) fn new<const R: usize>(layout: &Layout<R>) -> Result<Self> {
        if R > MAX_RANK {
            return Err(invalid(format!(
                "window rank {R} exceeds C-family metadata rank {MAX_RANK}"
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
                "window offset {} exceeds signed C-family address range",
                layout.offset()
            ))
        })?;
        let (_, maximum_offset) = layout
            .checked_min_max_offsets()
            .map_err(|error| invalid(format!("window layout address range rejected: {error}")))?;
        i32::try_from(maximum_offset).map_err(|_| {
            invalid(format!(
                "window maximum offset {maximum_offset} exceeds signed C-family address range"
            ))
        })?;

        let mut shape = [0_u32; MAX_RANK];
        let mut strides = [0_i32; MAX_RANK];
        for axis in 0..R {
            shape[axis] = u32::try_from(layout.shape()[axis]).map_err(|_| {
                invalid(format!(
                    "window extent {} on axis {axis} exceeds u32 range",
                    layout.shape()[axis]
                ))
            })?;
            strides[axis] = i32::try_from(layout.strides()[axis]).map_err(|_| {
                invalid(format!(
                    "window stride {} on axis {axis} exceeds i32 range",
                    layout.strides()[axis]
                ))
            })?;
            let span = layout.shape()[axis]
                .saturating_sub(1)
                .checked_mul(layout.strides()[axis].unsigned_abs())
                .ok_or_else(|| invalid(format!("window address span overflows on axis {axis}")))?;
            if span > i32::MAX as usize {
                return Err(invalid(format!(
                    "window address span {span} on axis {axis} exceeds signed C-family range"
                )));
            }
        }

        let mut product = 1_u32;
        for (axis, &extent) in shape[..R].iter().enumerate() {
            product = product.checked_mul(extent).ok_or_else(|| {
                invalid(format!(
                    "window element product overflows C-family u32 at axis {axis}"
                ))
            })?;
            if product > i32::MAX as u32 {
                return Err(invalid(format!(
                    "window element product {product} exceeds signed C-family address range"
                )));
            }
        }

        Ok(Self {
            shape,
            strides,
            offset,
            rank: u32::try_from(R).expect("invariant: metadata rank is bounded by five"),
        })
    }

    pub(super) const fn empty() -> Self {
        Self {
            shape: [0; MAX_RANK],
            strides: [0; MAX_RANK],
            offset: 0,
            rank: 0,
        }
    }
}

/// C-compatible uniform metadata shared by pooling and sliding-window kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct WindowMeta {
    pub(super) source: WindowLayoutMeta,
    pub(super) target: WindowLayoutMeta,
    pub(super) destination: WindowLayoutMeta,
    pub(super) kernel: [u32; SPATIAL_RANK],
    pub(super) stride: [u32; SPATIAL_RANK],
    pub(super) padding: [u32; SPATIAL_RANK],
    pub(super) dilation: [u32; SPATIAL_RANK],
    pub(super) output_spatial: [u32; SPATIAL_RANK],
    /// `[spatial_rank, kernel_volume, output_locations, batch]`.
    pub(super) geometry: [u32; 4],
}

impl WindowMeta {
    pub(super) fn new<const S: usize>(
        source: WindowLayoutMeta,
        target: WindowLayoutMeta,
        destination: WindowLayoutMeta,
        plan: &WindowPlan<S>,
    ) -> Result<Self> {
        if !(1..=SPATIAL_RANK).contains(&S) {
            return Err(invalid(format!(
                "window spatial rank {S} exceeds C-family rank {SPATIAL_RANK}"
            )));
        }
        Ok(Self {
            source,
            target,
            destination,
            kernel: parameters(plan.parameters.kernel(), "kernel")?,
            stride: parameters(plan.parameters.stride(), "stride")?,
            padding: parameters(plan.parameters.padding(), "padding")?,
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

fn parameters<const S: usize>(values: &[usize; S], name: &str) -> Result<[u32; SPATIAL_RANK]> {
    if S > SPATIAL_RANK {
        return Err(invalid(format!(
            "window spatial rank {S} exceeds metadata rank {SPATIAL_RANK}"
        )));
    }
    let mut output = [0_u32; SPATIAL_RANK];
    for (axis, &value) in values.iter().enumerate() {
        output[axis] = u32::try_from(value)
            .map_err(|_| invalid(format!("window {name}[{axis}] exceeds u32 range")))?;
    }
    Ok(output)
}

fn invalid(message: impl Into<String>) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: message.into(),
    }
}

const _: () = assert!(core::mem::size_of::<WindowLayoutMeta>() == 48);
const _: () = assert!(core::mem::size_of::<WindowMeta>() == 220);
