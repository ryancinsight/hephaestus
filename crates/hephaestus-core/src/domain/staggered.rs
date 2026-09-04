//! Backend-neutral parameters for the three-dimensional staggered
//! gradient/divergence pair.
//!
//! # What this is for
//!
//! A Yee leapfrog needs two operators that are negative adjoints, `D = -Gᵀ`,
//! or it has no conserved discrete energy. Leto owns that pair on the CPU as
//! `leto_ops::StaggeredLeapfrog3D`; this is the device side of the same
//! contract, so one FDTD sweep reaches either backend.
//!
//! # Why the taps travel in the parameter block
//!
//! The coefficients of an order-`2N` staggered stencil come from solving a
//! Taylor system. Deriving them per dispatch would put a linear solve inside a
//! timestep, so they are derived once by the caller — through
//! `leto_ops::staggered_first_derivative_coefficients`, the same provider
//! function the CPU path uses, which is what makes the two paths the same
//! stencil rather than two stencils that happen to agree — and ride to the
//! device in the uniform block.
//!
//! They are a parameter rather than something this crate derives because
//! `hephaestus-core` carries Leto's layout vocabulary and no CPU compute
//! dependency (atlas ADR 0001). A linear solve here would be exactly the
//! dependency that boundary exists to keep out. The conformance suite closes
//! the gap the parameter opens: it asserts a device dispatch against the CPU
//! operator built at the same order, so taps that did not come from the
//! provider fail the contract.
//!
//! # Where this differs from the CPU path, and why
//!
//! Leto's reflection loops, so a stencil deeper than its axis still terminates.
//! The device kernels require `extent >= 2 * N` on the differentiated axis and
//! [`Staggered3DParams::new`](crate::Staggered3DParams::new) rejects anything smaller with a typed error. A
//! loop whose trip count depends on the data is the wrong shape for a GPU
//! kernel, and a grid thinner than the stencil it is being swept with is a
//! configuration error rather than a case worth serving on an accelerator.
//! Consumers that genuinely need such a grid run it on the CPU backend, where
//! the contract is unchanged.

use eunomia::{Pod, Zeroable};

use crate::{HephaestusError, Result};

/// Largest half-order the uniform block carries.
///
/// The taps ride in two four-lane vectors, which is what lets a shader index
/// them without depending on a scalar-array stride rule. Eight tap pairs is
/// order sixteen — past any order a wave solver profitably runs, and past the
/// range in which the provider's derivation stays well conditioned.
pub const MAX_HALF_ORDER: usize = 8;

/// Which spatial axis a dispatch differentiates along.
///
/// The lane order matches the row-major `[nx, ny, nz]` shape the buffers carry:
/// `X` is the outermost axis, `Z` the contiguous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaggeredAxis {
    /// The outermost (slowest-varying) axis.
    X,
    /// The middle axis.
    Y,
    /// The innermost (contiguous) axis.
    Z,
}

impl StaggeredAxis {
    /// Position of this axis in a `[nx, ny, nz]` shape.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Uniform parameters shared by every device staggered dispatch.
///
/// The representation is four 32-bit lanes for the grid and axis, four for the
/// stencil half-order, four float lanes for the reciprocal spacings, and two
/// four-lane vectors carrying the derived taps `c_1 … c_8` zero-padded. It is
/// suitable for direct uniform-block upload, and its vector-of-four grouping is
/// what lets a shader index the taps without a scalar-array stride rule.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Staggered3DParams {
    /// `(nx, ny, nz, axis_index)`.
    pub dims_axis: [u32; 4],
    /// `(half_order, 0, 0, 0)`.
    pub order: [u32; 4],
    /// `(1/dx, 1/dy, 1/dz, 0)`; a dispatch reads only its own axis.
    pub inv_spacing: [f32; 4],
    /// Derived taps `c_1 … c_4`.
    pub taps_low: [f32; 4],
    /// Derived taps `c_5 … c_8`, zero beyond the half-order.
    pub taps_high: [f32; 4],
}

impl Staggered3DParams {
    /// Derive the taps and build a validated parameter block.
    ///
    /// `taps` are the derived staggered coefficients `c_1 … c_N` from
    /// `leto_ops::staggered_first_derivative_coefficients`; their count sets
    /// the half-order, so the accuracy order is `2 * taps.len()`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when `taps` is empty,
    /// longer than [`MAX_HALF_ORDER`], or carries a non-finite value; when a
    /// spacing is not finite and positive; when an axis is empty or the
    /// flattened grid overflows `usize`; or when the differentiated axis is
    /// shorter than the stencil it would be swept with (`extent < 2 * N`).
    pub fn new(
        nx: u32,
        ny: u32,
        nz: u32,
        axis: StaggeredAxis,
        taps: &[f32],
        spacing: [f32; 3],
    ) -> Result<Self> {
        let half_order = taps.len();
        if half_order == 0 || half_order > MAX_HALF_ORDER {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "staggered taps must number 1..={MAX_HALF_ORDER}, got {half_order}"
                ),
            });
        }
        if let Some(tap) = taps.iter().find(|tap| !tap.is_finite()) {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("staggered taps must all be finite, found {tap}"),
            });
        }
        let order = 2 * half_order;

        let dims = [nx, ny, nz];
        for (index, extent) in dims.iter().enumerate() {
            if *extent == 0 {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!("staggered grid axis {index} is empty: dims={dims:?}"),
                });
            }
        }
        let axis_extent = dims[axis.index()];
        let required =
            u32::try_from(order).map_err(|error| HephaestusError::InvalidConfiguration {
                message: format!("staggered order does not fit u32: order={order}, {error}"),
            })?;
        if axis_extent < required {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "the device staggered kernels need at least {required} cells on the \
                     differentiated axis for order {order}, got {axis_extent}; run a thinner \
                     grid on the CPU backend, whose reflection folds repeatedly"
                ),
            });
        }

        let mut inv_spacing = [0.0_f32; 4];
        for (lane, (value, name)) in inv_spacing
            .iter_mut()
            .zip(spacing.iter().zip(["dx", "dy", "dz"]))
        {
            if !value.is_finite() || *value <= 0.0 {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!("staggered {name} must be finite and positive, got {value}"),
                });
            }
            *lane = 1.0 / *value;
        }

        let mut lanes = [0.0_f32; 2 * 4];
        for (lane, tap) in lanes.iter_mut().zip(taps) {
            *lane = *tap;
        }
        let (low, high) = lanes.split_at(4);

        let half_order =
            u32::try_from(half_order).map_err(|error| HephaestusError::InvalidConfiguration {
                message: format!("staggered half-order does not fit u32: {error}"),
            })?;
        let axis_index =
            u32::try_from(axis.index()).map_err(|error| HephaestusError::InvalidConfiguration {
                message: format!("staggered axis index does not fit u32: {error}"),
            })?;

        let params = Self {
            dims_axis: [nx, ny, nz, axis_index],
            order: [half_order, 0, 0, 0],
            inv_spacing,
            taps_low: low.try_into().expect("invariant: split_at(4) yields four"),
            taps_high: high.try_into().expect("invariant: split_at(4) leaves four"),
        };
        params.cell_count()?;
        Ok(params)
    }

    /// Flattened cell count, or the overflow that prevents one.
    ///
    /// Named for the quantity rather than as `len`, because this is a grid
    /// descriptor and not a collection: there is no empty case to ask about,
    /// since an empty axis is rejected at construction.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when the product does
    /// not fit `usize`.
    pub fn cell_count(&self) -> Result<usize> {
        let mut total = 1_usize;
        for extent in &self.dims_axis[..3] {
            let extent = usize::try_from(*extent).map_err(|error| {
                HephaestusError::InvalidConfiguration {
                    message: format!("staggered grid extent does not fit usize: {error}"),
                }
            })?;
            total =
                total
                    .checked_mul(extent)
                    .ok_or_else(|| HephaestusError::InvalidConfiguration {
                        message: format!(
                            "staggered grid size overflows usize: dims={:?}",
                            &self.dims_axis[..3]
                        ),
                    })?;
        }
        Ok(total)
    }

    /// Confirm both buffers hold exactly the grid.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::LengthMismatch`] when either length
    /// differs from the grid, and the overflow from [`Self::cell_count`].
    pub fn validate_storage(&self, input_len: usize, output_len: usize) -> Result<()> {
        let expected = self.cell_count()?;
        for actual in [input_len, output_len] {
            if actual != expected {
                return Err(HephaestusError::LengthMismatch {
                    host_len: actual,
                    device_len: expected,
                });
            }
        }
        Ok(())
    }

    /// The stencil half-order `N`: the number of tap pairs, and the halo the
    /// sweep reads on either side.
    #[must_use]
    pub const fn half_order(&self) -> u32 {
        self.order[0]
    }

    /// The differentiated axis.
    #[must_use]
    pub const fn axis(&self) -> StaggeredAxis {
        match self.dims_axis[3] {
            0 => StaggeredAxis::X,
            1 => StaggeredAxis::Y,
            _ => StaggeredAxis::Z,
        }
    }
}

/// Device-neutral dispatch of the three-dimensional staggered pair.
///
/// # Why this is separate from [`StencilOps`]
///
/// A backend that has no staggered kernels should not be able to claim this
/// capability, and folding these methods into [`StencilOps`] would force every
/// backend to supply bodies — a body that returns zeros or an error is a mock
/// wearing a trait impl. Consumers bind whichever seam they need.
///
/// The operand scalar is fixed at `f32` by [`Staggered3DParams`]'s lane layout,
/// for the same reason the 2-D Laplacian is: WGSL does not guarantee `f64`
/// storage, so a generic scalar here would be a falsely generic boundary.
///
/// [`StencilOps`]: crate::StencilOps
pub trait Staggered3DOps<D: crate::ComputeDevice> {
    /// Compiled staggered gradient/divergence kernel pair, reusable across
    /// dispatches.
    type Staggered3D;

    /// Compile the staggered kernels for a device.
    ///
    /// # Errors
    ///
    /// Returns the backend's kernel compilation or layout failure.
    fn prepare_staggered_3d(&self, device: &D) -> Result<Self::Staggered3D>;

    /// Gradient along the axis in `params`: cell-centred `input` to
    /// face-centred `output`, face `i+½` stored at index `i`, with taps outside
    /// the grid reflected about the wall.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the grid, or the backend
    /// dispatch failure.
    fn staggered_gradient_into(
        &self,
        device: &D,
        kernel: &Self::Staggered3D,
        input: &D::Buffer<f32>,
        output: &D::Buffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()>;

    /// Divergence along the axis in `params`: face-centred `input` back to
    /// cell-centred `output`. This is `-Gᵀ` of
    /// [`Self::staggered_gradient_into`].
    ///
    /// # Errors
    ///
    /// See [`Self::staggered_gradient_into`].
    fn staggered_divergence_into(
        &self,
        device: &D,
        kernel: &Self::Staggered3D,
        input: &D::Buffer<f32>,
        output: &D::Buffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()>;
}
