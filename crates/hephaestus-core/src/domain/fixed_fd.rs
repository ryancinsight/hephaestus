//! Backend-neutral parameters for fixed-scheme three-dimensional
//! first derivatives.
//!
//! # What this is for
//!
//! The central (second, fourth, sixth order) and Yee staggered
//! (forward/backward) families whose coefficients are fixed by the scheme
//! rather than derived per order. Leto owns these sweeps on the CPU as
//! `leto_ops::FiniteDifference3D`; this is the device side of the same
//! contract, so one sweep reaches either backend.
//!
//! # Where this differs from the staggered pair, and why it is separate
//!
//! [`Staggered3DOps`](crate::Staggered3DOps) carries derived taps in its
//! parameter block because an order-`2N` stencil comes from a Taylor solve.
//! These schemes need no taps: their coefficients are small integers baked
//! into each kernel. Folding five more dispatches into the staggered trait
//! would force every staggered backend to supply bodies for schemes it may
//! never serve — a body that returns an error is a mock wearing a trait
//! impl — so this seam stands alone and backends bind it when they have the
//! kernels.
//!
//! # Bit-exactness against the CPU path
//!
//! The kernels reproduce Leto's per-lane arithmetic in the same operation
//! order, including the boundary fall-back chain (fourth order degrades to
//! second, then first, at the walls; sixth degrades through fourth). The
//! reciprocal scales ride precomputed in the parameter block, evaluated on
//! the host in `f32` with the same parenthesization Leto uses
//! (`1.0 / (12.0 * h)`), so no shader-side reassociation can drift the
//! rounding. The conformance suite asserts a device dispatch against the CPU
//! operator lane by lane.
//!
//! # What is deliberately not here
//!
//! Leto's fused fourth-order `divergence_into` and the closure-fused
//! `map_axis_derivatives` family stay CPU-only: the former composes on a
//! device from three sweeps plus existing elementwise addition, and the
//! latter takes a Rust closure, which has no device spelling. The portable
//! core — the per-axis sweeps — is what this seam covers.

use eunomia::{Pod, Zeroable};

use crate::{HephaestusError, Result, StaggeredAxis};

/// Fixed-coefficient stencil family a dispatch sweeps.
///
/// The variant set mirrors `leto_ops::FiniteDifference3DScheme`; the seam
/// maps that enum onto this one rather than depending on the provider, since
/// `hephaestus-core` carries no CPU compute dependency (atlas ADR 0001).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FixedFd3DScheme {
    /// `(f[i+1] − f[i−1]) / 2Δ` interior, one-sided at the walls.
    CentralSecondOrder = 0,
    /// `(−f[i+2] + 8f[i+1] − 8f[i−1] + f[i−2]) / 12Δ` interior, degrading to
    /// second then first order toward the walls, flat on a singleton axis.
    CentralFourthOrder = 1,
    /// `(−f[i−3] + 9f[i−2] − 45f[i−1] + 45f[i+1] − 9f[i+2] + f[i+3]) / 60Δ`
    /// interior, degrading through fourth, second, then first order.
    CentralSixthOrder = 2,
    /// `(f[i+1] − f[i]) / Δ`; the output is one cell shorter on the axis.
    StaggeredForward = 3,
    /// `(f[i] − f[i−1]) / Δ`, with a forward fall-back at `i = 0`.
    StaggeredBackward = 4,
}

impl FixedFd3DScheme {
    /// Fewest points the differentiated axis needs for this scheme.
    ///
    /// Fourth order takes any non-empty axis (a singleton is flat); the rest
    /// match the provider's rejection thresholds exactly.
    #[must_use]
    pub const fn min_axis_extent(self) -> u32 {
        match self {
            Self::CentralSecondOrder => 3,
            Self::CentralFourthOrder => 1,
            Self::CentralSixthOrder => 7,
            Self::StaggeredForward | Self::StaggeredBackward => 2,
        }
    }

    /// Whether the output keeps the input shape on the differentiated axis.
    #[must_use]
    pub const fn preserves_shape(self) -> bool {
        !matches!(self, Self::StaggeredForward)
    }
}

/// Uniform parameters shared by every device fixed-scheme dispatch.
///
/// The representation is four 32-bit lanes for the grid and axis, four for
/// the scheme discriminant, and four float lanes carrying the host-evaluated
/// reciprocal scales `(1/h, 1/2h, 1/12h, 1/60h)` for the differentiated
/// axis. It is suitable for direct uniform-block upload.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct FixedFd3DParams {
    /// `(nx, ny, nz, axis_index)` of the input field.
    pub dims_axis: [u32; 4],
    /// `(scheme_discriminant, 0, 0, 0)`.
    pub scheme: [u32; 4],
    /// `(1/h, 1/2h, 1/12h, 1/60h)` for the differentiated axis.
    pub scales: [f32; 4],
}

impl FixedFd3DParams {
    /// Build a validated parameter block.
    ///
    /// `spacing` is `(dx, dy, dz)`; every lane must be finite and positive,
    /// matching the provider constructor, while only the differentiated
    /// axis's scales ride in the block.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when a spacing is
    /// not finite and positive; when an axis is empty or the flattened grid
    /// overflows `usize`; or when the differentiated axis is shorter than
    /// [`FixedFd3DScheme::min_axis_extent`] for the scheme.
    pub fn new(
        nx: u32,
        ny: u32,
        nz: u32,
        axis: StaggeredAxis,
        scheme: FixedFd3DScheme,
        spacing: [f32; 3],
    ) -> Result<Self> {
        let dims = [nx, ny, nz];
        for (index, extent) in dims.iter().enumerate() {
            if *extent == 0 {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd grid axis {index} is empty: dims={dims:?}"),
                });
            }
        }
        let axis_extent = dims[axis.index()];
        let required = scheme.min_axis_extent();
        if axis_extent < required {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "fixed-fd {scheme:?} needs at least {required} points on the \
                     differentiated axis, got {axis_extent}"
                ),
            });
        }

        for (value, name) in spacing.iter().zip(["dx", "dy", "dz"]) {
            if !value.is_finite() || *value <= 0.0 {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!("fixed-fd {name} must be finite and positive, got {value}"),
                });
            }
        }
        // The same parenthesization the provider uses, in f32: identical
        // IEEE operations round identically on host and device.
        let h = spacing[axis.index()];
        let scales = [1.0 / h, 1.0 / (2.0 * h), 1.0 / (12.0 * h), 1.0 / (60.0 * h)];

        let axis_index =
            u32::try_from(axis.index()).map_err(|error| HephaestusError::InvalidConfiguration {
                message: format!("fixed-fd axis index does not fit u32: {error}"),
            })?;
        let params = Self {
            dims_axis: [nx, ny, nz, axis_index],
            scheme: [scheme as u32, 0, 0, 0],
            scales,
        };
        params.cell_count()?;
        Ok(params)
    }

    /// Flattened input cell count, or the overflow that prevents one.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when the product
    /// does not fit `usize`.
    pub fn cell_count(&self) -> Result<usize> {
        grid_cells(&self.dims_axis[..3], "fixed-fd")
    }

    /// Flattened output cell count.
    ///
    /// Every scheme preserves the grid except [`FixedFd3DScheme::StaggeredForward`],
    /// which drops one plane on the differentiated axis.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when a product does
    /// not fit `usize`.
    pub fn output_cell_count(&self) -> Result<usize> {
        if self.scheme().preserves_shape() {
            return self.cell_count();
        }
        let mut dims = self.dims_axis;
        let axis = self.axis().index();
        dims[axis] -= 1;
        grid_cells(&dims[..3], "fixed-fd output")
    }

    /// Confirm the buffers hold exactly the input grid and the scheme's
    /// output grid.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::LengthMismatch`] when either length
    /// differs from its grid, and the overflow from [`Self::cell_count`].
    pub fn validate_storage(&self, input_len: usize, output_len: usize) -> Result<()> {
        let expected_in = self.cell_count()?;
        if input_len != expected_in {
            return Err(HephaestusError::LengthMismatch {
                host_len: input_len,
                device_len: expected_in,
            });
        }
        let expected_out = self.output_cell_count()?;
        if output_len != expected_out {
            return Err(HephaestusError::LengthMismatch {
                host_len: output_len,
                device_len: expected_out,
            });
        }
        Ok(())
    }

    /// Confirm the buffers hold the transpose sweep's grids: the upstream
    /// has the forward sweep's output shape, the gradient the forward
    /// sweep's input shape.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::LengthMismatch`] when either length
    /// differs from its grid, and the overflow from [`Self::cell_count`].
    pub fn validate_adjoint_storage(&self, upstream_len: usize, grad_len: usize) -> Result<()> {
        let expected_upstream = self.output_cell_count()?;
        if upstream_len != expected_upstream {
            return Err(HephaestusError::LengthMismatch {
                host_len: upstream_len,
                device_len: expected_upstream,
            });
        }
        let expected_grad = self.cell_count()?;
        if grad_len != expected_grad {
            return Err(HephaestusError::LengthMismatch {
                host_len: grad_len,
                device_len: expected_grad,
            });
        }
        Ok(())
    }

    /// Input grid extents `[nx, ny, nz]`.
    #[must_use]
    pub const fn dims(&self) -> [u32; 3] {
        [self.dims_axis[0], self.dims_axis[1], self.dims_axis[2]]
    }

    /// Output extents `[nx, ny, nz]`, shrunk on the axis for a forward sweep.
    #[must_use]
    pub fn output_dims(&self) -> [u32; 3] {
        let mut dims = [self.dims_axis[0], self.dims_axis[1], self.dims_axis[2]];
        if !self.scheme().preserves_shape() {
            dims[self.axis().index()] -= 1;
        }
        dims
    }

    /// The stencil family this block dispatches.
    #[must_use]
    pub const fn scheme(&self) -> FixedFd3DScheme {
        match self.scheme[0] {
            0 => FixedFd3DScheme::CentralSecondOrder,
            1 => FixedFd3DScheme::CentralFourthOrder,
            2 => FixedFd3DScheme::CentralSixthOrder,
            3 => FixedFd3DScheme::StaggeredForward,
            _ => FixedFd3DScheme::StaggeredBackward,
        }
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

fn grid_cells(dims: &[u32], role: &str) -> Result<usize> {
    let mut total = 1_usize;
    for extent in dims {
        let extent =
            usize::try_from(*extent).map_err(|error| HephaestusError::InvalidConfiguration {
                message: format!("{role} grid extent does not fit usize: {error}"),
            })?;
        total = total
            .checked_mul(extent)
            .ok_or_else(|| HephaestusError::InvalidConfiguration {
                message: format!("{role} grid size overflows usize: dims={dims:?}"),
            })?;
    }
    Ok(total)
}

/// Device-neutral dispatch of the fixed-scheme three-dimensional sweeps.
///
/// # Why this is separate from [`Staggered3DOps`](crate::Staggered3DOps)
///
/// A backend that has staggered kernels but no fixed-scheme kernels should
/// not be able to claim this capability, and folding these sweeps into the
/// staggered trait would force every backend to supply bodies — a body that
/// returns an error is a mock wearing a trait impl. Consumers bind whichever
/// seam they need.
///
/// The operand scalar is fixed at `f32` for the same reason the staggered
/// pair is: WGSL does not guarantee `f64` storage, so a generic scalar here
/// would be a falsely generic boundary.
pub trait FixedFd3DOps<D: crate::ComputeDevice> {
    /// Compiled fixed-scheme sweep kernel, reusable across dispatches.
    type FixedFd3D;

    /// Compile the sweep kernel for a device.
    ///
    /// # Errors
    ///
    /// Returns the backend's kernel compilation or layout failure.
    fn prepare_fixed_fd_3d(&self, device: &D) -> Result<Self::FixedFd3D>;

    /// Sweep the scheme in `params` along its axis, `input` into `output`.
    ///
    /// The output shape is [`FixedFd3DParams::output_dims`]: the input grid,
    /// except a [`FixedFd3DScheme::StaggeredForward`] sweep drops one plane
    /// on the differentiated axis.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against either grid, or the backend
    /// dispatch failure.
    fn fixed_fd_into(
        &self,
        device: &D,
        kernel: &Self::FixedFd3D,
        input: &D::Buffer<f32>,
        output: &D::Buffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()>;

    /// Sweep the transpose of the scheme in `params` along its axis,
    /// `upstream` into `grad`.
    ///
    /// For a forward sweep `y = A f`, this maps the upstream shaped like `y`
    /// to the gradient shaped like `f`: the pullback an autograd backward
    /// pass needs. A forward sweep shrinks the grid, so its adjoint fans
    /// back out; lanes accumulate the provider's predicated terms in
    /// canonical order, so CPU and device agree bit for bit in `f32`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against either grid, or the backend
    /// dispatch failure.
    fn fixed_fd_adjoint_into(
        &self,
        device: &D,
        kernel: &Self::FixedFd3D,
        upstream: &D::Buffer<f32>,
        grad: &D::Buffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMES: [FixedFd3DScheme; 5] = [
        FixedFd3DScheme::CentralSecondOrder,
        FixedFd3DScheme::CentralFourthOrder,
        FixedFd3DScheme::CentralSixthOrder,
        FixedFd3DScheme::StaggeredForward,
        FixedFd3DScheme::StaggeredBackward,
    ];

    #[test]
    fn minimum_extents_match_the_provider_thresholds() {
        let minima = [3, 1, 7, 2, 2];
        for (scheme, minimum) in SCHEMES.into_iter().zip(minima) {
            assert_eq!(scheme.min_axis_extent(), minimum, "{scheme:?}");
            for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
                let mut dims = [8, 8, 8];
                dims[axis.index()] = minimum - 1;
                if minimum == 1 {
                    continue;
                }
                let rejected =
                    FixedFd3DParams::new(dims[0], dims[1], dims[2], axis, scheme, [0.5, 0.5, 0.5]);
                assert!(
                    rejected.is_err(),
                    "{scheme:?} on {axis:?} accepts below minimum"
                );
                dims[axis.index()] = minimum;
                let accepted =
                    FixedFd3DParams::new(dims[0], dims[1], dims[2], axis, scheme, [0.5, 0.5, 0.5]);
                assert!(
                    accepted.is_ok(),
                    "{scheme:?} on {axis:?} rejects its minimum"
                );
            }
        }
    }

    #[test]
    fn scales_follow_the_provider_parenthesization() {
        let params = FixedFd3DParams::new(8, 8, 8, StaggeredAxis::Y, SCHEMES[1], [0.5, 0.25, 1.0])
            .expect("valid block");
        let h = 0.25_f32;
        assert_eq!(
            params.scales,
            [1.0 / h, 1.0 / (2.0 * h), 1.0 / (12.0 * h), 1.0 / (60.0 * h)]
        );
    }

    #[test]
    fn forward_shrinks_the_output_on_its_axis_only() {
        for axis in [StaggeredAxis::X, StaggeredAxis::Y, StaggeredAxis::Z] {
            let params =
                FixedFd3DParams::new(8, 6, 10, axis, FixedFd3DScheme::StaggeredForward, [0.5; 3])
                    .expect("valid block");
            let mut expected = [8, 6, 10];
            expected[axis.index()] -= 1;
            assert_eq!(params.output_dims(), expected);
            let cells: usize = expected.iter().map(|extent| *extent as usize).product();
            assert_eq!(params.output_cell_count().expect("count"), cells);
            assert!(params.validate_storage(8 * 6 * 10, cells).is_ok());
            assert!(params.validate_storage(8 * 6 * 10, 8 * 6 * 10).is_err());
        }
    }

    #[test]
    fn adjoint_storage_swaps_the_forward_grids() {
        let shrunk = FixedFd3DParams::new(
            8,
            6,
            10,
            StaggeredAxis::X,
            FixedFd3DScheme::StaggeredForward,
            [0.5; 3],
        )
        .expect("valid block");
        // The upstream has the forward output shape (one plane short), the
        // gradient the full input grid.
        assert!(
            shrunk
                .validate_adjoint_storage(7 * 6 * 10, 8 * 6 * 10)
                .is_ok()
        );
        assert!(
            shrunk
                .validate_adjoint_storage(8 * 6 * 10, 8 * 6 * 10)
                .is_err()
        );
        assert!(
            shrunk
                .validate_adjoint_storage(7 * 6 * 10, 7 * 6 * 10)
                .is_err()
        );

        let kept = FixedFd3DParams::new(
            8,
            8,
            8,
            StaggeredAxis::X,
            FixedFd3DScheme::CentralSixthOrder,
            [0.5; 3],
        )
        .expect("valid block");
        assert!(kept.validate_adjoint_storage(512, 512).is_ok());
        assert!(kept.validate_adjoint_storage(511, 512).is_err());
    }

    #[test]
    fn central_and_backward_preserve_the_grid() {
        for scheme in [
            FixedFd3DScheme::CentralSecondOrder,
            FixedFd3DScheme::CentralFourthOrder,
            FixedFd3DScheme::CentralSixthOrder,
            FixedFd3DScheme::StaggeredBackward,
        ] {
            let params = FixedFd3DParams::new(8, 8, 8, StaggeredAxis::X, scheme, [0.5; 3])
                .expect("valid block");
            assert_eq!(params.output_dims(), [8, 8, 8]);
            assert!(params.validate_storage(512, 512).is_ok());
            assert!(params.validate_storage(512, 511).is_err());
        }
    }
}
