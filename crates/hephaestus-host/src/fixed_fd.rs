//! Leto as the fixed-scheme three-dimensional sweep implementor (ADR 0046).
//!
//! [`HostFixedFdOps`] runs `leto_ops::FiniteDifference3D` directly, so the
//! host is the provider rather than a second implementation of its stencils.
//! The parameter block carries reciprocal scales, while the provider takes
//! spacings, so the host recovers `h = 1 / inv_h` per dispatch; that single
//! rounding round-trip is the only place the host can differ from a device
//! kernel reading the block's scales, and the conformance epsilon absorbs it.

use hephaestus_core::{
    FixedFd3DOps, FixedFd3DParams, FixedFd3DScheme, HephaestusError, Result, StaggeredAxis,
};
use leto::{ArrayView3, ArrayViewMut3, Layout};
use leto_ops::{Axis, FiniteDifference3D, FiniteDifference3DScheme};

use crate::operands::require_disjoint_output;
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Fixed-scheme sweeps for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostFixedFdOps;

fn map_scheme(scheme: FixedFd3DScheme) -> FiniteDifference3DScheme {
    match scheme {
        FixedFd3DScheme::CentralSecondOrder => FiniteDifference3DScheme::CentralSecondOrder,
        FixedFd3DScheme::CentralFourthOrder => FiniteDifference3DScheme::CentralFourthOrder,
        FixedFd3DScheme::CentralSixthOrder => FiniteDifference3DScheme::CentralSixthOrder,
        FixedFd3DScheme::StaggeredForward => FiniteDifference3DScheme::StaggeredForward,
        FixedFd3DScheme::StaggeredBackward => FiniteDifference3DScheme::StaggeredBackward,
    }
}

fn map_axis(axis: StaggeredAxis) -> Axis {
    match axis {
        StaggeredAxis::X => Axis::X,
        StaggeredAxis::Y => Axis::Y,
        StaggeredAxis::Z => Axis::Z,
    }
}

fn dense_layout(shape: [usize; 3]) -> Result<Layout<3>> {
    let [nx, ny, nz] = shape;
    let stride = |extent: usize| {
        isize::try_from(extent).map_err(|error| HephaestusError::InvalidConfiguration {
            message: format!("fixed-fd stride does not fit isize: {error}"),
        })
    };
    let layout = Layout::<3>::try_new([nx, ny, nz], [stride(ny * nz)?, stride(nz)?, 1], 0)
        .map_err(map_leto_error)?;
    Ok(layout)
}

impl FixedFd3DOps<HostDevice> for HostFixedFdOps {
    /// The host compiles nothing; the provider runs per dispatch.
    type FixedFd3D = ();

    fn prepare_fixed_fd_3d(&self, _device: &HostDevice) -> Result<Self::FixedFd3D> {
        Ok(())
    }

    fn fixed_fd_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::FixedFd3D,
        input: &HostBuffer<f32>,
        output: &HostBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        require_disjoint_output(input, input, output)?;
        let input_cells = input.read();
        let mut output_cells = output.write();
        params.validate_storage(input_cells.len(), output_cells.len())?;
        // The block's scales are exact reciprocals of positive spacings, so
        // the recovered spacing is positive and finite. Only the swept
        // axis's spacing reaches the sweep; the block carries that lane.
        let h = 1.0 / params.scales[0];
        let operator = FiniteDifference3D::new(map_scheme(params.scheme()), h, h, h)
            .map_err(map_leto_error)?;
        let [nx, ny, nz] = params.output_dims().map(|extent| extent as usize);
        let [ix, iy, iz] = [
            params.dims_axis[0] as usize,
            params.dims_axis[1] as usize,
            params.dims_axis[2] as usize,
        ];
        let input_layout = dense_layout([ix, iy, iz])?;
        let output_layout = dense_layout([nx, ny, nz])?;
        let field = ArrayView3::try_new(input_layout, &input_cells).map_err(map_leto_error)?;
        let mut target =
            ArrayViewMut3::try_new(output_layout, &mut output_cells).map_err(map_leto_error)?;
        match map_axis(params.axis()) {
            Axis::X => operator.apply_x_into(field, &mut target),
            Axis::Y => operator.apply_y_into(field, &mut target),
            Axis::Z => operator.apply_z_into(field, &mut target),
        }
        .map_err(map_leto_error)
    }

    fn fixed_fd_adjoint_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::FixedFd3D,
        upstream: &HostBuffer<f32>,
        grad: &HostBuffer<f32>,
        params: &FixedFd3DParams,
    ) -> Result<()> {
        require_disjoint_output(upstream, upstream, grad)?;
        let upstream_cells = upstream.read();
        let mut grad_cells = grad.write();
        params.validate_adjoint_storage(upstream_cells.len(), grad_cells.len())?;
        let h = 1.0 / params.scales[0];
        let operator = FiniteDifference3D::new(map_scheme(params.scheme()), h, h, h)
            .map_err(map_leto_error)?;
        let [nx, ny, nz] = params.dims().map(|extent| extent as usize);
        let [ux, uy, uz] = params.output_dims().map(|extent| extent as usize);
        let upstream_layout = dense_layout([ux, uy, uz])?;
        let grad_layout = dense_layout([nx, ny, nz])?;
        let field =
            ArrayView3::try_new(upstream_layout, &upstream_cells).map_err(map_leto_error)?;
        let mut target =
            ArrayViewMut3::try_new(grad_layout, &mut grad_cells).map_err(map_leto_error)?;
        match map_axis(params.axis()) {
            Axis::X => operator.adjoint_x_into(field, &mut target),
            Axis::Y => operator.adjoint_y_into(field, &mut target),
            Axis::Z => operator.adjoint_z_into(field, &mut target),
        }
        .map_err(map_leto_error)
    }
}
