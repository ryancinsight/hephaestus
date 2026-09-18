//! Leto as the three-dimensional staggered gradient/divergence implementor
//! (ADR 0046).
//!
//! [`HostStaggeredOps`] rebuilds leto-ops' `StaggeredLeapfrog3D` from the
//! parameter block's own taps and reciprocal spacings, through
//! `TapCoefficients::from_taps` and `StaggeredLeapfrog3D::from_parts`, so the
//! host applies exactly the coefficients a device kernel reads from the same
//! block rather than re-deriving them.

use hephaestus_core::{HephaestusError, Result, Staggered3DOps, Staggered3DParams, StaggeredAxis};
use leto::{ArrayView, ArrayViewMut, Layout};
use leto_ops::{Axis, StaggeredLeapfrog3D, TapCoefficients};

use crate::operands::require_disjoint_output;
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Staggered gradient and divergence for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostStaggeredOps;

/// Which half of the negative-adjoint pair a sweep applies.
#[derive(Clone, Copy)]
enum Sweep {
    Gradient,
    Divergence,
}

/// Rebuild the operator from the block's taps and reciprocal spacings.
fn operator(params: &Staggered3DParams) -> Result<StaggeredLeapfrog3D<f32>> {
    let mut taps = [0.0f32; 8];
    taps[..4].copy_from_slice(&params.taps_low);
    taps[4..].copy_from_slice(&params.taps_high);
    let half_order = params.half_order() as usize;
    let taps = taps
        .get(..half_order)
        .ok_or_else(|| HephaestusError::InvalidConfiguration {
            message: format!("staggered half-order {half_order} exceeds the eight carried taps"),
        })?;
    let coefficients = TapCoefficients::from_taps(taps).map_err(map_leto_error)?;
    let [x, y, z, _] = params.inv_spacing;
    StaggeredLeapfrog3D::from_parts(coefficients, [x, y, z]).map_err(map_leto_error)
}

fn sweep(
    input: &HostBuffer<f32>,
    output: &HostBuffer<f32>,
    params: &Staggered3DParams,
    direction: Sweep,
) -> Result<()> {
    require_disjoint_output(input, input, output)?;
    let operator = operator(params)?;
    let axis = match params.axis() {
        StaggeredAxis::X => Axis::X,
        StaggeredAxis::Y => Axis::Y,
        StaggeredAxis::Z => Axis::Z,
    };
    let input_cells = input.read();
    let mut output_cells = output.write();
    params.validate_storage(input_cells.len(), output_cells.len())?;
    let [nx, ny, nz, _] = params.dims_axis;
    let layout =
        Layout::c_contiguous([nx, ny, nz].map(|extent| extent as usize)).map_err(map_leto_error)?;
    let field = ArrayView::try_new(layout, &input_cells).map_err(map_leto_error)?;
    let mut target = ArrayViewMut::try_new(layout, &mut output_cells).map_err(map_leto_error)?;
    match direction {
        Sweep::Gradient => operator.gradient_into(axis, field, &mut target),
        Sweep::Divergence => operator.divergence_into(axis, field, &mut target),
    }
    .map_err(map_leto_error)
}

impl Staggered3DOps<HostDevice> for HostStaggeredOps {
    /// The host compiles nothing; the operator is rebuilt from each call's
    /// parameter block.
    type Staggered3D = ();

    fn prepare_staggered_3d(&self, _device: &HostDevice) -> Result<Self::Staggered3D> {
        Ok(())
    }

    fn staggered_gradient_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::Staggered3D,
        input: &HostBuffer<f32>,
        output: &HostBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        sweep(input, output, params, Sweep::Gradient)
    }

    fn staggered_divergence_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::Staggered3D,
        input: &HostBuffer<f32>,
        output: &HostBuffer<f32>,
        params: &Staggered3DParams,
    ) -> Result<()> {
        sweep(input, output, params, Sweep::Divergence)
    }
}
