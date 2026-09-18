//! Leto as a two-dimensional Laplacian stencil implementor (ADR 0046).
//!
//! [`HostStencilOps`] rebuilds leto's `Laplacian2D` contract from the
//! parameter block's signed inverse squared spacings exactly, through
//! `Laplacian2D::from_signed_inverse_spacing_squared`, and applies leto-ops'
//! `laplacian_2d_into`. No spacing passes through a square root, so the host
//! applies the coefficients the device kernels read from the same block.

use hephaestus_core::{Laplacian2DParams, Result, StencilOps};
use leto::{ArrayView, ArrayViewMut, Laplacian2D, Layout};

use crate::operands::require_disjoint_output;
use crate::{HostBuffer, HostDevice, map_leto_error};

/// Two-dimensional Laplacian stencil for the host reference device.
#[derive(Clone, Copy, Debug, Default)]
pub struct HostStencilOps;

impl StencilOps<HostDevice> for HostStencilOps {
    /// The host compiles nothing; the stencil is rebuilt from each call's
    /// parameter block.
    type Laplacian2D = ();

    fn prepare_laplacian_2d(&self, _device: &HostDevice) -> Result<Self::Laplacian2D> {
        Ok(())
    }

    fn laplacian_2d_into(
        &self,
        _device: &HostDevice,
        _kernel: &Self::Laplacian2D,
        input: &HostBuffer<f32>,
        output: &HostBuffer<f32>,
        params: &Laplacian2DParams,
    ) -> Result<()> {
        require_disjoint_output(input, input, output)?;
        let (nx, ny, boundary, _) = params.contract();
        let stencil = Laplacian2D::from_signed_inverse_spacing_squared(
            nx,
            ny,
            [params.inv2[0], params.inv2[1]],
            boundary,
        )
        .map_err(map_leto_error)?;
        let input_cells = input.read();
        let mut output_cells = output.write();
        params.validate_storage(input_cells.len(), output_cells.len())?;
        let layout = Layout::c_contiguous([input_cells.len()]).map_err(map_leto_error)?;
        let input_view = ArrayView::try_new(layout, &input_cells).map_err(map_leto_error)?;
        let mut output_view =
            ArrayViewMut::try_new(layout, &mut output_cells).map_err(map_leto_error)?;
        leto_ops::laplacian_2d_into(&stencil, &input_view, &mut output_view).map_err(map_leto_error)
    }
}
