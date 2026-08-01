//! Backend-neutral two-dimensional Laplacian stencil parameters.

use aequitas::systems::si::quantities::Length;
use bytemuck::{Pod, Zeroable};

use crate::{HephaestusError, Result};

pub use leto::{BoundaryCondition, LaplacianPolarity};

/// Uniform parameters shared by every device Laplacian implementation.
///
/// The representation is four 32-bit lanes for dimensions and boundary
/// selection followed by four 32-bit lanes for signed inverse squared
/// spacings. It is suitable for direct uniform/parameter-block upload.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Laplacian2DParams {
    /// `(nx, ny, boundary_condition, padding)`.
    pub dims_bc: [u32; 4],
    /// `(signed_dx^-2, signed_dy^-2, 0, 0)`.
    pub inv2: [f32; 4],
}

impl Laplacian2DParams {
    /// Build validated parameters for a uniform Cartesian grid.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when an axis is too
    /// short, the flattened grid size overflows, or a spacing is not finite
    /// and positive.
    pub fn new(
        nx: u32,
        ny: u32,
        dx: Length<f32>,
        dy: Length<f32>,
        boundary: BoundaryCondition,
        polarity: LaplacianPolarity,
    ) -> Result<Self> {
        let nx_usize =
            usize::try_from(nx).map_err(|error| HephaestusError::InvalidConfiguration {
                message: format!("Laplacian nx does not fit usize: nx={nx}, error={error}"),
            })?;
        let ny_usize =
            usize::try_from(ny).map_err(|error| HephaestusError::InvalidConfiguration {
                message: format!("Laplacian ny does not fit usize: ny={ny}, error={error}"),
            })?;
        let contract = leto::Laplacian2D::new(nx_usize, ny_usize, dx, dy, boundary)
            .map_err(|error| HephaestusError::InvalidConfiguration {
                message: error.to_string(),
            })?
            .with_polarity(polarity);
        let [dx_inv2, dy_inv2] = contract.signed_inverse_spacing_squared();
        Ok(Self {
            dims_bc: [nx, ny, u32::from(boundary), 0],
            inv2: [dx_inv2, dy_inv2, 0.0, 0.0],
        })
    }

    /// Number of x-axis points.
    #[must_use]
    pub const fn nx(self) -> usize {
        self.dims_bc[0] as usize
    }

    /// Number of y-axis points.
    #[must_use]
    pub const fn ny(self) -> usize {
        self.dims_bc[1] as usize
    }

    /// Required flattened storage length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.nx() * self.ny()
    }

    /// Whether the validated grid has no storage elements.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Validate input and output storage before a backend launch.
    pub fn validate_storage(self, input_len: usize, output_len: usize) -> Result<()> {
        if input_len != output_len {
            return Err(HephaestusError::LengthMismatch {
                host_len: input_len,
                device_len: output_len,
            });
        }
        let expected = self.len();
        if input_len != expected {
            return Err(HephaestusError::LengthMismatch {
                host_len: input_len,
                device_len: expected,
            });
        }
        Ok(())
    }

    /// Return the dimensions, boundary condition, and polarity encoded in the
    /// parameter block.
    #[must_use]
    pub const fn contract(self) -> (usize, usize, BoundaryCondition, LaplacianPolarity) {
        let boundary = match self.dims_bc[2] {
            0 => BoundaryCondition::Dirichlet,
            1 => BoundaryCondition::Neumann,
            _ => BoundaryCondition::Periodic,
        };
        let polarity = if self.inv2[0].is_sign_negative() {
            LaplacianPolarity::NegativeLaplacian
        } else {
            LaplacianPolarity::Laplacian
        };
        (self.nx(), self.ny(), boundary, polarity)
    }
}

/// Device-neutral 2D stencil dispatch.
///
/// Implementors are zero-sized per-backend markers; the prepared kernel is
/// the backend's compiled stencil pipeline, reusable across dispatches.
/// The operand scalar is fixed at `f32` by [`Laplacian2DParams`]'s lane
/// layout.
pub trait StencilOps<D: crate::ComputeDevice> {
    /// Compiled 2D Laplacian kernel, reusable across dispatches.
    type Laplacian2D;

    /// Compile the 2D Laplacian kernel for a device.
    ///
    /// # Errors
    ///
    /// Returns the backend's kernel compilation or layout failure.
    fn prepare_laplacian_2d(&self, device: &D) -> Result<Self::Laplacian2D>;

    /// Apply the 2D Laplacian stencil: `output = L(input)` under the grid,
    /// boundary, and polarity contract carried by `params`.
    ///
    /// # Errors
    ///
    /// Returns a storage-length mismatch against the grid, or the backend
    /// dispatch failure.
    fn laplacian_2d_into(
        &self,
        device: &D,
        kernel: &Self::Laplacian2D,
        input: &D::Buffer<f32>,
        output: &D::Buffer<f32>,
        params: &Laplacian2DParams,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use aequitas::systems::si::units::Meter;

    use super::*;

    #[test]
    fn validates_layout_and_storage() {
        let params = Laplacian2DParams::new(
            4,
            5,
            Length::from_unit::<Meter>(0.1),
            Length::from_unit::<Meter>(0.2),
            BoundaryCondition::Periodic,
            LaplacianPolarity::Laplacian,
        )
        .expect("valid grid");

        assert_eq!(params.dims_bc, [4, 5, 2, 0]);
        assert_eq!(params.len(), 20);
        assert_eq!(params.contract().2, BoundaryCondition::Periodic);
        params.validate_storage(20, 20).expect("matching storage");
    }

    #[test]
    fn rejects_invalid_grid_and_storage() {
        let error = Laplacian2DParams::new(
            1,
            4,
            Length::from_unit::<Meter>(1.0),
            Length::from_unit::<Meter>(1.0),
            BoundaryCondition::Dirichlet,
            LaplacianPolarity::Laplacian,
        )
        .expect_err("one-point axis is invalid");
        assert!(matches!(
            error,
            HephaestusError::InvalidConfiguration { .. }
        ));

        let params = Laplacian2DParams::new(
            4,
            4,
            Length::from_unit::<Meter>(1.0),
            Length::from_unit::<Meter>(1.0),
            BoundaryCondition::Dirichlet,
            LaplacianPolarity::Laplacian,
        )
        .expect("valid grid");
        assert!(matches!(
            params.validate_storage(15, 16),
            Err(HephaestusError::LengthMismatch { .. })
        ));
        assert!(matches!(
            params.validate_storage(16, 15),
            Err(HephaestusError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn negative_polarity_changes_coefficients() {
        let params = Laplacian2DParams::new(
            4,
            4,
            Length::from_unit::<Meter>(0.1),
            Length::from_unit::<Meter>(0.2),
            BoundaryCondition::Neumann,
            LaplacianPolarity::NegativeLaplacian,
        )
        .expect("valid grid");
        assert!(params.inv2[0].is_sign_negative());
        assert!(params.inv2[1].is_sign_negative());
        assert_eq!(params.contract().3, LaplacianPolarity::NegativeLaplacian);
    }
}
