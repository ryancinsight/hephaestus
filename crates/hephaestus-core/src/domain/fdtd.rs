//! Provider-neutral three-dimensional acoustic FDTD contracts.

use eunomia::{Pod, Zeroable};

use crate::{ComputeDevice, HephaestusError, Result};

/// One cell's collocated particle-velocity storage layout.
///
/// The fourth lane preserves the 16-byte alignment required by common GPU
/// storage layouts for a three-component vector.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct FdtdVelocity {
    /// Cartesian velocity components in metres per second.
    pub components: [f32; 3],
    /// Alignment padding; it is ignored by the kernel.
    pub padding: f32,
}

/// One cell's homogeneous or heterogeneous acoustic-medium properties.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct FdtdMedium {
    density: f32,
    sound_speed: f32,
}

impl FdtdMedium {
    /// Construct validated density and sound-speed properties.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] when either value is
    /// non-finite or not strictly positive.
    pub fn new(density: f32, sound_speed: f32) -> Result<Self> {
        if !density.is_finite() || density <= 0.0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("FDTD density must be finite and positive: {density}"),
            });
        }
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("FDTD sound speed must be finite and positive: {sound_speed}"),
            });
        }
        Ok(Self {
            density,
            sound_speed,
        })
    }

    /// Return the density in kilograms per cubic metre.
    #[must_use]
    pub const fn density(self) -> f32 {
        self.density
    }

    /// Return the sound speed in metres per second.
    #[must_use]
    pub const fn sound_speed(self) -> f32 {
        self.sound_speed
    }
}

/// Validated dimensions, spacing, and timestep for a collocated FDTD grid.
///
/// The provider contract is explicitly `f32`: WGPU storage kernels do not
/// guarantee `f64`, and a provider that cannot honor the precision must reject
/// acquisition rather than widen and narrow silently. The timestep remains
/// subject to the caller's acoustic CFL analysis for its medium and grid.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Fdtd3dParams {
    dimensions: [u32; 4],
    spacing_dt: [f32; 4],
}

impl Fdtd3dParams {
    /// Construct a validated three-dimensional uniform-grid contract.
    ///
    /// Each dimension must be at least three cells so the central-difference
    /// stencil has both neighbors. Spacings and timestep must be finite and
    /// strictly positive. The flattened storage length must fit `usize`.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::InvalidConfiguration`] for invalid geometry,
    /// timestep, or flattened storage size.
    pub fn new(nx: u32, ny: u32, nz: u32, dx: f32, dy: f32, dz: f32, dt: f32) -> Result<Self> {
        if nx < 3 || ny < 3 || nz < 3 {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("FDTD dimensions must be at least 3: ({nx}, {ny}, {nz})"),
            });
        }
        for (name, value) in [("dx", dx), ("dy", dy), ("dz", dz), ("dt", dt)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(HephaestusError::InvalidConfiguration {
                    message: format!("FDTD {name} must be finite and positive: {value}"),
                });
            }
        }
        let nx_len = usize::try_from(nx).map_err(|error| invalid_dimension("nx", nx, error))?;
        let ny_len = usize::try_from(ny).map_err(|error| invalid_dimension("ny", ny, error))?;
        let nz_len = usize::try_from(nz).map_err(|error| invalid_dimension("nz", nz, error))?;
        nx_len
            .checked_mul(ny_len)
            .and_then(|xy| xy.checked_mul(nz_len))
            .ok_or_else(|| HephaestusError::InvalidConfiguration {
                message: format!(
                    "FDTD flattened storage length overflows usize: ({nx}, {ny}, {nz})"
                ),
            })?;

        Ok(Self {
            dimensions: [nx, ny, nz, 0],
            spacing_dt: [dx, dy, dz, dt],
        })
    }

    /// Return the x-axis cell count.
    #[must_use]
    pub const fn nx(self) -> u32 {
        self.dimensions[0]
    }

    /// Return the y-axis cell count.
    #[must_use]
    pub const fn ny(self) -> u32 {
        self.dimensions[1]
    }

    /// Return the z-axis cell count.
    #[must_use]
    pub const fn nz(self) -> u32 {
        self.dimensions[2]
    }

    /// Return the x-axis spacing.
    #[must_use]
    pub const fn dx(self) -> f32 {
        self.spacing_dt[0]
    }

    /// Return the y-axis spacing.
    #[must_use]
    pub const fn dy(self) -> f32 {
        self.spacing_dt[1]
    }

    /// Return the z-axis spacing.
    #[must_use]
    pub const fn dz(self) -> f32 {
        self.spacing_dt[2]
    }

    /// Return the explicit Euler timestep.
    #[must_use]
    pub const fn dt(self) -> f32 {
        self.spacing_dt[3]
    }

    /// Return the flattened storage length after validating the encoded dimensions.
    pub fn storage_len(self) -> Result<usize> {
        let nx = usize::try_from(self.nx())
            .map_err(|error| invalid_dimension("nx", self.nx(), error))?;
        let ny = usize::try_from(self.ny())
            .map_err(|error| invalid_dimension("ny", self.ny(), error))?;
        let nz = usize::try_from(self.nz())
            .map_err(|error| invalid_dimension("nz", self.nz(), error))?;
        nx.checked_mul(ny)
            .and_then(|xy| xy.checked_mul(nz))
            .ok_or_else(|| HephaestusError::InvalidConfiguration {
                message: format!(
                    "FDTD flattened storage length overflows usize: ({}, {}, {})",
                    self.nx(),
                    self.ny(),
                    self.nz()
                ),
            })
    }

    /// Validate the pressure, velocity, and medium buffers against the grid.
    pub fn validate_storage(
        self,
        pressure_len: usize,
        velocity_len: usize,
        medium_len: usize,
    ) -> Result<()> {
        let expected = self.storage_len()?;
        for actual in [pressure_len, velocity_len, medium_len] {
            if actual != expected {
                return Err(HephaestusError::LengthMismatch {
                    host_len: actual,
                    device_len: expected,
                });
            }
        }
        Ok(())
    }
}

/// Provider-owned three-dimensional collocated acoustic FDTD operations.
///
/// Implementations own pipeline compilation and dispatch. Consumers own the
/// physical source, medium construction, CPU reference, and comparison
/// policy; they pass typed provider buffers through this seam.
pub trait Fdtd3dOps<D: ComputeDevice> {
    /// Prepared provider-specific FDTD pipeline bundle.
    type Kernel;

    /// Compile the FDTD kernels for a device.
    ///
    /// # Errors
    ///
    /// Returns a provider compilation or layout error.
    fn prepare_fdtd_3d(&self, device: &D) -> Result<Self::Kernel>;

    /// Enqueue one explicit Euler FDTD step in place.
    ///
    /// The velocity update is ordered before the pressure update. Completion
    /// follows the device's normal submission contract; callers that need a
    /// host-visible result call [`ComputeDevice::synchronize`] before download.
    ///
    /// # Errors
    ///
    /// Returns a storage mismatch or provider dispatch error.
    fn step_fdtd_3d(
        &self,
        device: &D,
        kernel: &Self::Kernel,
        pressure: &D::Buffer<f32>,
        velocity: &D::Buffer<FdtdVelocity>,
        medium: &D::Buffer<FdtdMedium>,
        params: &Fdtd3dParams,
    ) -> Result<()>;
}

fn invalid_dimension(name: &str, value: u32, error: impl core::fmt::Display) -> HephaestusError {
    HephaestusError::InvalidConfiguration {
        message: format!("FDTD {name} does not fit usize: {value}, error={error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_geometry_and_storage() {
        let params = Fdtd3dParams::new(4, 5, 6, 0.1, 0.2, 0.3, 1.0e-5).expect("valid grid");
        assert_eq!((params.nx(), params.ny(), params.nz()), (4, 5, 6));
        assert_eq!(params.storage_len().expect("valid storage"), 120);
        params
            .validate_storage(120, 120, 120)
            .expect("matching buffers");
        assert!(matches!(
            params.validate_storage(119, 120, 120),
            Err(HephaestusError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn rejects_invalid_geometry_and_medium() {
        assert!(matches!(
            Fdtd3dParams::new(2, 4, 4, 1.0, 1.0, 1.0, 1.0),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            Fdtd3dParams::new(4, 4, 4, 0.0, 1.0, 1.0, 1.0),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
        assert!(matches!(
            FdtdMedium::new(f32::NAN, 1500.0),
            Err(HephaestusError::InvalidConfiguration { .. })
        ));
        let medium = FdtdMedium::new(1000.0, 1500.0).expect("valid medium");
        assert_eq!((medium.density(), medium.sound_speed()), (1000.0, 1500.0));
    }
}
