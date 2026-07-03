use crate::domain::device::ComputeDevice;
use crate::domain::error::{HephaestusError, Result};
use bytemuck::Pod;

/// Three-dimensional compute dispatch grid in backend workgroups.
///
/// The grid stores workgroup counts rather than element counts. Frontends
/// derive these counts from problem shape and backend-specific tile shape before
/// dispatch; the kernel contract receives only the launch shape every backend
/// can honor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DispatchGrid {
    /// Workgroups in x.
    pub x: u32,
    /// Workgroups in y.
    pub y: u32,
    /// Workgroups in z.
    pub z: u32,
}

impl DispatchGrid {
    /// Construct a dispatch grid from explicit workgroup counts.
    #[must_use]
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Derive a dispatch grid from a 3-D element domain and non-zero workgroup
    /// tile shape.
    ///
    /// # Errors
    /// Returns [`HephaestusError::DispatchFailed`] if any tile dimension is
    /// zero or if a computed workgroup count does not fit in `u32`.
    #[inline]
    pub fn covering_domain(domain: [usize; 3], workgroup_shape: [usize; 3]) -> Result<Self> {
        fn groups(axis: usize, tile: usize, name: &str) -> Result<u32> {
            if tile == 0 {
                return Err(HephaestusError::DispatchFailed {
                    message: format!("{name} workgroup tile is zero"),
                });
            }
            let groups = axis.div_ceil(tile);
            u32::try_from(groups).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("{name} workgroup count {groups} exceeds u32::MAX"),
            })
        }

        Ok(Self {
            x: groups(domain[0], workgroup_shape[0], "x")?,
            y: groups(domain[1], workgroup_shape[1], "y")?,
            z: groups(domain[2], workgroup_shape[2], "z")?,
        })
    }
}

/// Backend-neutral single-input/single-output storage-kernel dispatch.
///
/// This is the common contract for kernels whose device interface is one typed
/// input storage buffer, one typed output storage buffer, and one POD parameter
/// block. Backend crates provide concrete shader/module representations; the
/// consumer stays generic over the device, typed buffers, and dispatch grid.
pub trait UnaryStorageKernel<D, T, P>
where
    D: ComputeDevice,
    T: Pod,
    P: Pod,
{
    /// Dispatch the kernel.
    ///
    /// # Errors
    /// Returns the backend's typed dispatch failure if command construction,
    /// submission, or device execution fails.
    fn dispatch(
        &self,
        device: &D,
        input: &D::Buffer<T>,
        output: &D::Buffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()>;
}

/// Backend-neutral two-input/single-output storage-kernel dispatch.
///
/// This is the common contract for stencil and binary-field kernels whose
/// device interface is two typed input storage buffers, one typed output
/// storage buffer, and one POD parameter block. Backend crates provide concrete
/// shader/module representations; consumers stay generic over the device,
/// typed buffers, and dispatch grid.
pub trait BinaryStorageKernel<D, T, P>
where
    D: ComputeDevice,
    T: Pod,
    P: Pod,
{
    /// Dispatch the kernel.
    ///
    /// # Errors
    /// Returns the backend's typed dispatch failure if command construction,
    /// submission, or device execution fails.
    fn dispatch(
        &self,
        device: &D,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
        output: &D::Buffer<T>,
        params: &P,
        grid: DispatchGrid,
    ) -> Result<()>;
}

/// Backend-neutral multi-storage-buffer kernel dispatch.
///
/// This contract covers kernels whose storage interface is wider than the
/// unary/binary forms. The binding bundle `B` is backend-defined so WGPU, CUDA,
/// Metal, and future backends can expose the binding representation their launch
/// API requires without leaking that representation into `hephaestus-core`.
/// Consumers remain generic over the device, the POD parameter block, and the
/// kernel trait; backend crates provide concrete binding-bundle types.
pub trait MultiStorageKernel<D, P, B>
where
    D: ComputeDevice,
    P: Pod,
{
    /// Dispatch the kernel.
    ///
    /// # Errors
    /// Returns the backend's typed dispatch failure if bindings do not match
    /// the compiled layout, command construction fails, or submission fails.
    fn dispatch(&self, device: &D, bindings: B, params: &P, grid: DispatchGrid) -> Result<()>;
}

/// Compute device support for backend-defined multi-storage binding bundles.
///
/// [`MultiStorageKernel`] keeps the binding representation backend-defined so
/// WGPU can use bind-group storage entries while CUDA can use flat launch
/// arguments. This trait gives generic consumers one constructor for that
/// representation from typed provider buffers.
pub trait MultiStorageDevice: ComputeDevice {
    /// Backend-specific storage binding handle.
    type StorageBinding<'a>: Copy
    where
        Self: 'a;

    /// Bind a typed device buffer to a storage slot.
    #[must_use]
    fn storage_binding<T: Pod>(binding: u32, buffer: &Self::Buffer<T>) -> Self::StorageBinding<'_>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covering_domain_uses_ceil_division() {
        let grid = DispatchGrid::covering_domain([9, 17, 5], [8, 8, 4]).unwrap();
        assert_eq!(grid, DispatchGrid::new(2, 3, 2));
    }

    #[test]
    fn covering_domain_rejects_zero_tile() {
        let err = DispatchGrid::covering_domain([1, 1, 1], [1, 0, 1]).unwrap_err();
        assert!(matches!(err, HephaestusError::DispatchFailed { .. }));
    }
}
