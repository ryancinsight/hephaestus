//! The device-API seam separating vendor mechanics from generic orchestration.

use bytemuck::Pod;

use crate::domain::device::ComputeDevice;
use crate::domain::dialect::KernelDialect;
use crate::domain::error::Result;
use crate::domain::launch::BlockWidth;

/// Launch shape for a one-dimensional grid of fixed-width blocks.
///
/// `groups` blocks each run `width` lanes and receive `shared_bytes` of
/// dynamically sized shared memory. This is the geometry every op family in
/// [`super`] plans; richer shapes belong to the families that need them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchGeometry {
    /// Number of blocks in the launch.
    pub groups: u32,
    /// Lanes per block.
    pub width: BlockWidth,
    /// Dynamically sized shared memory per block, in bytes.
    pub shared_bytes: u32,
}

impl LaunchGeometry {
    /// A linear grid of `groups` blocks with no dynamic shared memory.
    #[must_use]
    #[inline]
    pub const fn linear(groups: u32, width: BlockWidth) -> Self {
        Self {
            groups,
            width,
            shared_bytes: 0,
        }
    }

    /// A linear grid of `groups` blocks with `shared_bytes` per block.
    #[must_use]
    #[inline]
    pub const fn linear_shared(groups: u32, width: BlockWidth, shared_bytes: u32) -> Self {
        Self {
            groups,
            width,
            shared_bytes,
        }
    }
}

/// A device that compiles kernel source at runtime and launches it against
/// raw device addresses.
///
/// This is the narrow per-vendor axis beneath the generic accelerator layer.
/// An implementor supplies four mechanics and nothing else: how a compiled
/// kernel is named and cached, how a typed buffer erases to a device address,
/// how two buffers are tested for aliasing, and how a parameter block plus an
/// ordered operand list becomes a native launch. Validation, planning, and
/// every host-side algorithm live once in [`super`].
///
/// The seam is deliberately open — no private supertrait — because a device
/// API is exactly the component a new vendor crate supplies. Its
/// [`Dialect`](Self::Dialect) is likewise open (see
/// [`KernelDialect`](crate::KernelDialect)).
///
/// # Scalar parameterization
///
/// [`device_ptr`](Self::device_ptr) and [`buffers_alias`](Self::buffers_alias)
/// are generic over `T: Pod` rather than fixed to one element type, so an op
/// family built on this seam is scalar-generic end to end.
pub trait DeviceApi: ComputeDevice {
    /// The kernel-source dialect this backend compiles.
    type Dialect: KernelDialect;

    /// A compiled, cached kernel handle. Cheap to clone — backends return a
    /// shared handle rather than recompiling.
    type Kernel: Clone;

    /// This device API's raw address representation (an integer address on
    /// CUDA, a pointer on HIP).
    type DevicePtr: Copy;

    /// The backend's pipeline-cache key.
    ///
    /// Left associated rather than fixed so each backend keeps one key enum
    /// covering all of its op families. A generic op family requires only
    /// that its own key converts into this one, expressed as a
    /// `Self::CacheKey: From<FamilyKey>` bound at the family's entry point
    /// rather than here — adding a family therefore does not widen this
    /// trait.
    type CacheKey;

    /// Fetch `entry` from the pipeline cache under `key`, compiling `source`
    /// on a miss.
    ///
    /// `source` is a closure so a cache hit never pays for source generation.
    /// Failed compilations are not cached.
    ///
    /// # Errors
    ///
    /// Returns the backend's compilation or module-load failure.
    fn compile_cached(
        &self,
        key: Self::CacheKey,
        entry: &str,
        source: impl FnOnce() -> String,
    ) -> Result<Self::Kernel>;

    /// Erase a typed buffer borrow to this device API's raw address.
    fn device_ptr<T: Pod>(buffer: &Self::Buffer<T>) -> Self::DevicePtr;

    /// Whether two buffers name the same device allocation.
    ///
    /// Used to reject in-place operand aliasing before dispatch.
    fn buffers_alias<T: Pod>(lhs: &Self::Buffer<T>, rhs: &Self::Buffer<T>) -> bool;

    /// Launch `kernel` over `geometry` with `params` followed by `operands`
    /// as its argument list.
    ///
    /// The argument order is the kernel signature's order: the parameter
    /// block first, then each operand address. Packing that list into the
    /// native calling convention is the implementor's responsibility, because
    /// it is the one genuinely ABI-specific step.
    ///
    /// # Errors
    ///
    /// Returns the backend's launch failure.
    fn launch<P: Pod>(
        &self,
        kernel: &Self::Kernel,
        geometry: LaunchGeometry,
        params: &P,
        operands: &[Self::DevicePtr],
    ) -> Result<()>;
}
