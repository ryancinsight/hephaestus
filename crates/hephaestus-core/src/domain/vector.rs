//! Dense rank-one vector operations over device-resident buffers.

use eunomia::Pod;

use super::device::ComputeDevice;
use super::dialect::{CudaC, DialectScalar, HipC, KernelDialect, Wgsl};
use super::error::Result;

/// Backend-neutral dense vector operations for iterative solvers.
///
/// # Why this seam exists
///
/// Each backend crate already provides `dot`, `norm_l2`, and the elementwise
/// families, but as free functions taking that backend's concrete device type,
/// and the elementwise family rejects an aliased output so it cannot express an
/// in-place update at all. [`ComputeDevice`] covers allocation and transfer
/// only. A consumer generic over the device could therefore reach no vector
/// operation, and had to bind to one backend or hand-write kernels of its own.
/// This trait closes that gap: a Krylov solver, a preconditioner, or any other
/// vector-recurrence consumer programs against `V: DenseVectorOps<D, T>` and
/// runs on every backend implementing it.
///
/// # Shape
///
/// `Self` is a backend-provided bundle of prepared kernels, constructed once
/// against a device, and the device is passed per call. This mirrors
/// [`super::kernel::MultiStorageKernel`] and keeps kernel compilation out of the
/// operation path — preparing a shader per call would dominate every operation
/// it performs.
///
/// # Contract
///
/// All operands are dense, contiguous, rank-one buffers of equal length; an
/// implementation returns [`crate::HephaestusError::LengthMismatch`] otherwise.
/// Operations naming a `target` write it in place, which is what keeps a solver
/// iteration free of scratch allocation and of round trips through host memory.
/// The binary elementwise methods write into caller-owned output buffers so
/// they preserve the same allocation-free shape as the Leto `Scalar` slice
/// operations.
/// A zero-length operand is a no-op, not an error.
///
/// # Prepared reductions
///
/// `dot` and `norm_l2` reduce a whole vector to one scalar the host must read to
/// test convergence. Preparing a reduction binds its intermediate storage and
/// dispatch resources to a fixed operand allocation once, so a solver reusing
/// the same buffers across iterations allocates nothing after setup. A prepared
/// handle is valid only for the allocations it was prepared against;
/// implementations reject a mismatched operand rather than reading the wrong
/// memory.
///
/// # Special values
///
/// Floating-point NaN and infinity behaviour follows the kernel
/// dialect's declared capability: see
/// [`KernelDialect::IEEE_SPECIAL_VALUES`]
/// (ADR 0043) for what is and is not promised per dialect.
pub trait DenseVectorOps<D: ComputeDevice, T: Pod> {
    /// Prepared resources for a dot product over a fixed operand pair.
    type PreparedDot<'a>
    where
        Self: 'a;
    /// Prepared resources for a Euclidean norm over a fixed operand.
    type PreparedNorm<'a>
    where
        Self: 'a;

    /// Copy `source` into `target` device-to-device.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend transfer failure.
    fn copy_vector(&self, device: &D, source: &D::Buffer<T>, target: &D::Buffer<T>) -> Result<()>;

    /// Scale `target` in place by `factor`.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn scale_vector(&self, device: &D, target: &D::Buffer<T>, factor: T) -> Result<()>;

    /// Apply `target += factor * source` in place.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend dispatch failure.
    fn axpy(
        &self,
        device: &D,
        target: &D::Buffer<T>,
        source: &D::Buffer<T>,
        factor: T,
    ) -> Result<()>;

    /// Apply `target = source + factor * target` in place.
    ///
    /// The companion of [`Self::axpy`] with the scaling on the accumulator
    /// rather than the increment. Krylov direction recurrences take this shape,
    /// and one kernel keeps it to a single traversal where a scale followed by
    /// an axpy would take two.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend dispatch failure.
    fn xpay(
        &self,
        device: &D,
        target: &D::Buffer<T>,
        source: &D::Buffer<T>,
        factor: T,
    ) -> Result<()>;

    /// Compute `output = left - right` into distinct caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, an aliased output, or the backend dispatch
    /// failure.
    fn subtract_into(
        &self,
        device: &D,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
        output: &D::Buffer<T>,
    ) -> Result<()>;

    /// Compute `output = left + right` into distinct caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, an aliased output, or the backend dispatch
    /// failure.
    fn add_into(
        &self,
        device: &D,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
        output: &D::Buffer<T>,
    ) -> Result<()>;

    /// Compute `output = left * right` into distinct caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, an aliased output, or the backend dispatch
    /// failure.
    fn multiply_into(
        &self,
        device: &D,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
        output: &D::Buffer<T>,
    ) -> Result<()>;

    /// Compute `output = left / right` into distinct caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, an aliased output, or the backend dispatch
    /// failure.
    fn divide_into(
        &self,
        device: &D,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
        output: &D::Buffer<T>,
    ) -> Result<()>;

    /// Prepare a dot product bound to `left` and `right`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend preparation failure.
    fn prepare_dot<'a>(
        &self,
        device: &D,
        left: &'a D::Buffer<T>,
        right: &'a D::Buffer<T>,
    ) -> Result<Self::PreparedDot<'a>>;

    /// Execute a prepared dot product and read the scalar back to the host.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn dot_prepared<'a>(
        &self,
        device: &D,
        prepared: &Self::PreparedDot<'a>,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
    ) -> Result<T>;

    /// Prepare a Euclidean norm bound to `vector`.
    ///
    /// # Errors
    ///
    /// Returns the backend preparation failure.
    fn prepare_norm_l2<'a>(
        &self,
        device: &D,
        vector: &'a D::Buffer<T>,
    ) -> Result<Self::PreparedNorm<'a>>;

    /// Execute a prepared Euclidean norm and read the scalar back to the host.
    ///
    /// # Errors
    ///
    /// Returns a prepared-operand mismatch or the backend dispatch failure.
    fn norm_l2_prepared<'a>(
        &self,
        device: &D,
        prepared: &Self::PreparedNorm<'a>,
        vector: &D::Buffer<T>,
    ) -> Result<T>;

    /// Compute a one-shot dot product.
    ///
    /// # Errors
    ///
    /// Returns the backend preparation or dispatch failure.
    fn dot(&self, device: &D, left: &D::Buffer<T>, right: &D::Buffer<T>) -> Result<T> {
        let prepared = self.prepare_dot(device, left, right)?;
        self.dot_prepared(device, &prepared, left, right)
    }

    /// Compute a one-shot Euclidean norm.
    ///
    /// # Errors
    ///
    /// Returns the backend preparation or dispatch failure.
    fn norm_l2(&self, device: &D, vector: &D::Buffer<T>) -> Result<T> {
        let prepared = self.prepare_norm_l2(device, vector)?;
        self.norm_l2_prepared(device, &prepared, vector)
    }

    /// Compute a one-shot L1 norm `Σ|xᵢ|`.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn norm_l1(&self, device: &D, vector: &D::Buffer<T>) -> Result<T>;

    /// Compute a one-shot max norm `maxᵢ|xᵢ|`.
    ///
    /// # Errors
    ///
    /// Returns the backend dispatch failure.
    fn norm_max(&self, device: &D, vector: &D::Buffer<T>) -> Result<T>;
}

/// Scalar representable in the dense-vector kernels of dialect `L`.
///
/// This is the compiler-enforced scalar matrix for [`DenseVectorOps`]: a
/// backend implements the seam for `T` exactly when `T: VectorScalar<L>`
/// (plus the reduction-identity bounds each operation needs). Integers are
/// deliberately absent: the host seam bounds `T` by leto's `RealScalar`
/// (floats only), and the norm family has no integer meaning there, so a
/// GPU integer implementation would diverge from the reference rather than
/// match it.
///
/// # Backend × scalar matrix
///
/// | Backend | Scalars | `f64` availability |
/// |---|---|---|
/// | Host (CPU reference) | every `T: Pod + RealScalar` | always; GPU results are asserted against it |
/// | CUDA | `f32`, `f64` | always; no feature gate |
/// | ROCm | `f32`, `f64` | tests are Linux-gated, so Windows checks compilation only |
/// | WGSL | `f32`, `f64` | needs the `ShaderF64` device feature, else the case skips |
/// | Metal | `f32` on hardware | compiles through the shared WGSL sources but always skips: Apple targets cannot offer `ShaderF64` |
///
/// Coverage is differential: the shared `hephaestus-conformance` suite
/// asserts every GPU result against the host reference per scalar, so each
/// row holds only where its tests ran on hardware.
pub trait VectorScalar<L: KernelDialect>: DialectScalar<L> {
    /// Spell `factor * multiplicand + addend` in this dialect for this scalar.
    ///
    /// Called once per kernel build, never on a dispatch path. The `f32`
    /// spellings preserve each backend's existing fused instruction; other
    /// scalars use whatever the dialect guarantees for them.
    #[must_use]
    fn madd_expr(factor: &str, multiplicand: &str, addend: &str) -> String;
}

impl VectorScalar<CudaC> for f32 {
    fn madd_expr(factor: &str, multiplicand: &str, addend: &str) -> String {
        format!("fmaf({factor}, {multiplicand}, {addend})")
    }
}

impl VectorScalar<CudaC> for f64 {
    fn madd_expr(factor: &str, multiplicand: &str, addend: &str) -> String {
        format!("fma({factor}, {multiplicand}, {addend})")
    }
}

impl VectorScalar<HipC> for f32 {
    fn madd_expr(factor: &str, multiplicand: &str, addend: &str) -> String {
        format!("fmaf({factor}, {multiplicand}, {addend})")
    }
}

impl VectorScalar<HipC> for f64 {
    fn madd_expr(factor: &str, multiplicand: &str, addend: &str) -> String {
        format!("fma({factor}, {multiplicand}, {addend})")
    }
}

impl VectorScalar<Wgsl> for f32 {
    fn madd_expr(factor: &str, multiplicand: &str, addend: &str) -> String {
        format!("fma({factor}, {multiplicand}, {addend})")
    }
}

impl VectorScalar<Wgsl> for f64 {
    fn madd_expr(factor: &str, multiplicand: &str, addend: &str) -> String {
        // Plain multiply-add: WGSL's `fma` is only guaranteed for `f32`, and
        // this spelling matches the host evaluation order exactly.
        format!("({factor} * {multiplicand} + {addend})")
    }
}

/// Scalar whose factor rides a sixteen-byte `u32`-word uniform block in dialect `L`.
///
/// Dialects where every scalar can appear directly in a uniform block (CUDA,
/// HIP) do not need this: they widen the factor instead. WGSL needs it
/// because an `f64` field would force the `FLOAT64` shader capability onto
/// every kernel, including `f32` ones, so each scalar declares its own
/// uniform layout within its own capability: `f32` reconstructs the factor
/// from exact words with a shader-side `bitcast`, while `f64` — whose
/// shader already requires `FLOAT64` — reads it as a native field. (A
/// shader-side `bitcast<f64>` from `u32` words is not an option: naga
/// rejects every such spelling.)
///
/// Every declaration describes the same sixteen host bytes: `len` at offset
/// 0, then the factor's exact bits at the offsets the declaration names.
/// [`Self::uniform_block`] and [`Self::PARAMS_DECL`] live in the same impl
/// so the words and the layout they are read under cannot drift apart.
pub trait VectorFactor<L: KernelDialect>: DialectScalar<L> {
    /// The `Params` uniform-block declaration for this scalar's kernels.
    const PARAMS_DECL: &'static str;

    /// The exact sixteen-byte uniform block: word 0 echoes `len` (already
    /// range-checked by the caller) so each impl reads as the complete
    /// memory image, and the remaining words place the factor's exact bits
    /// at the offsets [`Self::PARAMS_DECL`] names.
    #[must_use]
    fn uniform_block(len: u32, factor: Self) -> [u32; 4];

    /// Spell the factor load from the `params` uniform value in this dialect.
    ///
    /// Called once per kernel build, never on a dispatch path. Each spelling
    /// reconstructs the factor bit-exactly within its own scalar's shader
    /// capability: no `f64` value appears in an `f32` shader.
    #[must_use]
    fn factor_expr(params: &str) -> String;
}

impl VectorFactor<Wgsl> for f32 {
    const PARAMS_DECL: &'static str =
        "struct Params {\n    len: u32,\n    lo: u32,\n    hi: u32,\n    _pad: u32,\n}";

    fn uniform_block(len: u32, factor: Self) -> [u32; 4] {
        [len, factor.to_bits(), 0, 0]
    }

    fn factor_expr(params: &str) -> String {
        format!("bitcast<f32>({params}.lo)")
    }
}

impl VectorFactor<Wgsl> for f64 {
    // `factor` sits at offset 8 with 8-byte alignment, so the struct is
    // exactly sixteen bytes: the same image the host words below describe.
    const PARAMS_DECL: &'static str =
        "struct Params {\n    len: u32,\n    _pad: u32,\n    factor: f64,\n}";

    fn uniform_block(len: u32, factor: Self) -> [u32; 4] {
        let bits = factor.to_bits();
        // Bit shifts, not byte order: exact on every target. The words land
        // at bytes 8..16, which is where `PARAMS_DECL` reads the `f64`.
        [len, 0, bits as u32, (bits >> 32) as u32]
    }

    fn factor_expr(params: &str) -> String {
        format!("{params}.factor")
    }
}

/// Reduction resources a consumer can **retain** across iterations.
///
/// [`DenseVectorOps::prepare_dot`] hands back a handle borrowing its operands,
/// which suits a caller that prepares and dispatches inside one scope. An
/// iterative solver cannot use that shape: it stores its reductions beside the
/// vectors they measure for the lifetime of a workspace, and every iteration
/// mutates the very vector a handle would borrow. Holding the borrow across
/// that mutation does not borrow-check, and moving the handle to a narrower
/// scope does not help, because the conflict is the borrow itself rather than
/// where it lives.
///
/// This trait supplies the same reductions with **owned** handles, so a
/// consumer may store one for as long as it likes and present the operands
/// again at dispatch. Implementations that bake operand bindings into the
/// handle keep the operands alive by cloning their buffer handles; a backend
/// whose buffers are owned allocations rather than shared handles cannot do
/// that and simply does not implement this trait, keeping the borrowing form
/// from [`DenseVectorOps`].
///
/// A retained handle is valid only for the allocations it was built against;
/// implementations reject a mismatched operand rather than reducing the wrong
/// memory.
pub trait RetainedReductions<D: ComputeDevice, T: Pod>: DenseVectorOps<D, T> {
    /// Owned dot-product resources.
    type RetainedDot;
    /// Owned Euclidean-norm resources.
    type RetainedNorm;

    /// Build retained dot-product resources bound to `left` and `right`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend preparation failure.
    fn retain_dot(
        &self,
        device: &D,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
    ) -> Result<Self::RetainedDot>;

    /// Execute retained dot-product resources and read the scalar to the host.
    ///
    /// # Errors
    ///
    /// Returns a retained-operand mismatch or the backend dispatch failure.
    fn dot_retained(
        &self,
        device: &D,
        retained: &Self::RetainedDot,
        left: &D::Buffer<T>,
        right: &D::Buffer<T>,
    ) -> Result<T>;

    /// Build retained Euclidean-norm resources bound to `vector`.
    ///
    /// # Errors
    ///
    /// Returns the backend preparation failure.
    fn retain_norm_l2(&self, device: &D, vector: &D::Buffer<T>) -> Result<Self::RetainedNorm>;

    /// Execute retained norm resources and read the scalar to the host.
    ///
    /// # Errors
    ///
    /// Returns a retained-operand mismatch or the backend dispatch failure.
    fn norm_l2_retained(
        &self,
        device: &D,
        retained: &Self::RetainedNorm,
        vector: &D::Buffer<T>,
    ) -> Result<T>;
}
