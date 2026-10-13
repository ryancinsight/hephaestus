use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;
/// Oracle-minimal accessors on a Bunch–Kaufman factorization of a
/// symmetric indefinite matrix (ADR 0042 staging, stage 5).
pub trait BunchKaufmanHandle<D: ComputeDevice> {
    /// Factored dimension `n`.
    fn order(&self) -> usize;

    /// Device-resident unit-lower factor `L`, `n × n` row-major.
    fn l_buffer(&self) -> &D::Buffer<f32>;

    /// Device-resident block-diagonal factor `D`: either dense `n × n`
    /// row-major or a packed diagonal of length `n` — the buffer length
    /// discriminates, and the conformance clause reconstructs through
    /// whichever the backend stores.
    fn d_buffer(&self) -> &D::Buffer<f32>;

    /// Symmetric permutation vector (applied to both sides).
    fn permutation(&self) -> &[usize];
}

/// Oracle-minimal accessors on a `U·D·Uᵀ` factorization of a symmetric
/// matrix (ADR 0042 staging, stage 5).
pub trait UduHandle<D: ComputeDevice> {
    /// Factored dimension `n`.
    fn order(&self) -> usize;

    /// Device-resident unit-upper factor `U`, `n × n` row-major.
    fn u_buffer(&self) -> &D::Buffer<f32>;

    /// Device-resident diagonal factor `D` as a length-`n` vector.
    fn d_buffer(&self) -> &D::Buffer<f32>;

    /// Determinant `∏ D[i]`.
    fn det(&self) -> f32;

    /// Solve `A·x = rhs`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend dispatch failure.
    fn solve(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}
