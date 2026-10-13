use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;
/// Access to an LU factorization `P·A = L·U` of a square matrix.
pub trait LuHandle<D: ComputeDevice> {
    /// Matrix order `n`.
    fn order(&self) -> usize;
    /// Device-resident packed `L\U` factors, `n × n` row-major (unit
    /// diagonal of `L` implicit).
    fn factors(&self) -> &D::Buffer<f32>;
    /// Row permutation as a permutation vector: row `k` of `P·A` is row
    /// `pivots()[k]` of `A` (leto's pivot convention, shared by every
    /// backend's factorization machinery).
    fn pivots(&self) -> &[usize];
    /// Determinant of `A`, including the permutation sign.
    fn det(&self) -> f32;
    /// Solve `A·x = rhs` for one right-hand side.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, a singular factor, or the backend
    /// dispatch failure.
    fn solve(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Access to a Householder QR factorization `A = Q·R` with `m ≥ n`.
pub trait QrHandle<D: ComputeDevice> {
    /// Input shape `(m, n)`.
    fn shape(&self) -> (usize, usize);
    /// Device-resident upper-triangular `R`, `m × n` row-major (leto's
    /// convention — rows beyond `n` vanish). The Householder sign choice is
    /// the backend's own; only entry magnitudes and the triangular structure
    /// are contract.
    fn r_buffer(&self) -> &D::Buffer<f32>;
    /// Solve the least-squares problem `min ‖A·x − rhs‖₂`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch, a rank-deficient factor, or the backend
    /// dispatch failure.
    fn solve_least_squares(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Access to a Cholesky factorization `A = L·Lᵀ` of an SPD matrix.
pub trait CholeskyHandle<D: ComputeDevice> {
    /// Matrix order `n`.
    fn order(&self) -> usize;
    /// Device-resident lower-triangular factor `L`, `n × n` row-major.
    fn lower(&self) -> &D::Buffer<f32>;
    /// Determinant of `A`.
    fn det(&self) -> f32;
    /// Solve `A·x = rhs` for one right-hand side.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch or the backend dispatch failure.
    fn solve(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}
