use crate::domain::device::ComputeDevice;
use crate::domain::error::Result;
/// Oracle-minimal accessors on a column-pivoted QR result (ADR 0042
/// staging, stage 1).
pub trait ColPivQrHandle<D: ComputeDevice> {
    /// `(rows, cols)` of the factored matrix.
    fn shape(&self) -> (usize, usize);

    /// Numerical rank revealed by the pivoted factorization.
    fn rank(&self) -> usize;

    /// Column permutation as a permutation vector (leto's convention; the
    /// conformance clause pins the gather direction against the host
    /// reference).
    fn permutation(&self) -> &[usize];

    /// Solve `min ‖A·x − rhs‖₂`.
    ///
    /// # Errors
    ///
    /// Returns a length mismatch against the factored shape or the
    /// backend dispatch failure.
    fn solve_least_squares(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}

/// Oracle-minimal accessors on a fully pivoted LU result (ADR 0042
/// staging, stage 1).
pub trait FullPivLuHandle<D: ComputeDevice> {
    /// Factored dimension `n`.
    fn order(&self) -> usize;

    /// Numerical rank revealed by full pivoting.
    fn rank(&self) -> usize;

    /// Determinant accumulated during elimination.
    fn det(&self) -> f32;

    /// Device-resident packed `L`/`U` factors, `n × n` row-major (unit
    /// lower diagonal implicit), of the fully pivoted elimination.
    fn factors(&self) -> &D::Buffer<f32>;

    /// Row permutation vector (leto's convention).
    fn row_permutation(&self) -> &[usize];

    /// Column permutation vector (leto's convention).
    fn col_permutation(&self) -> &[usize];

    /// Solve `A·x = rhs` for a full-rank factorization.
    ///
    /// # Errors
    ///
    /// Returns a rank-deficiency rejection, a length mismatch, or the
    /// backend dispatch failure.
    fn solve(&self, device: &D, rhs: &D::Buffer<f32>) -> Result<D::Buffer<f32>>;
}
