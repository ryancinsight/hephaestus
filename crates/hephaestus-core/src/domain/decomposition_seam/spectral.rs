use crate::domain::device::ComputeDevice;

/// Oracle-minimal accessors on a symmetric eigendecomposition result
/// (ADR 0042 staging, stage 2).
pub trait SymmetricEigenHandle<D: ComputeDevice> {
    /// Dimension of the factored symmetric matrix.
    fn order(&self) -> usize;

    /// Device-resident eigenvalues (ordering is the backend's Jacobi
    /// sweep output; the conformance clause asserts the value multiset,
    /// not an order).
    fn eigenvalues(&self) -> &D::Buffer<f32>;

    /// Device-resident eigenvectors, `n × n` row-major with eigenvector
    /// `k` in column `k`, paired index-for-index with `eigenvalues`.
    fn eigenvectors(&self) -> &D::Buffer<f32>;
}

/// Oracle-minimal accessors on a singular value decomposition result
/// (ADR 0042 staging, stage 3).
pub trait SvdHandle<D: ComputeDevice> {
    /// `(rows, cols)` of the factored matrix.
    fn shape(&self) -> (usize, usize);

    /// Device-resident left singular vectors, row-major with vector `k`
    /// in column `k`, paired index-for-index with `singular_values`.
    fn u(&self) -> &D::Buffer<f32>;

    /// Device-resident right singular vectors, row-major with vector `k`
    /// in column `k` (so reconstruction reads `V[j][k]`, not a stored
    /// transpose).
    fn v(&self) -> &D::Buffer<f32>;

    /// Device-resident singular values, non-negative, paired with the
    /// vector columns.
    fn singular_values(&self) -> &D::Buffer<f32>;
}
