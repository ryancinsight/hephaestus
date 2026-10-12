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

/// Oracle-minimal accessors on a real Schur form `A = Q·T·Qᵀ`
/// (ADR 0042 staging, stage 4).
pub trait SchurHandle<D: ComputeDevice> {
    /// Factored dimension `n`.
    fn order(&self) -> usize;

    /// Device-resident orthogonal factor `Q`, `n × n` row-major.
    fn q_buffer(&self) -> &D::Buffer<f32>;

    /// Device-resident quasi-upper factor `T`, `n × n` row-major
    /// (1×1/2×2 diagonal blocks; complex pairs stay as 2×2 blocks).
    fn t_buffer(&self) -> &D::Buffer<f32>;
}

/// Oracle-minimal accessors on a Hessenberg reduction `A = Q·H·Qᵀ`
/// (ADR 0042 staging, stage 4).
pub trait HessenbergHandle<D: ComputeDevice> {
    /// Factored dimension `n`.
    fn order(&self) -> usize;

    /// Device-resident orthogonal factor `Q`, `n × n` row-major.
    fn q_buffer(&self) -> &D::Buffer<f32>;

    /// Device-resident upper-Hessenberg factor `H`, `n × n` row-major
    /// (zero below the first subdiagonal).
    fn h_buffer(&self) -> &D::Buffer<f32>;
}

/// Oracle-minimal accessors on a bidiagonal reduction `A = U·B·Vᵀ`
/// (ADR 0042 staging, stage 4).
pub trait BidiagonalHandle<D: ComputeDevice> {
    /// `(rows, cols)` of the factored matrix.
    fn shape(&self) -> (usize, usize);

    /// Device-resident left orthogonal factor `U`, row-major.
    fn u_buffer(&self) -> &D::Buffer<f32>;

    /// Device-resident bidiagonal core `B`, row-major (main diagonal and
    /// the superdiagonal only; storage extent is the backend's, length
    /// discriminates as with QR's `R`).
    fn b_buffer(&self) -> &D::Buffer<f32>;

    /// Device-resident right orthogonal factor `V`, row-major with vector
    /// `k` in column `k`.
    fn v_buffer(&self) -> &D::Buffer<f32>;
}
