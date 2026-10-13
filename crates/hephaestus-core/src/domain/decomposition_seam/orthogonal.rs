use crate::domain::device::ComputeDevice;

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
