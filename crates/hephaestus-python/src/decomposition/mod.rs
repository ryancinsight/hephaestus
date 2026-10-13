//! Matrix factorisations: Cholesky, LU (partial and full pivoting), QR
//! (plain and column-pivoted), Hessenberg, bidiagonalisation, and
//! Bunch-Kaufman. Factor-splitting math lives in the backends; this module
//! only marshals buffers. Both GPU backends split packed LU factors on-device,
//! so **L** and **U** remain resident without host staging.

mod orthogonal;
mod pivoted;
mod symmetric;

#[cfg(test)]
mod tests;

pub(crate) use orthogonal::{bidiagonalize, col_piv_qr, hessenberg, qr};
pub(crate) use pivoted::{full_piv_lu, lu};
pub(crate) use symmetric::{bunch_kaufman, cholesky};
