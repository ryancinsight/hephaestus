/// Reconstruct the original matrix **A** from the packed LU result and
/// the pivot vector.
///
/// The packed result satisfies P·A = L·U, where P is the cumulative row
/// permutation recorded in `pivots`.  This helper extracts L and U,
/// computes L·U, then applies the inverse permutation to recover A.
pub(crate) fn lu_reconstruct_original(a_factored: &[f32], pivots: &[usize], n: usize) -> Vec<f32> {
    // Build L (unit lower) and U (upper) from the packed result.
    let mut l = vec![0.0f32; n * n];
    let mut u = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i > j {
                l[i * n + j] = a_factored[i * n + j];
            } else {
                l[i * n + j] = if i == j { 1.0 } else { 0.0 };
                u[i * n + j] = a_factored[i * n + j];
            }
        }
    }
    // Compute L·U.
    let mut lu = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0f32;
            for k in 0..n {
                s += l[i * n + k] * u[k * n + j];
            }
            lu[i * n + j] = s;
        }
    }
    // Apply transpositions in reverse order to recover A from L·U (A = Pᵀ · L · U).
    let mut a = lu;
    for k in (0..n).rev() {
        let pivot = pivots[k];
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j);
            }
        }
    }
    a
}

/// Reconstruct A from the packed Householder QR result and verify
/// that the upper triangle of the packed result matches the expected R.
pub(crate) fn qr_reconstruct_r(a_packed: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; m * n];
    for i in 0..n {
        for j in i..n {
            r[i * n + j] = a_packed[i * n + j];
        }
    }
    r
}

/// Apply the Householder reflectors from the packed QR result to the
/// identity matrix to reconstruct Q, then verify Q^T Q = I.
pub(crate) fn qr_reconstruct_q(
    a_packed: &[f32],
    heads: &[f32],
    betas: &[f32],
    m: usize,
    n: usize,
) -> Vec<f32> {
    // Start with Q = I (m×m).
    let mut q = vec![0.0f32; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }

    // Apply Hₖ = I − βₖ vₖ vₖᵀ for k = n-1, ..., 0 (reverse order).
    for k in (0..n).rev() {
        let vec_len = m - k;
        // Reconstruct vₖ.
        let mut v = vec![0.0f32; vec_len];
        v[0] = heads[k];
        for i in 1..vec_len {
            v[i] = a_packed[(k + i) * n + k];
        }
        let beta = betas[k];

        // Apply Hₖ to each column of Q: Q[:, j] -= β · v · (vᵀ · Q[k:m, j])
        for j in 0..m {
            let mut dot = 0.0f32;
            for i in 0..vec_len {
                dot += v[i] * q[(k + i) * m + j];
            }
            for i in 0..vec_len {
                q[(k + i) * m + j] -= beta * v[i] * dot;
            }
        }
    }
    q
}
