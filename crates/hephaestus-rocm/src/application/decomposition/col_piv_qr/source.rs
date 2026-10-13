use super::types::ColPivQrMeta;

pub(crate) fn kernel_source() -> String {
    r#"
struct ColPivQrMeta {
    unsigned int rows;
    unsigned int cols;
    unsigned int pivots;
    unsigned int k;
};

extern "C" __global__ void col_piv_qr_validate(
    const float* matrix,
    float* q,
    unsigned int* permutation,
    float* heads,
    float* betas,
    unsigned int* status,
    unsigned int* rank,
    float* threshold,
    ColPivQrMeta meta
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elements = meta.rows * meta.cols;
    unsigned int q_elements = meta.rows * meta.rows;
    if (index < q_elements) {
        unsigned int row = index / meta.rows;
        unsigned int column = index % meta.rows;
        q[index] = row == column ? 1.0f : 0.0f;
    }
    if (index < meta.cols) {
        permutation[index] = index;
    }
    if (index == 0u) {
        rank[0] = meta.pivots;
    }
    if (index < elements && !isfinite(matrix[index])) {
        atomicExch(status, 1u);
    }
}

extern "C" __global__ void col_piv_qr_step(
    float* matrix,
    float* q,
    unsigned int* permutation,
    float* heads,
    float* betas,
    unsigned int* status,
    unsigned int* rank,
    float* threshold,
    ColPivQrMeta meta
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u || status[0] != 0u || rank[0] != meta.pivots) {
        return;
    }

    unsigned int m = meta.rows;
    unsigned int n = meta.cols;
    unsigned int k = meta.k;
    if (k == 0u) {
        float reference_norm = 0.0f;
        for (unsigned int col = 0u; col < n; col++) {
            float norm_sq = 0.0f;
            for (unsigned int row = 0u; row < m; row++) {
                float value = matrix[row * n + col];
                norm_sq += value * value;
            }
            float norm = sqrtf(norm_sq);
            if (norm > reference_norm) {
                reference_norm = norm;
            }
        }
        threshold[0] = reference_norm / 1000000000000.0f;
    }

    unsigned int pivot_col = k;
    float pivot_norm_sq = 0.0f;
    for (unsigned int row = k; row < m; row++) {
        float value = matrix[row * n + k];
        pivot_norm_sq += value * value;
    }
    for (unsigned int col = k + 1u; col < n; col++) {
        float candidate_norm_sq = 0.0f;
        for (unsigned int row = k; row < m; row++) {
            float value = matrix[row * n + col];
            candidate_norm_sq += value * value;
        }
        if (candidate_norm_sq > pivot_norm_sq) {
            pivot_norm_sq = candidate_norm_sq;
            pivot_col = col;
        }
    }
    if (!isfinite(pivot_norm_sq) || sqrtf(pivot_norm_sq) <= threshold[0]) {
        rank[0] = k;
        return;
    }

    if (pivot_col != k) {
        for (unsigned int row = 0u; row < m; row++) {
            float value = matrix[row * n + k];
            matrix[row * n + k] = matrix[row * n + pivot_col];
            matrix[row * n + pivot_col] = value;
        }
        unsigned int value = permutation[k];
        permutation[k] = permutation[pivot_col];
        permutation[pivot_col] = value;
    }

    float norm = sqrtf(pivot_norm_sq);
    float pivot = matrix[k * n + k];
    float sign = pivot < 0.0f ? -1.0f : 1.0f;
    float alpha = -sign * norm;
    float head = pivot - alpha;
    float vector_norm_sq = head * head;
    for (unsigned int row = k + 1u; row < m; row++) {
        float value = matrix[row * n + k];
        vector_norm_sq += value * value;
    }
    float beta = 2.0f / vector_norm_sq;
    if (!isfinite(alpha) || !isfinite(head) || !isfinite(beta)) {
        atomicExch(status, 1u);
        return;
    }

    // Q ← Q H, using the reflector vector before the R column is cleared.
    for (unsigned int row = 0u; row < m; row++) {
        float dot = q[row * m + k] * head;
        for (unsigned int tail = k + 1u; tail < m; tail++) {
            dot += q[row * m + tail] * matrix[tail * n + k];
        }
        float scale = beta * dot;
        q[row * m + k] -= scale * head;
        for (unsigned int tail = k + 1u; tail < m; tail++) {
            q[row * m + tail] -= scale * matrix[tail * n + k];
        }
    }

    // R ← H R, including column k so its reflector tail is cleared.
    for (unsigned int col = k; col < n; col++) {
        float dot = head * matrix[k * n + col];
        for (unsigned int row = k + 1u; row < m; row++) {
            dot += matrix[row * n + k] * matrix[row * n + col];
        }
        float scale = beta * dot;
        float top = matrix[k * n + col] - scale * head;
        if (!isfinite(top)) {
            atomicExch(status, 1u);
            return;
        }
        matrix[k * n + col] = top;
        for (unsigned int row = k + 1u; row < m; row++) {
            float value = matrix[row * n + col] - scale * matrix[row * n + k];
            if (!isfinite(value)) {
                atomicExch(status, 1u);
                return;
            }
            matrix[row * n + col] = value;
        }
    }
    matrix[k * n + k] = alpha;
    heads[k] = head;
    betas[k] = beta;
}
"#
    .to_string()
}
