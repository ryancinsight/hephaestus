use super::kernel_source;

#[test]
fn source_contains_column_pivot_stages() {
    let source = kernel_source();
    assert!(source.contains(
        "col_piv_qr_validate(\n    const float* matrix,\n    float* q,\n    unsigned int* permutation,\n    float* heads,\n    float* betas,\n    unsigned int* status,\n    unsigned int* rank,\n    float* threshold,\n    ColPivQrMeta meta"
    ));
    assert!(source.contains("col_piv_qr_step"));
    assert!(source.contains("pivot_norm_sq"));
    assert!(source.contains("permutation[k]"));
    assert!(source.contains("Q H"));
    assert!(source.contains("q[index] = row == column ? 1.0f : 0.0f"));
    assert!(source.contains("permutation[index] = index"));
    assert!(source.contains("rank[0] = meta.pivots"));
}
