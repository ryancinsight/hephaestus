//! Test module for this operation family, split out of the parent
//! source file to keep it under the 500-line conformance budget.

    use super::kernel_source;

    #[test]
    fn source_contains_complete_pivot_stages() {
        let source = kernel_source();
        assert!(source.contains(
            "full_piv_lu_validate(\n    const float* matrix,\n    unsigned int* row_perm,\n    unsigned int* col_perm,\n    unsigned int* status,\n    unsigned int* rank,\n    float* threshold,\n    FullPivLuMeta meta"
        ));
        assert!(source.contains("full_piv_lu_step"));
        assert!(source.contains("pivot_row"));
        assert!(source.contains("pivot_col"));
        assert!(source.contains("threshold[0]"));
        assert!(source.contains("row_perm[index] = index"));
        assert!(source.contains("col_perm[index] = index"));
        assert!(source.contains("rank[0] = meta.n"));
    }
