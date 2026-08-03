pub(super) fn prelude() -> &'static str {
    r#"
typedef struct {
    int shape[3];
    int strides[3];
    int offset;
    int rank;
} LayoutMeta;

typedef struct {
    LayoutMeta query;
    LayoutMeta key;
    LayoutMeta value;
    LayoutMeta output;
    LayoutMeta weights;
    LayoutMeta grad_output;
    LayoutMeta destination;
    LayoutMeta keep;
    int heads_per_batch;
    int causal;
    int has_keep;
} AttentionMeta;

extern "C" __device__ __forceinline__ int physical3(
    const LayoutMeta layout,
    const int first,
    const int second,
    const int third
) {
    return layout.offset + first * layout.strides[0] +
        second * layout.strides[1] + third * layout.strides[2];
}

extern "C" __device__ __forceinline__ int physical2(
    const LayoutMeta layout,
    const int first,
    const int second
) {
    return layout.offset + first * layout.strides[0] + second * layout.strides[1];
}
"#
}
