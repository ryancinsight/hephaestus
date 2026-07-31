pub(super) fn prelude() -> &'static str {
    r#"
typedef struct {
    long long shape[3];
    long long strides[3];
    long long offset;
} LayoutMeta;

typedef struct {
    LayoutMeta query;
    LayoutMeta key;
    LayoutMeta value;
    LayoutMeta output;
    LayoutMeta weights;
    LayoutMeta keep;
    long long heads_per_batch;
    int causal;
    int keep_present;
} ForwardMeta;

typedef struct {
    LayoutMeta grad_output;
    LayoutMeta query;
    LayoutMeta key;
    LayoutMeta value;
    LayoutMeta weights;
    LayoutMeta target;
} BackwardMeta;

extern "C" __device__ __forceinline__ long long physical3(
    const LayoutMeta layout,
    const long long first,
    const long long second,
    const long long third
) {
    return layout.offset + first * layout.strides[0] +
        second * layout.strides[1] + third * layout.strides[2];
}

extern "C" __device__ __forceinline__ long long physical2(
    const LayoutMeta layout,
    const long long first,
    const long long second
) {
    return layout.offset + first * layout.strides[0] + second * layout.strides[1];
}
"#
}
