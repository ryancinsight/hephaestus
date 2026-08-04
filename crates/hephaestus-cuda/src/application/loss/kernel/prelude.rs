pub(super) fn prelude() -> &'static str {
    r#"
struct LayoutMeta {
    long long shape[2];
    long long strides[2];
    long long offset;
};

struct ForwardMeta {
    LayoutMeta logits;
    LayoutMeta targets;
    LayoutMeta loss;
    LayoutMeta probabilities;
};

struct BackwardMeta {
    LayoutMeta output_gradient;
    LayoutMeta probabilities;
    LayoutMeta targets;
    LayoutMeta logit_gradient;
    float tolerance;
};

extern "C" __device__ __forceinline__ long long physical1(
    const LayoutMeta layout,
    const long long index
) {
    return layout.offset + index * layout.strides[0];
}

extern "C" __device__ __forceinline__ long long physical2(
    const LayoutMeta layout,
    const long long row,
    const long long column
) {
    return layout.offset + row * layout.strides[0] + column * layout.strides[1];
}

extern "C" __device__ __forceinline__ void cross_entropy_fail(
    unsigned int* status,
    const unsigned int code
) {
    atomicMin(status, code);
}
"#
}
