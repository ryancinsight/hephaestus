pub(super) const fn source() -> &'static str {
    r#"
#include <float.h>
#include <math.h>

typedef struct {
    int shape;
    int stride;
    int offset;
} Layout1Meta;

typedef struct {
    int rows;
    int columns;
    int row_stride;
    int column_stride;
    int offset;
} Layout2Meta;

typedef struct {
    Layout2Meta logits;
    Layout1Meta targets;
    Layout1Meta loss;
    Layout2Meta probabilities;
    Layout1Meta output_gradient;
    Layout2Meta logit_gradient;
    int batch;
    int classes;
    float probability_tolerance;
} CrossEntropyMeta;

extern "C" __device__ __forceinline__ int physical1(Layout1Meta layout, int index) {
    return layout.offset + index * layout.stride;
}

extern "C" __device__ __forceinline__ int physical2(
    Layout2Meta layout,
    int row,
    int column
) {
    return layout.offset + row * layout.row_stride + column * layout.column_stride;
}

extern "C" __device__ __forceinline__ void record_status(
    unsigned int* status,
    unsigned int code
) {
    atomicMin(status, code);
}
"#
}
