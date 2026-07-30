pub(super) fn prelude() -> &'static str {
    r#"
typedef struct {
    int shape[5];
    int strides[5];
    int offset;
    int rank;
} LayoutMeta;

typedef struct {
    LayoutMeta input;
    LayoutMeta weight;
    LayoutMeta output;
    LayoutMeta destination;
    int stride[3];
    int padding[3];
    int dilation[3];
} ConvolutionMeta;

extern "C" __device__ __forceinline__ int physical(
    const LayoutMeta descriptor,
    const int coordinates[5]
) {
    int position = descriptor.offset;
    for (int axis = 0; axis < descriptor.rank; ++axis) {
        position += coordinates[axis] * descriptor.strides[axis];
    }
    return position;
}

extern "C" __device__ __forceinline__ void decode_layout(
    int linear,
    const LayoutMeta descriptor,
    int coordinates[5]
) {
    for (int axis = 0; axis < 5; ++axis) {
        coordinates[axis] = 0;
    }
    for (int axis = descriptor.rank - 1; axis >= 0; --axis) {
        coordinates[axis] = linear % descriptor.shape[axis];
        linear /= descriptor.shape[axis];
    }
}

extern "C" __device__ __forceinline__ int spatial_elements(
    const LayoutMeta descriptor
) {
    int elements = 1;
    for (int axis = 2; axis < descriptor.rank; ++axis) {
        elements *= descriptor.shape[axis];
    }
    return elements;
}

extern "C" __device__ __forceinline__ void decode_spatial(
    int linear,
    const LayoutMeta descriptor,
    int coordinates[3]
) {
    for (int axis = 0; axis < 3; ++axis) {
        coordinates[axis] = 0;
    }
    for (int axis = descriptor.rank - 1; axis >= 2; --axis) {
        coordinates[axis - 2] = linear % descriptor.shape[axis];
        linear /= descriptor.shape[axis];
    }
}

extern "C" __device__ __forceinline__ void coordinates_with_spatial(
    int batch,
    int channel,
    const int spatial[3],
    int rank,
    int coordinates[5]
) {
    for (int axis = 0; axis < 5; ++axis) {
        coordinates[axis] = 0;
    }
    coordinates[0] = batch;
    coordinates[1] = channel;
    for (int axis = 2; axis < rank; ++axis) {
        coordinates[axis] = spatial[axis - 2];
    }
}
"#
}
