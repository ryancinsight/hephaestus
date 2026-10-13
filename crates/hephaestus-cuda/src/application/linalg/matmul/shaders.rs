use core::marker::PhantomData;

use hephaestus_core::{CudaC, DialectScalar};

pub(crate) struct MatmulKernel<T>(PhantomData<T>);
pub(crate) struct BatchedMatmulKernel<T>(PhantomData<T>);

pub(crate) fn matmul_shader_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}
struct MatrixLayout {{
    unsigned int shape[2];
    int strides[2];
    unsigned int offset;
}};

extern "C" __global__ void matmul_kernel(
    const {ty}* a,
    const {ty}* b,
    {ty}* c,
    MatrixLayout a_layout,
    MatrixLayout b_layout,
    MatrixLayout c_layout
) {{
    __shared__ {ty} A_shared[16][16];
    __shared__ {ty} B_shared[16][16];

    unsigned int row = blockIdx.y * 16u + threadIdx.y;
    unsigned int col = blockIdx.x * 16u + threadIdx.x;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;

    unsigned int m = a_layout.shape[0];
    unsigned int k = a_layout.shape[1];
    unsigned int n = b_layout.shape[1];

    int stride_a_row = a_layout.strides[0];
    int stride_a_col = a_layout.strides[1];
    int stride_b_row = b_layout.strides[0];
    int stride_b_col = b_layout.strides[1];

    {ty} sum = 0;
    unsigned int num_tiles = (k + 15u) / 16u;

    for (unsigned int tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {{
        // 1. Load A element into shared memory
        unsigned int col_a = tile_idx * 16u + local_col;
        if (row < m && col_a < k) {{
            int offset_a = (int)a_layout.offset + (int)row * stride_a_row + (int)col_a * stride_a_col;
            A_shared[local_row][local_col] = a[offset_a];
        }} else {{
            A_shared[local_row][local_col] = 0;
        }}

        // 2. Load B element into shared memory
        unsigned int row_b = tile_idx * 16u + local_row;
        if (row_b < k && col < n) {{
            int offset_b = (int)b_layout.offset + (int)row_b * stride_b_row + (int)col * stride_b_col;
            B_shared[local_row][local_col] = b[offset_b];
        }} else {{
            B_shared[local_row][local_col] = 0;
        }}

        __syncthreads();

        // 3. Accumulate product of the current tile
        for (unsigned int i = 0u; i < 16u; i++) {{
            sum += A_shared[local_row][i] * B_shared[i][local_col];
        }}

        __syncthreads();
    }}

    if (row < m && col < n) {{
        int stride_c_row = c_layout.strides[0];
        int stride_c_col = c_layout.strides[1];
        int offset_c = (int)c_layout.offset + (int)row * stride_c_row + (int)col * stride_c_col;
        c[offset_c] = sum;
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
    )
}

pub(crate) fn batched_matmul_shader_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}
struct MatrixLayout {{
    unsigned int shape[2];
    int strides[2];
    unsigned int offset;
}};

extern "C" __global__ void batched_matmul_kernel(
    const {ty}* a,
    const {ty}* b,
    {ty}* c,
    MatrixLayout a_layout,
    MatrixLayout b_layout,
    MatrixLayout c_layout,
    long long a_batch_stride,
    long long b_batch_stride,
    long long c_batch_stride,
    unsigned int batch_offset
) {{
    __shared__ {ty} A_shared[16][16];
    __shared__ {ty} B_shared[16][16];

    unsigned int batch_idx = batch_offset + blockIdx.z;
    unsigned int row = blockIdx.y * 16u + threadIdx.y;
    unsigned int col = blockIdx.x * 16u + threadIdx.x;
    unsigned int local_row = threadIdx.y;
    unsigned int local_col = threadIdx.x;

    unsigned int m = a_layout.shape[0];
    unsigned int k = a_layout.shape[1];
    unsigned int n = b_layout.shape[1];

    long long a_batch_off = (long long)batch_idx * a_batch_stride;
    long long b_batch_off = (long long)batch_idx * b_batch_stride;
    long long c_batch_off = (long long)batch_idx * c_batch_stride;

    int stride_a_row = a_layout.strides[0];
    int stride_a_col = a_layout.strides[1];
    int stride_b_row = b_layout.strides[0];
    int stride_b_col = b_layout.strides[1];

    {ty} sum = 0;
    unsigned int num_tiles = (k + 15u) / 16u;

    for (unsigned int tile_idx = 0u; tile_idx < num_tiles; tile_idx++) {{
        // 1. Load A element into shared memory
        unsigned int col_a = tile_idx * 16u + local_col;
        if (row < m && col_a < k) {{
            long long offset_a = a_batch_off + (long long)a_layout.offset
                + (long long)row * stride_a_row + (long long)col_a * stride_a_col;
            A_shared[local_row][local_col] = a[offset_a];
        }} else {{
            A_shared[local_row][local_col] = 0;
        }}

        // 2. Load B element into shared memory
        unsigned int row_b = tile_idx * 16u + local_row;
        if (row_b < k && col < n) {{
            long long offset_b = b_batch_off + (long long)b_layout.offset
                + (long long)row_b * stride_b_row + (long long)col * stride_b_col;
            B_shared[local_row][local_col] = b[offset_b];
        }} else {{
            B_shared[local_row][local_col] = 0;
        }}

        __syncthreads();

        // 3. Accumulate product of the current tile
        for (unsigned int i = 0u; i < 16u; i++) {{
            sum += A_shared[local_row][i] * B_shared[i][local_col];
        }}

        __syncthreads();
    }}

    if (row < m && col < n) {{
        int stride_c_row = c_layout.strides[0];
        int stride_c_col = c_layout.strides[1];
        long long offset_c = c_batch_off + (long long)c_layout.offset
            + (long long)row * stride_c_row + (long long)col * stride_c_col;
        c[offset_c] = sum;
    }}
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
    )
}
