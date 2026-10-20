//! ROCm application-layer compute operations.

/// Device-neutral adaptive-pooling seam implementation.
pub mod adaptive_pooling_seam;
/// Device-neutral rank-2 argmax/argmin seam implementation.
pub mod arg_reduce_seam;
/// Native HIP scaled dot-product attention.
pub mod attention;
/// Rank-2 axis reductions over leto layouts.
pub mod axis_reduction;
/// Device-neutral axis-reduction seam implementation.
pub mod axis_reduction_seam;
/// Device-neutral n-D concatenation seam implementation.
pub mod cat_seam;
/// Native HIP regular and transposed convolution.
#[cfg(all(feature = "rocm", target_os = "linux"))]
pub mod convolution;
/// Device-neutral batched cross-product seam implementation.
pub mod cross_product_seam;
/// Device-resident dense matrix decompositions.
#[cfg(feature = "decomposition")]
pub mod decomposition;
/// Device-neutral decomposition seam implementation.
#[cfg(feature = "decomposition")]
pub mod decomposition_seam;
/// Device-neutral dense product seam implementation.
pub mod dense_product_seam;
/// Device-API seam backing the generic accelerator layer.
pub mod device_api;
/// Device-neutral inner-product seam implementation.
pub mod dot_seam;
pub mod elementwise;
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
/// Device-neutral embedding-table seam implementation.
pub mod embedding_seam;
/// Three-dimensional fixed-scheme first-derivative sweep.
pub mod fixed_fd;
#[cfg(all(feature = "rocm", target_os = "linux"))]
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Runtime-rank expression fusion implemented by the ROCm provider.
pub mod fusion;
/// Device-neutral n-D gather seam implementation.
pub mod gather_seam;
/// Device-neutral n-D scatter-assignment seam implementation.
pub mod index_put_seam;
/// Device-neutral n-D index-selection seam implementation.
pub mod index_select_seam;
/// Device-neutral rank-2 axis-resampling seam implementation.
pub mod interpolation_seam;
/// Rank-2 matrix multiplication over strided layouts.
pub mod linalg;
/// Native HIP mean cross-entropy.
pub mod loss;
/// Device-neutral n-D masked-fill seam implementation.
pub mod masked_fill_seam;
/// Device-neutral nonzero-coordinate seam implementation.
pub mod nonzero_seam;
/// Device-neutral n-D padding seam implementation.
pub mod pad_seam;
/// Runtime-parameter unary elementwise seam implementation.
pub mod parameterized_elementwise;
pub(crate) mod pipeline;
/// Reusable rank-2 axis reduction plans.
pub mod prepared_axis_reduction;
/// Reusable dot-product and L2-norm map-reduction plans.
pub mod prepared_map_reduction;
/// Reusable multi-pass scalar reduction plans.
pub mod prepared_reduction;
/// Reusable strided elementwise plans.
pub mod prepared_strided_elementwise;
/// Seeded host-delegated random initializers.
pub mod random;
/// Device-neutral seeded random initialization seam.
pub mod random_seam;
/// Contiguous multi-pass tree reductions.
pub mod reduction;
/// Device-neutral n-D circular-shift seam implementation.
pub mod roll_seam;
/// Device-neutral n-D scatter-add seam implementation.
pub mod scatter_seam;
/// Device-neutral stable-sort seam implementation.
pub mod sort_seam;
/// Device-resident CSR sparse matrix products.
pub mod sparse;
/// Device-neutral n-D splitting seam implementation.
pub mod split_seam;
/// Device-neutral n-D stacking seam implementation.
pub mod stack_seam;
/// Three-dimensional staggered gradient/divergence kernels.
pub mod staggered;
/// Provider-owned stateful parameter updates.
pub mod stateful_update;
/// Two-dimensional Laplacian stencil kernels.
pub mod stencil;
/// Backend-neutral multi-storage kernel dispatch.
pub mod storage_kernel;
/// Backend-neutral authored-kernel command streams.
pub mod stream;
/// Layout-aware operand descriptors.
pub mod strided;
/// Rank-≤8 layout-aware elementwise operations.
pub mod strided_elementwise;
/// Device-neutral n-D tiling seam implementation.
pub mod tile_seam;
/// Device-neutral rank-2 top-k selection seam implementation.
pub mod topk_seam;
/// Device-neutral rank-2 triangular masking seam implementation.
pub mod triangular_seam;
/// Dense vector recurrences and prepared reductions.
pub mod vector;
/// Volume ray-integral kernels.
pub mod volume;
/// Device-neutral n-D conditional-selection seam implementation.
pub mod where_seam;
/// Generic pooling and sliding-window operations.
pub mod window;
