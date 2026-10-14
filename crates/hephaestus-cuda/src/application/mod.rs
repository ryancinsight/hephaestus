//! Monomorphized compute dispatch over the CUDA device.

/// Native scaled dot-product attention operations.
pub mod attention;
/// Device-neutral axis-reduction seam implementation.
pub mod axis_reduction_seam;
/// Device-neutral n-D concatenation seam implementation.
pub mod cat_seam;
/// Native regular and transposed convolution operations.
pub mod convolution;
/// Device-neutral decomposition seam implementation.
#[cfg(feature = "decomposition")]
pub mod decomposition_seam;
/// Device-neutral dense product seam implementation.
pub mod dense_product_seam;
/// Device-API seam backing the generic accelerator layer.
pub mod device_api;
/// Device-neutral inner-product seam implementation.
pub mod dot_seam;
/// Contiguous elementwise operations.
pub mod elementwise;
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
/// Three-dimensional fixed-scheme first-derivative sweep.
pub mod fixed_fd;
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Runtime-rank expression fusion implemented by the CUDA provider.
pub mod fusion;
/// Device-neutral n-D gather seam implementation.
pub mod gather_seam;
/// Device-neutral n-D scatter-assignment seam implementation.
pub mod index_put_seam;
/// Device-neutral n-D index-selection seam implementation.
pub mod index_select_seam;
/// Linear algebra operations (matmul, batch matmul, trace, dot, norms).
pub mod linalg;
/// Native mean cross-entropy operations.
pub mod loss;
/// Device-neutral n-D masked-fill seam implementation.
pub mod masked_fill_seam;
/// Device-neutral nonzero-coordinate seam implementation.
pub mod nonzero_seam;
/// Runtime-parameter unary elementwise seam implementation.
pub mod parameterized_elementwise;
/// Pipeline compilation and launch helpers.
pub mod pipeline;
/// Reusable rank-2 axis reduction plans.
pub mod prepared_axis_reduction;
/// Reusable dot-product and L2-norm map-reduction plans.
pub mod prepared_map_reduction;
/// Reusable multi-pass scalar reduction plans.
pub mod prepared_reduction;
/// Reusable strided elementwise plans.
pub mod prepared_strided_elementwise;
/// Device-neutral seeded random initialization seam.
pub mod random_seam;
/// Multi-pass tree reductions.
pub mod reduction;
/// Device-neutral n-D scatter-add seam implementation.
pub mod scatter_seam;
/// Device-neutral stable-sort seam implementation.
pub mod sort_seam;
/// Device-neutral n-D splitting seam implementation.
pub mod split_seam;
/// Device-neutral n-D stacking seam implementation.
pub mod stack_seam;
/// Three-dimensional staggered gradient/divergence kernels.
pub mod staggered;
/// Device-neutral stateful-update seam implementation.
pub mod stateful_update;
/// Two-dimensional Laplacian stencil kernels.
pub mod stencil;
/// Backend-neutral multi-storage kernel dispatch.
pub mod storage_kernel;
/// Backend-neutral command stream implementation for authored CUDA C kernels.
pub mod stream;
/// Layout-aware strided elementwise operations.
pub mod strided;
/// Device-neutral n-D tiling seam implementation.
pub mod tile_seam;
/// Dense vector recurrences and prepared reductions.
pub mod vector;
/// Volume ray-integral kernels.
pub mod volume;
/// Device-neutral n-D conditional-selection seam implementation.
pub mod where_seam;
/// Generic pooling and sliding-window operations.
pub mod window;

#[cfg(feature = "decomposition")]
/// Dense matrix decompositions (Cholesky, LU, QR) backed by leto-ops.
pub mod decomposition;

/// Seeded host-delegated PRNG initializers.
pub mod random;
/// GPU Compressed Sparse Row (CSR) sparse matrix operations.
pub mod sparse;
