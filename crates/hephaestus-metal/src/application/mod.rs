//! Compute dispatch delegation to Metal.

/// Device-neutral adaptive-pooling seam implementation.
pub mod adaptive_pooling_seam;
/// Device-neutral rank-2 argmax/argmin seam implementation.
pub mod arg_reduce_seam;
/// Zero-copy attention delegation to Metal-selected WGPU.
pub mod attention;
/// Device-neutral axis-reduction seam implementation.
pub mod axis_reduction_seam;
/// Device-neutral n-D concatenation seam implementation.
pub mod cat_seam;
/// Zero-copy convolution delegation to Metal-selected WGPU.
pub mod convolution;
/// Device-neutral batched cross-product seam implementation.
pub mod cross_product_seam;
/// Matrix decompositions.
#[cfg(feature = "decomposition")]
pub mod decomposition;
/// Device-neutral decomposition seam implementation.
#[cfg(feature = "decomposition")]
pub mod decomposition_seam;
/// Device-neutral dense product seam implementation.
pub mod dense_product_seam;
/// Device-neutral dot-product seam implementation.
pub mod dot_seam;
/// Elementwise compute dispatch.
pub mod elementwise;
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
/// Device-neutral embedding-table seam implementation.
pub mod embedding_seam;
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Runtime-rank expression fusion implemented by the Metal provider.
pub mod fusion;
/// Device-neutral n-D gather seam implementation.
pub mod gather_seam;
/// Device-neutral index-put seam implementation.
pub mod index_put_seam;
/// Device-neutral index-select seam implementation.
pub mod index_select_seam;
/// Device-neutral rank-2 axis-resampling seam implementation.
pub mod interpolation_seam;
/// Linear algebra operations.
pub mod linalg;
/// Fluent dense-matrix traits.
pub mod linalg_traits;
/// Zero-copy mean cross-entropy delegation to Metal-selected WGPU.
pub mod loss;
/// Device-neutral masked-fill seam implementation.
pub mod masked_fill_seam;
/// Device-neutral nonzero-index seam implementation.
pub mod nonzero_seam;
/// Device-neutral n-D padding seam implementation.
pub mod pad_seam;
/// Runtime-parameter unary elementwise seam implementation.
pub mod parameterized_elementwise;
/// Reusable dot-product and L2-norm map-reduction plans.
pub mod prepared_map_reduction;
/// Seeded host-delegated random initializers.
pub mod random;
/// Device-neutral seeded random initialization seam.
pub mod random_seam;
/// Reduction operations.
pub mod reduction;
/// Device-neutral n-D circular-shift seam implementation.
pub mod roll_seam;
/// Scan operations.
pub mod scan;
/// Device-neutral scan seam implementation.
pub mod scan_seam;
/// Device-neutral n-D scatter seam implementation.
pub mod scatter_seam;
/// Device-neutral segmented sort seam implementation.
pub mod sort_seam;
/// GPU-resident CSR sparse matrix operations.
pub mod sparse;
/// Device-neutral sparse operator seam implementation.
pub mod sparse_seam;
/// Device-neutral n-D split seam implementation.
pub mod split_seam;
/// Device-neutral n-D stack seam implementation.
pub mod stack_seam;
/// Stateful parameter updates through the native Metal-selected WGPU device.
pub mod stateful_update;
/// 2D Laplacian stencil delegation.
pub mod stencil;
/// Metal-selected storage-kernel dispatch.
pub mod storage_kernel;
/// Metal-selected authored-kernel command streams.
pub mod stream;
/// Strided layout wrappers.
pub mod strided;
/// Device-neutral n-D tiling seam implementation.
pub mod tile_seam;
/// Device-neutral rank-2 top-k selection seam implementation.
pub mod topk_seam;
/// Device-neutral rank-2 triangular masking seam implementation.
pub mod triangular_seam;
/// Dense vector recurrence operations.
pub mod vector;
/// Volume ray-integral delegation.
pub mod volume;
/// Device-neutral where-select seam implementation.
pub mod where_seam;
/// Zero-copy pooling and sliding-window delegation.
pub mod window;
