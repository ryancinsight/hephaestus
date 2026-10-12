//! Compute dispatch delegation to Metal.

/// Zero-copy attention delegation to Metal-selected WGPU.
/// Device-neutral adaptive-pooling seam implementation.
pub mod adaptive_pooling_seam;
/// Device-neutral rank-2 argmax/argmin seam implementation.
pub mod arg_reduce_seam;
/// Zero-copy attention delegation to Metal-selected WGPU.
pub mod attention;
/// Device-neutral axis-reduction seam implementation.
pub mod axis_reduction_seam;
/// Zero-copy concatenation delegation to Metal-selected WGPU.
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
/// Provider-owned inner-product seam implementation.
pub mod dot_seam;
/// Elementwise compute dispatch.
pub mod elementwise;
/// Device-neutral elementwise seam implementation.
pub mod elementwise_seam;
/// Device-neutral embedding-gather seam implementation.
pub mod embedding_seam;
/// Device-neutral full-reduction seam implementation.
pub mod full_reduction_seam;
/// Zero-copy gather delegation to Metal-selected WGPU.
pub mod gather_seam;
/// Zero-copy scatter-assignment delegation to Metal-selected WGPU.
pub mod index_put_seam;
/// Zero-copy index-selection delegation to Metal-selected WGPU.
pub mod index_select_seam;
/// Device-neutral rank-2 axis-resampling seam implementation.
pub mod interpolation_seam;
/// Linear algebra operations.
pub mod linalg;
/// Fluent dense-matrix traits.
pub mod linalg_traits;
/// Zero-copy mean cross-entropy delegation to Metal-selected WGPU.
pub mod loss;
/// Zero-copy masked-fill delegation to Metal-selected WGPU.
pub mod masked_fill_seam;
/// Zero-copy nonzero-coordinate delegation to Metal-selected WGPU.
pub mod nonzero_seam;
/// Runtime-parameter unary elementwise seam implementation.
/// Device-neutral n-D padding seam implementation.
pub mod pad_seam;
pub mod parameterized_elementwise;
/// Reusable dot-product and L2-norm map-reduction plans.
pub mod prepared_map_reduction;
/// Seeded host-delegated random initializers.
pub mod random;
/// Device-neutral seeded random initialization seam.
pub mod random_seam;
/// Reduction operations.
pub mod reduction;
/// Zero-copy circular-shift delegation to Metal-selected WGPU.
pub mod roll_seam;
/// Scan operations.
pub mod scan;
/// Device-neutral scan seam implementation.
pub mod scan_seam;
/// Zero-copy scatter-add delegation to Metal-selected WGPU.
pub mod scatter_seam;
/// Device-neutral sparse operator seam implementation.
/// Zero-copy stable-sort delegation to Metal-selected WGPU.
pub mod sort_seam;
/// GPU-resident CSR sparse matrix operations.
pub mod sparse;
pub mod sparse_seam;
/// Zero-copy splitting delegation to Metal-selected WGPU.
pub mod split_seam;
/// Zero-copy stacking delegation to Metal-selected WGPU.
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
/// Zero-copy tiling delegation to Metal-selected WGPU.
pub mod tile_seam;
/// Device-neutral rank-2 top-k selection seam implementation.
pub mod topk_seam;
/// Device-neutral rank-2 triangular masking seam implementation.
pub mod triangular_seam;
/// Dense vector recurrence operations.
pub mod vector;
/// Volume ray-integral delegation.
pub mod volume;
/// Zero-copy conditional-selection delegation to Metal-selected WGPU.
pub mod where_seam;
/// Zero-copy pooling and sliding-window delegation.
pub mod window;
