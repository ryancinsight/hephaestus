//! Domain contracts: errors, typed device buffers, and accelerator seams.

/// One generic accelerator layer over the narrow device-API seam.
pub mod accelerator;
/// Device-neutral rank-2 argmax/argmin seam.
pub mod arg_reduce;
/// Device-neutral scaled dot-product attention contracts and planning.
pub mod attention;
/// Typed device-buffer contract.
pub mod buffer;
/// Device-neutral convolution operands, planning, and dispatch seam.
pub mod convolution;
/// Device-neutral batched 3-vector cross product seam.
pub mod cross_product;
/// Shared CPU-side panel factorisation routines for blocked decomposition.
pub mod decomposition;
/// Device-neutral dense decomposition seam (ADR 0042).
pub mod decomposition_seam;
/// Device-neutral dense product seam (ADR 0044).
pub mod dense_product;
/// Compute-device acquisition and transfer seam.
pub mod device;
/// Kernel-dialect markers and per-dialect scalar tokens.
pub mod dialect;
/// Backend-neutral elementwise operations over strided n-D views.
pub mod elementwise;
/// Device-neutral embedding-table gather seam.
pub mod embedding;
/// Error contracts shared by all backends.
pub mod error;
/// Backend-neutral volume ray-integral geometry and validation.
/// Provider-neutral three-dimensional acoustic FDTD contracts.
pub mod fdtd;
/// Device-neutral dense complex Fourier-transform contracts.
pub mod fft;
/// Runtime-rank expression-fusion contracts.
pub mod fusion;
/// Backend-neutral kernel authoring: interface and source declarations.
pub mod interface;
/// Device-neutral rank-2 axis resampling (nearest / linear) seam.
pub mod interpolation;
/// Kernel-dispatch contracts shared by accelerator backends.
pub mod kernel;
/// Launch-shape vocabulary for occupancy-planned dispatch.
pub mod launch;
/// Device-neutral classification-loss operands, planning, and dispatch seam.
pub mod loss;
/// Zero-sized operation markers with per-dialect shader expressions.
pub mod ops;
/// Device-neutral n-D padding seam.
pub mod pad;
/// Runtime-parameter unary expressions and their backend-neutral dispatch seam.
pub mod parameterized;
/// Shared narrowing/error helpers for dispatch planning.
pub(crate) mod planning;
/// Device-neutral spatial pooling operands, planning, and dispatch seam.
pub mod pooling;
/// Device-neutral seeded random initialization seam.
pub mod random;
/// Backend-neutral axis-reduction validation and dispatch planning.
pub mod reduction;
/// Backend-neutral axis-scan validation and dispatch planning.
pub mod scan;
/// Device-neutral spatial unfold/fold operands, planning, and dispatch seam.
pub mod sliding_window;
/// Device-neutral sparse operator contracts.
pub mod sparse;
/// Backend-neutral two-dimensional Laplacian stencil parameters.
pub mod staggered;
/// Provider-owned stateful parameter-update rules and dispatch seam.
pub mod stateful_update;
pub mod stencil;
/// Authored-kernel dispatch seam: prepared pipelines, bindings, streams.
pub mod stream;
/// Device-agnostic strided kernel metadata and layout packing.
pub mod strided;
/// Device-neutral rank-2 top-k selection seam.
pub mod topk;
/// Dense vector-operation contracts.
pub mod vector;
/// Device-neutral strided views over backend buffers.
pub mod view;
/// Volume contracts.
pub mod volume;
/// Shared spatial-window geometry and validation.
pub mod window;
