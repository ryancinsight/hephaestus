//! The aggregate backend conformance entry point.
//!
//! Each seam in this crate carries its own `assert_*_contract`, and a backend
//! instantiates the ones it wires up. That shape has a failure mode the
//! individual clauses cannot see: **coverage is opt-in**, so a backend that
//! never wires a clause reads as green.
//!
//! It is not hypothetical. Measured 2026-09-10:
//!
//! - `hephaestus-wgpu` implements `CrossEntropyOps` for `WgpuCrossEntropyOps`
//!   (`src/application/loss/seam.rs:41`) and covers the seam in its own unit
//!   tests, but **no integration test ever runs
//!   [`assert_cross_entropy_contract`](crate::assert_cross_entropy_contract)**
//!   against it — there are zero `cross_entropy` references anywhere under
//!   `crates/hephaestus-wgpu/tests/`.
//! - `hephaestus-rocm` and `hephaestus-metal` never run
//!   [`assert_staggered_3d_contract`](crate::assert_staggered_3d_contract) at
//!   all.
//!
//! Each backend does carry a case-count guard (wgpu asserts
//! `CONTRACT_CASES.len() == 143 / 190 / 148 / 192` by feature set), but a count
//! cannot catch this class of gap: removing one seam's cases and adding
//! unrelated ones leaves the total unchanged and the run green.
//!
//! [`BackendUnderTest`] closes the hole by naming every seam as a field, so a
//! backend that implements the operations but omits a seam fails to compile —
//! the one failure mode that cannot be silently ignored.
//!
//! # Shape
//!
//! One generic struct of borrowed seams, not a list of function parameters: the
//! workspace denies `clippy::pedantic` and does not allow `too_many_arguments`,
//! so a thirty-seven-parameter function is not available here. A struct also survives
//! new seams without changing its own signature.
//!
//! The seams must stay as type parameters rather than `dyn`. Every clause needs
//! `Sized` associated types, and several declare marker bounds
//! (`SumOp: CombineExpr<R::Dialect>`, `f32: OpIdentity<SumOp>`, …) whose marker
//! types resolve from the *concrete* seam type at the call site, so trait
//! objects cannot express them.

use crate::typed_elementwise::TypedElementwiseBackend;
use eunomia::Pod;
use hephaestus_core::{
    AbsOp, AdaGrad, AdaGradParameters, Adam, AdamParameters, AdamW, AdamWParameters,
    AdaptivePoolingOps, AddOp, ArgReduceOps, AttentionOps, AxisReductionOps, BatchSubmitOps,
    BinaryExpr, CatOps, CeluGradOp, CeluOp, CombineExpr, ComputeDevice, ConvolutionOps,
    CrossEntropyOps, CrossProductOps, CtcOps, CumProdOp, CumSumOp, DecompositionOps,
    DenseCompositionOps, DenseMatrixFunctionOps, DenseProductOps, DenseVectorOps, DialectScalar,
    DivOp, DotOps, ElementwiseOps, EmbeddingOps, EqOp, FixedFd3DOps, FullReductionOps, GatherOps,
    GeOp, GtOp, HardshrinkGradOp, HardshrinkOp, HardtanhGradOp, HardtanhOp, IdentityToken,
    IndexPutOps, IndexSelectOps, InterpolationOps, LeOp, LeakyReluGradOp, LeakyReluOp, LtOp,
    MaskedFillOps, MaxOp, MinOp, MulOp, NeOp, NegOp, NonzeroOps, OpIdentity, PadOps,
    ParameterizedUnaryExpr, ParameterizedUnaryOps, ProdOp, RandomInitOps, RayIntegralOps, RmsProp,
    RmsPropParameters, RollOps, ScanOps, ScatterOps, Sgd, SgdParameters, SoftshrinkGradOp,
    SoftshrinkOp, SortOps, SparseOperatorOps, SplitOps, SqrtOp, StackOps, Staggered3DOps,
    StatefulUpdateOps, StatefulUpdateRule, StencilOps, SubOp, SumOp, ThresholdGradOp, ThresholdOp,
    TileOps, TopKOps, TriangularOps, TypedBinaryExpr, UnaryExpr, WhereOps,
};

/// A backend's seam implementations, borrowed for one conformance run.
///
/// Every field is a seam this crate carries a clause for. Construct it with
/// struct literal syntax and hand it to [`assert_backend_contract`].
///
/// A backend missing a seam implementation cannot construct this type, which is
/// the point: the omission is a compile error at the call site rather than a
/// seam that silently stops being checked.
///
/// The type parameters are the seam types, in field order. They are unnamed in
/// practice — construct the struct with a literal and let inference bind them.
pub struct BackendUnderTest<
    'a,
    D,
    Ap,
    Ar,
    A,
    R,
    Ct,
    C,
    X,
    Cp,
    V,
    P,
    K,
    M,
    U,
    I,
    E,
    Em,
    Ga,
    G,
    F,
    Ip,
    Pd,
    N,
    T,
    O,
    B,
    S,
    Sp,
    St,
    Sc,
    Is,
    Iu,
    Mf,
    Ro,
    Tl,
    Wh,
    Sr,
    Nz,
    Do,
    H,
    L,
    Y,
    W,
    Tk,
    Tr,
    Z,
> {
    /// The device every clause runs against.
    pub device: &'a D,
    /// `AdaptivePoolingOps<D, f32>`.
    pub adaptive_pooling: &'a Ap,
    /// `ArgReduceOps<D, i32>`.
    pub arg_reduce: &'a Ar,
    /// `AttentionOps<D, f32>`.
    pub attention: &'a A,
    /// `AxisReductionOps<D, f32>`.
    pub axis_reduction: &'a R,
    /// `CatOps<D, i32>`.
    pub cat: &'a Ct,
    /// `ConvolutionOps<D, f32>`.
    pub convolution: &'a C,
    /// `CrossEntropyOps<D, f32>`.
    pub cross_entropy: &'a X,
    /// `CrossProductOps<D, f32>`.
    pub cross_product: &'a Cp,
    /// `CtcOps<D>`.
    pub ctc: &'a V,
    /// `DecompositionOps<D>`.
    pub decomposition: &'a P,
    /// `DenseCompositionOps<D>`.
    pub dense_composition: &'a K,
    /// `DenseMatrixFunctionOps<D>`.
    pub dense_matrix_function: &'a M,
    /// `DenseProductOps<D, f32>`.
    pub dense_product: &'a U,
    /// `DenseVectorOps<D, f32>`.
    pub dense_vector: &'a I,
    /// `ElementwiseOps<D, f32>`.
    pub elementwise: &'a E,
    /// `EmbeddingOps<D, f32>`.
    pub embedding: &'a Em,
    /// `GatherOps<D, i32>`.
    pub gather: &'a Ga,
    /// `FixedFd3DOps<D>`.
    pub fixed_fd_3d: &'a G,
    /// `FullReductionOps<D, f32>`.
    pub full_reduction: &'a F,
    /// `InterpolationOps<D, f32>`.
    pub interpolation: &'a Ip,
    /// `PadOps<D, i32>`.
    pub pad: &'a Pd,
    /// `ParameterizedUnaryOps<D>`.
    pub parameterized_unary: &'a N,
    /// `RandomInitOps<D, f32>`.
    pub random_init: &'a T,
    /// `RayIntegralOps<D>`.
    pub ray_integral: &'a O,
    /// `ScanOps<D, f32>`.
    pub scan: &'a B,
    /// `SparseOperatorOps<D, f32>`.
    pub sparse_operator: &'a S,
    /// `SplitOps<D, i32>`.
    pub split: &'a Sp,
    /// `StackOps<D, i32>`.
    pub stack: &'a St,
    /// `ScatterOps<D, i32>` and `ScatterOps<D, f32>`.
    pub scatter: &'a Sc,
    /// `IndexSelectOps<D, i32>`.
    pub index_select: &'a Is,
    /// `IndexPutOps<D, i32>`.
    pub index_put: &'a Iu,
    /// `MaskedFillOps<D, i32>`.
    pub masked_fill: &'a Mf,
    /// `RollOps<D, i32>`.
    pub roll: &'a Ro,
    /// `TileOps<D, i32>`.
    pub tile: &'a Tl,
    /// `WhereOps<D, i32>`.
    pub where_cond: &'a Wh,
    /// `SortOps<D, i32>`.
    pub sort: &'a Sr,
    /// `NonzeroOps<D, i32>`.
    pub nonzero: &'a Nz,
    /// `DotOps<D, i32>`.
    pub dot: &'a Do,
    /// `BatchSubmitOps<D, f32>`.
    pub batch_submit: &'a H,
    /// `Staggered3DOps<D>`.
    pub staggered_3d: &'a L,
    /// `StatefulUpdateOps<D>`.
    pub stateful_update: &'a Y,
    /// `StencilOps<D>`.
    pub stencil: &'a W,
    /// `TopKOps<D, i32>`.
    pub topk: &'a Tk,
    /// `TriangularOps<D, i32>`.
    pub triangular: &'a Tr,
    /// `TypedElementwiseBackend<D>`.
    pub typed_elementwise: &'a Z,
}

/// Run **every** clause in this crate against one backend.
///
/// This is the entry a backend calls to be held to the whole contract. Prefer it
/// over calling the individual `assert_*_contract` functions: those remain public
/// so a backend can run a subset while migrating, and reaching for one directly is
/// how the coverage gaps documented on this module arose.
///
/// # Panics
///
/// Panics with the violated clause's own located message when the backend does
/// not satisfy the contract.
#[expect(
    clippy::type_complexity,
    reason = "BackendUnderTest is already the factoring this lint asks for: the seam set is one named struct, and what remains here is that struct's own parameter list. The thirty-seven parameters cannot collapse further, because every clause below declares marker bounds whose marker types resolve from the concrete seam type, so each seam stays a distinct parameter rather than an associated type or a trait object; a type alias would have to repeat the same list to be usable."
)]
pub fn assert_backend_contract<
    D,
    Ap,
    Ar,
    A,
    R,
    Ct,
    C,
    X,
    Cp,
    V,
    P,
    K,
    M,
    U,
    I,
    E,
    Em,
    Ga,
    G,
    F,
    Ip,
    Pd,
    N,
    T,
    O,
    B,
    S,
    Sp,
    St,
    Sc,
    Is,
    Iu,
    Mf,
    Ro,
    Tl,
    Wh,
    Sr,
    Nz,
    Do,
    H,
    L,
    Y,
    W,
    Tk,
    Tr,
    Z,
>(
    backend: &BackendUnderTest<
        '_,
        D,
        Ap,
        Ar,
        A,
        R,
        Ct,
        C,
        X,
        Cp,
        V,
        P,
        K,
        M,
        U,
        I,
        E,
        Em,
        Ga,
        G,
        F,
        Ip,
        Pd,
        N,
        T,
        O,
        B,
        S,
        Sp,
        St,
        Sc,
        Is,
        Iu,
        Mf,
        Ro,
        Tl,
        Wh,
        Sr,
        Nz,
        Do,
        H,
        L,
        Y,
        W,
        Tk,
        Tr,
        Z,
    >,
) where
    D: ComputeDevice,
    Ap: AdaptivePoolingOps<D, f32>,
    Ar: ArgReduceOps<D, i32>,
    A: AttentionOps<D, f32>,
    R: AxisReductionOps<D, f32>,
    Ct: CatOps<D, i32>,
    C: ConvolutionOps<D, f32>,
    X: CrossEntropyOps<D, f32>,
    Cp: CrossProductOps<D, f32>,
    V: CtcOps<D>,
    P: DecompositionOps<D>,
    K: DenseCompositionOps<D>,
    M: DenseMatrixFunctionOps<D>,
    U: DenseProductOps<D, f32>,
    I: DenseVectorOps<D, f32>,
    E: ElementwiseOps<D, f32>,
    Em: EmbeddingOps<D, f32>,
    Ga: GatherOps<D, i32>,
    G: FixedFd3DOps<D>,
    F: FullReductionOps<D, f32>,
    Ip: InterpolationOps<D, f32>,
    Pd: PadOps<D, i32>,
    N: ParameterizedUnaryOps<D>,
    T: RandomInitOps<D, f32>,
    O: RayIntegralOps<D>,
    B: ScanOps<D, f32> + ScanOps<D, i32>,
    S: SparseOperatorOps<D, f32>,
    Sp: SplitOps<D, i32>,
    St: StackOps<D, i32>,
    Sc: ScatterOps<D, i32>,
    Sc: ScatterOps<D, f32>,
    Is: IndexSelectOps<D, i32>,
    Iu: IndexPutOps<D, i32>,
    Mf: MaskedFillOps<D, i32>,
    Ro: RollOps<D, i32>,
    Tl: TileOps<D, i32>,
    Wh: WhereOps<D, i32>,
    Sr: SortOps<D, i32>,
    Nz: NonzeroOps<D, i32>,
    Do: DotOps<D, i32>,
    H: BatchSubmitOps<D, f32>,
    L: Staggered3DOps<D>,
    Y: StatefulUpdateOps<D>,
    W: StencilOps<D>,
    Tk: TopKOps<D, i32>,
    Tr: TriangularOps<D, i32>,
    // Naming `TypedElementwiseBackend<D>` as a bound is only well-formed when its
    // own `where` clauses hold, so its scalar and comparison-operator
    // obligations are restated first.
    u32: DialectScalar<<Z as ElementwiseOps<D, u32>>::Dialect>,
    i32: DialectScalar<<Z as ElementwiseOps<D, i32>>::Dialect>,
    f32: DialectScalar<<Z as ElementwiseOps<D, f32>>::Dialect>,
    EqOp: TypedBinaryExpr<<Z as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Z as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<Z as ElementwiseOps<D, f32>>::Dialect, f32>,
    NeOp: TypedBinaryExpr<<Z as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Z as ElementwiseOps<D, f32>>::Dialect, f32>,
    LtOp: TypedBinaryExpr<<Z as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Z as ElementwiseOps<D, i32>>::Dialect, i32>
        + TypedBinaryExpr<<Z as ElementwiseOps<D, f32>>::Dialect, f32>,
    LeOp: TypedBinaryExpr<<Z as ElementwiseOps<D, u32>>::Dialect, u32>,
    GtOp: TypedBinaryExpr<<Z as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Z as ElementwiseOps<D, i32>>::Dialect, i32>,
    GeOp: TypedBinaryExpr<<Z as ElementwiseOps<D, u32>>::Dialect, u32>
        + TypedBinaryExpr<<Z as ElementwiseOps<D, f32>>::Dialect, f32>,
    AddOp: BinaryExpr<<Z as ElementwiseOps<D, f32>>::Dialect>,
    MulOp: BinaryExpr<<Z as ElementwiseOps<D, f32>>::Dialect>,
    Z: TypedElementwiseBackend<D>,
    // Every clause below declares marker bounds whose marker types are supplied
    // by the call site, not derived from the seam type. They cannot be
    // discharged generically, so each clause's own bound set is restated here
    // verbatim. This is the reason the aggregate is written against concrete
    // seam parameters instead of an abstract backend trait.
    //
    // axis-reduction markers
    ProdOp: CombineExpr<<R as AxisReductionOps<D, f32>>::Dialect>,
    SumOp: CombineExpr<<R as AxisReductionOps<D, f32>>::Dialect>,
    MinOp: CombineExpr<<R as AxisReductionOps<D, f32>>::Dialect>,
    MaxOp: CombineExpr<<R as AxisReductionOps<D, f32>>::Dialect>,
    f32: OpIdentity<ProdOp>
        + IdentityToken<ProdOp, <R as AxisReductionOps<D, f32>>::Dialect>
        + OpIdentity<SumOp>
        + IdentityToken<SumOp, <R as AxisReductionOps<D, f32>>::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, <R as AxisReductionOps<D, f32>>::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, <R as AxisReductionOps<D, f32>>::Dialect>,
    // elementwise markers
    f32: DialectScalar<<E as ElementwiseOps<D, f32>>::Dialect> + Pod,
    NegOp: UnaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    AbsOp: UnaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    SqrtOp: UnaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    AddOp: BinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    SubOp: BinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    MulOp: BinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    DivOp: BinaryExpr<<E as ElementwiseOps<D, f32>>::Dialect>,
    // full-reduction markers
    SumOp: CombineExpr<<F as FullReductionOps<D, f32>>::Dialect>,
    ProdOp: CombineExpr<<F as FullReductionOps<D, f32>>::Dialect>,
    MinOp: CombineExpr<<F as FullReductionOps<D, f32>>::Dialect>,
    MaxOp: CombineExpr<<F as FullReductionOps<D, f32>>::Dialect>,
    f32: OpIdentity<SumOp>
        + IdentityToken<SumOp, <F as FullReductionOps<D, f32>>::Dialect>
        + OpIdentity<ProdOp>
        + IdentityToken<ProdOp, <F as FullReductionOps<D, f32>>::Dialect>
        + OpIdentity<MinOp>
        + IdentityToken<MinOp, <F as FullReductionOps<D, f32>>::Dialect>
        + OpIdentity<MaxOp>
        + IdentityToken<MaxOp, <F as FullReductionOps<D, f32>>::Dialect>,
    // parameterized-unary markers
    HardtanhOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    HardtanhGradOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    ThresholdOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    ThresholdGradOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    LeakyReluOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    LeakyReluGradOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    HardshrinkOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    HardshrinkGradOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    SoftshrinkOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    SoftshrinkGradOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    CeluOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    CeluGradOp: ParameterizedUnaryExpr<<N as ParameterizedUnaryOps<D>>::Dialect>,
    // scan markers
    CumSumOp:
        CombineExpr<<B as ScanOps<D, f32>>::Dialect> + CombineExpr<<B as ScanOps<D, i32>>::Dialect>,
    CumProdOp: CombineExpr<<B as ScanOps<D, f32>>::Dialect>,
    f32: OpIdentity<CumSumOp>
        + IdentityToken<CumSumOp, <B as ScanOps<D, f32>>::Dialect>
        + OpIdentity<CumProdOp>
        + IdentityToken<CumProdOp, <B as ScanOps<D, f32>>::Dialect>,
    i32: OpIdentity<CumSumOp> + IdentityToken<CumSumOp, <B as ScanOps<D, i32>>::Dialect>,
    // stateful-update markers
    Sgd: StatefulUpdateRule<<Y as StatefulUpdateOps<D>>::Dialect, Parameters = SgdParameters>,
    Adam: StatefulUpdateRule<<Y as StatefulUpdateOps<D>>::Dialect, Parameters = AdamParameters>,
    AdamW: StatefulUpdateRule<<Y as StatefulUpdateOps<D>>::Dialect, Parameters = AdamWParameters>,
    RmsProp:
        StatefulUpdateRule<<Y as StatefulUpdateOps<D>>::Dialect, Parameters = RmsPropParameters>,
    AdaGrad:
        StatefulUpdateRule<<Y as StatefulUpdateOps<D>>::Dialect, Parameters = AdaGradParameters>,
{
    let device = backend.device;
    crate::adaptive_pooling::assert_adaptive_pooling_contract(device, backend.adaptive_pooling);
    crate::arg_reduce::assert_arg_reduce_contract(device, backend.arg_reduce);
    crate::arg_reduce::assert_arg_reduce_transposed_view_contract(device, backend.arg_reduce);
    crate::attention::assert_attention_contract(device, backend.attention);
    crate::axis_reduction::assert_axis_reduction_contract(device, backend.axis_reduction);
    crate::cat::assert_cat_contract::<_, _, i32>(device, backend.cat);
    crate::convolution::assert_convolution_contract(device, backend.convolution);
    crate::cross_entropy::assert_cross_entropy_contract(device, backend.cross_entropy);
    crate::cross_product::assert_cross_product_contract(device, backend.cross_product);
    crate::ctc::assert_ctc_contract(device, backend.ctc);
    crate::decomposition::assert_decomposition_contract(device, backend.decomposition);
    crate::dense_composition::assert_dense_composition_contract(device, backend.dense_composition);
    crate::dense_matrix_function::assert_dense_matrix_function_contract(
        device,
        backend.dense_matrix_function,
    );
    crate::dense_product::assert_dense_product_contract(device, backend.dense_product);
    crate::dense_vector::assert_dense_vector_contract(device, backend.dense_vector);
    crate::elementwise::assert_elementwise_contract(device, backend.elementwise);
    crate::embedding::assert_embedding_gather_contract(device, backend.embedding);
    crate::embedding::assert_embedding_gather_rejects_out_of_range_index(device, backend.embedding);
    crate::gather::assert_gather_contract::<_, _, i32>(device, backend.gather);
    crate::embedding::assert_embedding_scatter_add_contract(device, backend.embedding);
    crate::embedding::assert_embedding_scatter_add_rejects_out_of_range_index(
        device,
        backend.embedding,
    );
    crate::fixed_fd::assert_fixed_fd_3d_contract(device, backend.fixed_fd_3d);
    crate::full_reduction::assert_full_reduction_contract(device, backend.full_reduction);
    crate::interpolation::assert_interpolation_contract(device, backend.interpolation);
    crate::pad::assert_pad_contract::<_, _, i32>(device, backend.pad);
    crate::parameterized_unary::assert_parameterized_unary_contract(
        device,
        backend.parameterized_unary,
    );
    crate::random_init::assert_random_init_contract(device, backend.random_init);
    crate::ray_integral::assert_ray_integral_contract(device, backend.ray_integral);
    crate::scan::assert_scan_contract(device, backend.scan);
    crate::scan::assert_scan_leto_contract::<_, _, i32>(device, backend.scan);
    crate::scan::assert_scan_leto_contract::<_, _, f32>(device, backend.scan);
    crate::sparse::assert_sparse_operator_contract(device, backend.sparse_operator);
    crate::split::assert_split_contract::<_, _, i32>(device, backend.split);
    crate::stack::assert_stack_contract::<_, _, i32>(device, backend.stack);
    crate::scatter::assert_scatter_contract::<_, _, i32>(device, backend.scatter);
    crate::scatter::assert_scatter_order_contract(device, backend.scatter);
    crate::index_select::assert_index_select_contract::<_, _, i32>(device, backend.index_select);
    crate::index_put::assert_index_put_contract::<_, _, i32>(device, backend.index_put);
    crate::masked_fill::assert_masked_fill_contract::<_, _, i32>(device, backend.masked_fill);
    crate::roll::assert_roll_contract::<_, _, i32>(device, backend.roll);
    crate::tile::assert_tile_contract::<_, _, i32>(device, backend.tile);
    crate::where_cond::assert_where_contract::<_, _, i32>(device, backend.where_cond);
    crate::sort::assert_sort_contract::<_, _, i32>(device, backend.sort);
    crate::nonzero::assert_nonzero_contract::<_, _, i32>(device, backend.nonzero);
    crate::dot::assert_dot_contract::<_, _, i32>(device, backend.dot);
    crate::sparse::assert_batch_submit_contract(device, backend.batch_submit);
    crate::staggered::assert_staggered_3d_contract(device, backend.staggered_3d);
    crate::stateful_update::assert_stateful_update_contract(device, backend.stateful_update);
    crate::stencil::assert_stencil_contract(device, backend.stencil);
    crate::topk::assert_topk_contract(device, backend.topk);
    crate::topk::assert_topk_strided_output_contract(device, backend.topk);
    crate::triangular::assert_triangular_contract(device, backend.triangular);
    crate::typed_elementwise::assert_typed_elementwise_contract(device, backend.typed_elementwise);
    crate::transfer::assert_transfer_contract(device);
}
