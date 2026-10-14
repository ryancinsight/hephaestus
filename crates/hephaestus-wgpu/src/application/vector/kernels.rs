use core::marker::PhantomData;
use std::borrow::Cow;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{
    BindingDecl, HephaestusError, KernelInterface, KernelSource, Result, VectorFactor,
    VectorScalar, Wgsl,
};

/// Workgroup width shared by the in-place vector kernels.
const WORKGROUP_WIDTH: u32 = 256;

/// Uniform parameters for the in-place vector kernels.
///
/// WGSL uniform blocks are laid out on sixteen-byte boundaries, so the live
/// fields are padded rather than left to an implicit tail. The trailing
/// words carry the factor's exact bits, placed per scalar by
/// [`VectorFactor::uniform_block`] to match that scalar's
/// [`VectorFactor::PARAMS_DECL`] shader view: `f32` reads word zero through
/// a `bitcast`, `f64` reads words one and two as a native field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct VectorParams {
    len: u32,
    word0: u32,
    word1: u32,
    word2: u32,
}

impl VectorParams {
    pub(super) fn new<T: VectorFactor<Wgsl>>(scalar: T, len: usize) -> Result<Self> {
        let checked = u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
            message: format!("vector length {len} exceeds u32::MAX"),
        })?;
        let [len, word0, word1, word2] = T::uniform_block(checked, scalar);
        Ok(Self {
            len,
            word0,
            word1,
            word2,
        })
    }
}

pub(super) fn workgroup_count(len: usize) -> Result<u32> {
    let groups = len.div_ceil(WORKGROUP_WIDTH as usize);
    u32::try_from(groups).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("dense vector length {len} exceeds the dispatchable workgroup count"),
    })
}

/// `target = target * factor`, in place.
pub(super) struct ScaleKernel<T>(pub(super) PhantomData<T>);

/// `target = target + factor * source`, in place.
pub(super) struct AxpyKernel<T>(pub(super) PhantomData<T>);

/// `target = source + factor * target`, in place.
pub(super) struct XpayKernel<T>(pub(super) PhantomData<T>);

fn scale_source<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>>() -> String {
    let load = T::factor_expr("params");
    format!(
        r#"{params}

@group(0) @binding(0) var<storage, read_write> destination: array<{ty}>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < params.len) {{
        let f: {ty} = {load};
        destination[index] = destination[index] * f;
    }}
}}
"#,
        params = T::PARAMS_DECL,
        ty = T::TYPE_TOKEN,
        load = load,
    )
}

fn axpy_source<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>>() -> String {
    let madd = T::madd_expr("f", "source[index]", "destination[index]");
    let load = T::factor_expr("params");
    format!(
        r#"{params}

@group(0) @binding(0) var<storage, read_write> destination: array<{ty}>;
@group(0) @binding(1) var<storage, read> source: array<{ty}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < params.len) {{
        let f: {ty} = {load};
        destination[index] = {madd};
    }}
}}
"#,
        params = T::PARAMS_DECL,
        ty = T::TYPE_TOKEN,
        load = load,
    )
}

fn xpay_source<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>>() -> String {
    let madd = T::madd_expr("f", "destination[index]", "source[index]");
    let load = T::factor_expr("params");
    format!(
        r#"{params}

@group(0) @binding(0) var<storage, read_write> destination: array<{ty}>;
@group(0) @binding(1) var<storage, read> source: array<{ty}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn xpay(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let index = gid.x;
    if (index < params.len) {{
        let f: {ty} = {load};
        destination[index] = {madd};
    }}
}}
"#,
        params = T::PARAMS_DECL,
        ty = T::TYPE_TOKEN,
        load = load,
    )
}

impl<T: Pod> KernelInterface for ScaleKernel<T> {
    type Params = VectorParams;

    const LABEL: &'static str = "hephaestus-vector-scale";
    const BINDINGS: &'static [BindingDecl] = &[BindingDecl::read_write::<T>()];
    const WORKGROUP: [u32; 3] = [WORKGROUP_WIDTH, 1, 1];
}

impl<T: Pod> KernelInterface for AxpyKernel<T> {
    type Params = VectorParams;

    const LABEL: &'static str = "hephaestus-vector-axpy";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<T>(),
        BindingDecl::read_only::<T>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_WIDTH, 1, 1];
}

impl<T: Pod> KernelInterface for XpayKernel<T> {
    type Params = VectorParams;

    const LABEL: &'static str = "hephaestus-vector-xpay";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<T>(),
        BindingDecl::read_only::<T>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_WIDTH, 1, 1];
}

impl<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>> KernelSource<Wgsl> for ScaleKernel<T> {
    const ENTRY: &'static str = "scale";

    fn source(&self) -> Cow<'static, str> {
        Cow::Owned(scale_source::<T>())
    }
}

impl<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>> KernelSource<Wgsl> for AxpyKernel<T> {
    const ENTRY: &'static str = "axpy";

    fn source(&self) -> Cow<'static, str> {
        Cow::Owned(axpy_source::<T>())
    }
}

impl<T: VectorScalar<Wgsl> + VectorFactor<Wgsl>> KernelSource<Wgsl> for XpayKernel<T> {
    const ENTRY: &'static str = "xpay";

    fn source(&self) -> Cow<'static, str> {
        Cow::Owned(xpay_source::<T>())
    }
}
