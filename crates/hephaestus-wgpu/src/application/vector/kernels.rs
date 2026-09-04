use std::borrow::Cow;

use eunomia::{Pod, Zeroable};
use hephaestus_core::{BindingDecl, HephaestusError, KernelInterface, KernelSource, Result, Wgsl};

/// Workgroup width shared by the in-place vector kernels.
const WORKGROUP_WIDTH: u32 = 256;

/// Uniform parameters for the in-place vector kernels.
///
/// WGSL uniform blocks are laid out on sixteen-byte boundaries, so the two
/// live fields are padded rather than left to an implicit tail.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(super) struct VectorParams {
    scalar: f32,
    len: u32,
    padding: [u32; 2],
}

impl VectorParams {
    pub(super) fn new(scalar: f32, len: usize) -> Result<Self> {
        Ok(Self {
            scalar,
            len: u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("vector length {len} exceeds u32::MAX"),
            })?,
            padding: [0; 2],
        })
    }
}

/// `target = target * factor`, in place.
pub(super) struct ScaleKernel;

/// `target = target + factor * source`, in place.
pub(super) struct AxpyKernel;

/// `target = source + factor * target`, in place.
pub(super) struct XpayKernel;

impl KernelInterface for ScaleKernel {
    type Params = VectorParams;

    const LABEL: &'static str = "hephaestus-vector-scale";
    const BINDINGS: &'static [BindingDecl] = &[BindingDecl::read_write::<f32>()];
    const WORKGROUP: [u32; 3] = [WORKGROUP_WIDTH, 1, 1];
}

impl KernelSource<Wgsl> for ScaleKernel {
    const ENTRY: &'static str = "scale";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r"
struct Params {
    factor: f32,
    len: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> destination: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        destination[index] = destination[index] * params.factor;
    }
}
",
        )
    }
}

impl KernelInterface for AxpyKernel {
    type Params = VectorParams;

    const LABEL: &'static str = "hephaestus-vector-axpy";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_WIDTH, 1, 1];
}

impl KernelSource<Wgsl> for AxpyKernel {
    const ENTRY: &'static str = "axpy";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r"
struct Params {
    factor: f32,
    len: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> destination: array<f32>;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        destination[index] = fma(params.factor, source[index], destination[index]);
    }
}
",
        )
    }
}

impl KernelInterface for XpayKernel {
    type Params = VectorParams;

    const LABEL: &'static str = "hephaestus-vector-xpay";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [WORKGROUP_WIDTH, 1, 1];
}

impl KernelSource<Wgsl> for XpayKernel {
    const ENTRY: &'static str = "xpay";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r"
struct Params {
    factor: f32,
    len: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> destination: array<f32>;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn xpay(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        destination[index] = fma(params.factor, destination[index], source[index]);
    }
}
",
        )
    }
}

/// Workgroup count covering `len` elements.
pub(super) fn workgroup_count(len: usize) -> Result<u32> {
    let len = u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("vector length {len} exceeds u32::MAX"),
    })?;
    Ok(len.div_ceil(WORKGROUP_WIDTH))
}
