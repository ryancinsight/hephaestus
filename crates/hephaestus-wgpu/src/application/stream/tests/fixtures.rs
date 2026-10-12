use std::borrow::Cow;

use hephaestus_core::{
    BindingDecl, GroupedBindingDecl, GroupedKernelInterface, GroupedKernelSource, KernelInterface,
    KernelSource, Wgsl,
};

use crate::infrastructure::device::WgpuDevice;

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(super) struct ScaleParams {
    pub(super) len: u32,
    pub(super) factor: f32,
}

pub(super) struct ScaleKernel;

impl KernelInterface for ScaleKernel {
    type Params = ScaleParams;
    const LABEL: &'static str = "hephaestus-wgpu-stream-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(super) struct GroupedParams {
    pub(super) len: u32,
    pub(super) addend: f32,
}

pub(super) struct GroupedAddKernel;

impl GroupedKernelInterface for GroupedAddKernel {
    type Params = GroupedParams;
    const LABEL: &'static str = "hephaestus-wgpu-grouped-add";
    const BINDINGS: &'static [GroupedBindingDecl] = &[
        GroupedBindingDecl::read_only::<f32>(0, 0),
        GroupedBindingDecl::read_only::<f32>(1, 0),
        GroupedBindingDecl::read_write::<f32>(1, 1),
    ];
    const PARAM_GROUP: u32 = 0;
    const PARAM_BINDING: u32 = 1;
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

impl GroupedKernelSource<Wgsl> for GroupedAddKernel {
    const ENTRY: &'static str = "main";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
struct Params {
    len: u32,
    addend: f32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read> right: array<f32>;
@group(1) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = left[i] + right[i] + params.addend;
    }
}
"#,
        )
    }
}

pub(super) const OVERSIZED_PARAMETER_WORDS: usize = 16_385;
pub(super) static OVERSIZED_PARAMETERS: [u32; OVERSIZED_PARAMETER_WORDS] =
    [0; OVERSIZED_PARAMETER_WORDS];

pub(super) struct OversizedGroupedKernel;

impl GroupedKernelInterface for OversizedGroupedKernel {
    type Params = [u32; OVERSIZED_PARAMETER_WORDS];
    const LABEL: &'static str = "hephaestus-wgpu-oversized-grouped";
    const BINDINGS: &'static [GroupedBindingDecl] = GroupedAddKernel::BINDINGS;
    const PARAM_GROUP: u32 = GroupedAddKernel::PARAM_GROUP;
    const PARAM_BINDING: u32 = GroupedAddKernel::PARAM_BINDING;
    const WORKGROUP: [u32; 3] = GroupedAddKernel::WORKGROUP;
}

impl GroupedKernelSource<Wgsl> for OversizedGroupedKernel {
    const ENTRY: &'static str = GroupedAddKernel::ENTRY;

    fn source(&self) -> Cow<'static, str> {
        GroupedAddKernel.source()
    }
}

impl KernelSource<Wgsl> for ScaleKernel {
    const ENTRY: &'static str = "main";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r#"
struct Params {
    len: u32,
    factor: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < params.len) {
        output[i] = input[i] * params.factor;
    }
}
"#,
        )
    }
}

pub(super) fn try_device() -> Option<WgpuDevice> {
    WgpuDevice::try_with_power_preference_and_adapter_config(
        "hephaestus-wgpu-stream-test",
        wgpu::PowerPreference::HighPerformance,
        |_| wgpu::Features::empty(),
        |_| wgpu::Limits::default(),
    )
    .ok()
}
