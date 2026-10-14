use super::*;
use crate::infrastructure::device::WgpuDevice;
use hephaestus_core::{GroupedKernelSource, KernelSource, Wgsl};

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(crate) struct ScaleParams {
    pub(crate) len: u32,
    pub(crate) factor: f32,
}

pub(crate) struct ScaleKernel;

#[repr(C)]
#[derive(Clone, Copy, eunomia::Pod, eunomia::Zeroable)]
pub(crate) struct GroupedParams {
    pub(crate) len: u32,
    pub(crate) addend: f32,
}

pub(crate) struct GroupedAddKernel;

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

pub(crate) const OVERSIZED_PARAMETER_WORDS: usize = 16_385;

pub(crate) static OVERSIZED_PARAMETERS: [u32; OVERSIZED_PARAMETER_WORDS] =
    [0; OVERSIZED_PARAMETER_WORDS];

pub(crate) struct OversizedGroupedKernel;

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

pub(crate) fn try_device() -> Option<WgpuDevice> {
    WgpuDevice::try_with_power_preference_and_adapter_config(
        "hephaestus-wgpu-stream-test",
        wgpu::PowerPreference::HighPerformance,
        |_| wgpu::Features::empty(),
        |_| wgpu::Limits::default(),
    )
    .ok()
}
