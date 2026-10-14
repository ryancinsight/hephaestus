use super::*;
use hephaestus_core::{GroupedKernelSource, Wgsl};
use std::borrow::Cow;

use hephaestus_core::{
    Access, BindingDecl, ComputeDevice, GroupedBindingDecl, GroupedKernelInterface, KernelInterface,
};

impl KernelInterface for ScaleKernel {
    type Params = ScaleParams;
    const LABEL: &'static str = "hephaestus-wgpu-stream-scale";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [64, 1, 1];
}

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

mod command_stream;
mod fixtures;
mod grouped_dispatch;
mod stream_setup;
pub(crate) use fixtures::*;

pub(crate) use command_stream::*;
pub(crate) use grouped_dispatch::*;
