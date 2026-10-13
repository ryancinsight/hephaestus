//! Prepared command streams and their trait impls.

use crate::infrastructure::device::PipelineCache;
use std::marker::PhantomData;

/// Prepared WGPU pipeline for a kernel source `K`.
pub struct WgpuPrepared<K> {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) parameter_binding: u32,
    pub(crate) label: &'static str,
    pub(crate) marker: PhantomData<K>,
}

/// Prepared WGPU pipeline for a grouped kernel source `K`.
pub struct WgpuGroupedPrepared<K> {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layouts: Vec<(u32, wgpu::BindGroupLayout)>,
    pub(crate) parameter_group: u32,
    pub(crate) parameter_binding: u32,
    pub(crate) owner: PipelineCache,
    pub(crate) label: &'static str,
    pub(crate) marker: PhantomData<K>,
}

macro_rules! impl_prepared_traits {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        impl<K> core::fmt::Debug for $name<K> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let mut debug = f.debug_struct(stringify!($name));
                $(debug.field(stringify!($field), &self.$field);)+
                debug.finish_non_exhaustive()
            }
        }

        impl<K> Clone for $name<K> {
            fn clone(&self) -> Self {
                Self {
                    $($field: self.$field.clone(),)+
                    marker: PhantomData,
                }
            }
        }
    };
}

impl_prepared_traits!(WgpuGroupedPrepared {
    pipeline,
    bind_group_layouts,
    parameter_group,
    parameter_binding,
    owner,
    label
});
impl_prepared_traits!(WgpuPrepared {
    pipeline,
    bind_group_layout,
    parameter_binding,
    label
});
