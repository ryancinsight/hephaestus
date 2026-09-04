//! Metal-selected authored-kernel command streams.

use std::marker::PhantomData;

use eunomia::Pod;
use hephaestus_core::{
    Binding, CommandStream, DispatchGrid, GroupedBinding, GroupedCommandStream,
    GroupedKernelDevice, GroupedKernelSequence, GroupedKernelSource, KernelDevice, KernelSource,
    Result, Wgsl, validate_bindings, validate_grouped_bindings,
};
use hephaestus_wgpu::application::stream::{
    WgpuCommandStream, WgpuGroupedPrepared, WgpuGroupedSequence, WgpuPrepared,
};
use hephaestus_wgpu::infrastructure::device::WgpuDevice;

use crate::{MetalBuffer, MetalDevice};

/// Prepared Metal-selected pipeline for an authored kernel source type `K`.
pub struct MetalPrepared<K> {
    inner: WgpuPrepared<K>,
}

impl<K> core::fmt::Debug for MetalPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalPrepared")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for MetalPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Prepared Metal-selected pipeline for a grouped authored kernel source type
/// `K`.
pub struct MetalGroupedPrepared<K> {
    inner: WgpuGroupedPrepared<K>,
}

impl<K> core::fmt::Debug for MetalGroupedPrepared<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalGroupedPrepared")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl<K> Clone for MetalGroupedPrepared<K> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Metal command stream for ordered authored-kernel dispatch, copies, and
/// fills.
pub struct MetalCommandStream<'d> {
    inner: WgpuCommandStream<'d>,
}

/// Active grouped Metal sequence backed by one WGPU compute pass selected for
/// the native Metal adapter.
pub struct MetalGroupedSequence<'s> {
    inner: *mut (),
    marker: PhantomData<&'s mut ()>,
}

impl KernelDevice for MetalDevice {
    type Dialect = Wgsl;
    type BindingHandle<'a> = &'a hephaestus_wgpu::wgpu::Buffer;
    type Prepared<K: KernelSource<Wgsl>> = MetalPrepared<K>;
    type Stream<'d> = MetalCommandStream<'d>;

    #[inline]
    fn binding_handle<T: Pod>(buffer: &Self::Buffer<T>) -> Self::BindingHandle<'_> {
        buffer.wgpu_buffer().raw()
    }

    fn prepare<K: KernelSource<Wgsl>>(&self, kernel: &K) -> Result<Self::Prepared<K>> {
        Ok(MetalPrepared {
            inner: <WgpuDevice as KernelDevice>::prepare(self.wgpu_device(), kernel)?,
        })
    }

    fn stream(&self) -> Result<Self::Stream<'_>> {
        Ok(MetalCommandStream {
            inner: <WgpuDevice as KernelDevice>::stream(self.wgpu_device())?,
        })
    }
}

impl GroupedKernelDevice for MetalDevice {
    type GroupedPrepared<K: GroupedKernelSource<Wgsl>> = MetalGroupedPrepared<K>;
    type GroupedStream<'d> = MetalCommandStream<'d>;

    fn prepare_grouped<K: GroupedKernelSource<Wgsl>>(
        &self,
        kernel: &K,
    ) -> Result<Self::GroupedPrepared<K>> {
        Ok(MetalGroupedPrepared {
            inner: <WgpuDevice as GroupedKernelDevice>::prepare_grouped(
                self.wgpu_device(),
                kernel,
            )?,
        })
    }

    fn grouped_stream(&self) -> Result<Self::GroupedStream<'_>> {
        self.stream()
    }
}

impl<'d> CommandStream<'d, MetalDevice> for MetalCommandStream<'d> {
    fn encode<K: KernelSource<Wgsl>>(
        &mut self,
        prepared: &MetalPrepared<K>,
        bindings: &[Binding<'_, MetalDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_bindings::<MetalDevice>(K::LABEL, K::BINDINGS, bindings)?;
        let mapped = bindings.iter().map(to_wgpu_binding).collect::<Vec<_>>();
        CommandStream::encode(&mut self.inner, &prepared.inner, &mapped, params, grid)
    }

    fn copy<T: Pod>(&mut self, src: &MetalBuffer<T>, dst: &MetalBuffer<T>) -> Result<()> {
        CommandStream::copy(&mut self.inner, src.wgpu_buffer(), dst.wgpu_buffer())
    }

    fn copy_prefix<T: Pod>(
        &mut self,
        src: &MetalBuffer<T>,
        dst: &MetalBuffer<T>,
        elements: usize,
    ) -> Result<()> {
        CommandStream::copy_prefix(
            &mut self.inner,
            src.wgpu_buffer(),
            dst.wgpu_buffer(),
            elements,
        )
    }

    fn fill_zero<T: Pod>(&mut self, dst: &MetalBuffer<T>) -> Result<()> {
        CommandStream::fill_zero(&mut self.inner, dst.wgpu_buffer())
    }

    fn submit(self) -> Result<()> {
        CommandStream::submit(self.inner)
    }
}

impl<'d> GroupedCommandStream<'d, MetalDevice> for MetalCommandStream<'d> {
    type Sequence<'s> = MetalGroupedSequence<'s>;

    fn encode_grouped<K: GroupedKernelSource<Wgsl>>(
        &mut self,
        prepared: &MetalGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, MetalDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<MetalDevice>(K::LABEL, K::BINDINGS, bindings)?;
        let mapped = bindings
            .iter()
            .map(to_wgpu_grouped_binding)
            .collect::<Vec<_>>();
        GroupedCommandStream::encode_grouped(
            &mut self.inner,
            &prepared.inner,
            &mapped,
            params,
            grid,
        )
    }

    fn encode_grouped_sequence<F>(&mut self, label: &str, encode: F) -> Result<()>
    where
        F: FnOnce(&mut Self::Sequence<'_>) -> Result<()>,
    {
        GroupedCommandStream::encode_grouped_sequence(&mut self.inner, label, |inner| {
            let mut sequence = MetalGroupedSequence {
                inner: inner as *mut WgpuGroupedSequence<'_> as *mut (),
                marker: PhantomData,
            };
            encode(&mut sequence)
        })
    }

    fn submit_grouped(self) -> Result<()> {
        GroupedCommandStream::submit_grouped(self.inner)
    }
}

impl<'s> GroupedKernelSequence<'s, MetalDevice> for MetalGroupedSequence<'s> {
    fn encode_grouped<K: GroupedKernelSource<Wgsl>>(
        &mut self,
        prepared: &MetalGroupedPrepared<K>,
        bindings: &[GroupedBinding<'_, MetalDevice>],
        params: &K::Params,
        grid: DispatchGrid,
    ) -> Result<()> {
        validate_grouped_bindings::<MetalDevice>(K::LABEL, K::BINDINGS, bindings)?;
        let mapped = bindings
            .iter()
            .map(to_wgpu_grouped_binding)
            .collect::<Vec<_>>();
        // SAFETY: `inner` is created from the uniquely borrowed WGPU grouped
        // sequence for the duration of the enclosing callback. The callback
        // does not escape this wrapper, and no second mutable access exists
        // while this method is executing.
        let inner = unsafe { &mut *self.inner.cast::<WgpuGroupedSequence<'s>>() };
        GroupedKernelSequence::encode_grouped(inner, &prepared.inner, &mapped, params, grid)
    }
}

fn to_wgpu_binding<'a>(binding: &Binding<'a, MetalDevice>) -> Binding<'a, WgpuDevice> {
    Binding {
        access: binding.access,
        elem_size: binding.elem_size,
        len: binding.len,
        handle: binding.handle,
    }
}

fn to_wgpu_grouped_binding<'a>(
    binding: &GroupedBinding<'a, MetalDevice>,
) -> GroupedBinding<'a, WgpuDevice> {
    GroupedBinding {
        group: binding.group,
        binding: binding.binding,
        access: binding.access,
        elem_size: binding.elem_size,
        len: binding.len,
        handle: binding.handle,
    }
}
