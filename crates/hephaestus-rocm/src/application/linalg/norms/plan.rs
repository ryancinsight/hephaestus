//! Prepared and retained map-reduction plans on ROCm.

use std::marker::PhantomData;
use std::sync::Arc;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, ComputeDevice, DeviceBuffer, DialectScalar, HephaestusError, HipC, IdentityToken,
    OpIdentity, Result, checked_shared_bytes,
};
use leto::Layout;

use super::super::map_layout_err;
use super::shader::{MapReductionMeta, MapReductionOp, map_shape, map_strides, shader_source};
use crate::RocmDevice;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, RocmKernel, cached_kernel, grid_size, launch_kernel,
};
use crate::application::prepared_reduction::PreparedReductionPlan;
use crate::application::strided::StridedOperand;
use crate::infrastructure::{DevicePtr, RocmBuffer};

pub(crate) struct PreparedMapReduction<'a, Op, T, const N: usize> {
    resources: MapReductionResources<T>,
    a: &'a RocmBuffer<T>,
    b: &'a RocmBuffer<T>,
    _operation: PhantomData<Op>,
}

/// Owned dispatch resources behind the borrowing and retained map-reduction
/// plans.
///
/// Both plans launch through [`MapReductionResources::launch`], so their
/// dispatches cannot drift apart: the borrowing form passes its borrowed
/// operands' pointers, the retained form passes the pointers its caller
/// presents after the provenance check.
pub(crate) struct MapReductionResources<T> {
    device: RocmDevice,
    meta: MapReductionMeta,
    partial: RocmBuffer<T>,
    kernel: Option<Arc<RocmKernel>>,
    groups: u32,
    width: BlockWidth,
    shared_bytes: u32,
    reduction: Option<PreparedReductionPlan<T>>,
}

impl<T> MapReductionResources<T> {
    fn launch(&self, a_raw: DevicePtr, b_raw: DevicePtr) -> Result<()> {
        if let Some(kernel) = &self.kernel {
            let mut meta = self.meta;
            let mut a_ptr: DevicePtr = a_raw;
            let mut b_ptr: DevicePtr = b_raw;
            let mut output_ptr: DevicePtr = self.partial.raw();
            let mut args: [*mut core::ffi::c_void; 4] = [
                core::ptr::addr_of_mut!(meta)
                    .cast::<MapReductionMeta>()
                    .cast(),
                core::ptr::addr_of_mut!(a_ptr).cast::<DevicePtr>().cast(),
                core::ptr::addr_of_mut!(b_ptr).cast::<DevicePtr>().cast(),
                core::ptr::addr_of_mut!(output_ptr)
                    .cast::<DevicePtr>()
                    .cast(),
            ];
            launch_kernel(
                &self.device,
                kernel,
                LaunchConfig::linear_shared(self.groups, self.width, self.shared_bytes),
                &mut args,
            )?;
        }
        if let Some(reduction) = &self.reduction {
            reduction.dispatch(&self.partial)?;
        }
        Ok(())
    }

    fn output(&self) -> &RocmBuffer<T> {
        self.reduction
            .as_ref()
            .map_or(&self.partial, PreparedReductionPlan::output)
    }

    fn into_output(self) -> RocmBuffer<T> {
        match self.reduction {
            Some(reduction) => reduction.into_output(),
            None => self.partial,
        }
    }
}

/// Owned twin of [`PreparedMapReduction`]: the same dispatch resources with
/// `Copy` operand provenance instead of operand borrows.
///
/// `RocmBuffer` owns its allocation, so a retained handle cannot keep the
/// prepare-time operands alive by cloning. It does not need to: the launch
/// reads its operand pointers at dispatch time, so this form requires the
/// live buffers at every dispatch and launches only over the passed
/// pointers after the provenance check in
/// [`dispatch_with`](Self::dispatch_with). A retained handle therefore
/// reduces exactly the memory its caller presents on that dispatch — the
/// stored provenance is never dereferenced, only compared, so dropping the
/// prepare-time buffers cannot dangle a later dispatch.
///
/// The check is load-bearing for memory safety, not just caller hygiene:
/// the launch configuration was derived from the prepare-time shape, so a
/// dispatch over differently-sized memory would read out of bounds. That is
/// why the check lives inside [`dispatch_with`](Self::dispatch_with)
/// rather than with the caller.
pub(crate) struct RetainedMapReduction<Op, T, const N: usize> {
    resources: MapReductionResources<T>,
    a_raw: DevicePtr,
    a_len: usize,
    b_raw: DevicePtr,
    b_len: usize,
    _operation: PhantomData<Op>,
}

impl<Op, T, const N: usize> PreparedMapReduction<'_, Op, T, N>
where
    Op: MapReductionOp,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, HipC>,
{
    pub(crate) fn dispatch(&self) -> Result<()> {
        self.resources.launch(self.a.raw(), self.b.raw())
    }

    pub(crate) fn output(&self) -> &RocmBuffer<T> {
        self.resources.output()
    }

    pub(crate) fn into_output(self) -> RocmBuffer<T> {
        self.resources.into_output()
    }

    pub(crate) fn into_retained(self) -> RetainedMapReduction<Op, T, N> {
        let (a, b) = (self.a, self.b);
        RetainedMapReduction {
            resources: self.resources,
            a_raw: a.raw(),
            a_len: a.len(),
            b_raw: b.raw(),
            b_len: b.len(),
            _operation: PhantomData,
        }
    }
}

impl<Op, T, const N: usize> RetainedMapReduction<Op, T, N>
where
    Op: MapReductionOp,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, HipC>,
{
    /// Launch over `a` and `b` after verifying they are the retained
    /// allocations.
    ///
    /// # Errors
    ///
    /// Returns a typed error when either buffer names a different device
    /// allocation than the plan was retained against, or when a native
    /// launch fails.
    pub(crate) fn dispatch_with(&self, a: &RocmBuffer<T>, b: &RocmBuffer<T>) -> Result<()> {
        if a.raw() != self.a_raw
            || a.len() != self.a_len
            || b.raw() != self.b_raw
            || b.len() != self.b_len
        {
            return Err(HephaestusError::DispatchFailed {
                message: "retained reduction received different device allocations".to_string(),
            });
        }
        self.resources.launch(a.raw(), b.raw())
    }

    pub(crate) fn output(&self) -> &RocmBuffer<T> {
        self.resources.output()
    }
}

pub(crate) fn prepare_map_reduction<'a, Op, T, const N: usize>(
    device: &RocmDevice,
    a: StridedOperand<'a, T, N>,
    b: StridedOperand<'a, T, N>,
) -> Result<PreparedMapReduction<'a, Op, T, N>>
where
    Op: MapReductionOp,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, HipC>,
{
    prepare_map_reduction_with_layouts::<Op, T, N>(device, a.buffer, a.layout, b.buffer, b.layout)
}

pub(crate) fn prepare_map_reduction_with_layouts<'a, Op, T, const N: usize>(
    device: &RocmDevice,
    a: &'a RocmBuffer<T>,
    a_layout: &Layout<N>,
    b: &'a RocmBuffer<T>,
    b_layout: &Layout<N>,
) -> Result<PreparedMapReduction<'a, Op, T, N>>
where
    Op: MapReductionOp,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, HipC>,
{
    let len = a_layout.checked_size().map_err(map_layout_err)?;
    if len == 0 {
        return Ok(PreparedMapReduction {
            resources: MapReductionResources {
                device: device.clone(),
                meta: MapReductionMeta {
                    shape: [1; 4],
                    a_strides: [0; 4],
                    b_strides: [0; 4],
                    offsets: [0; 4],
                },
                partial: device.upload(&[T::IDENTITY])?,
                kernel: None,
                groups: 0,
                width: BlockWidth::DEFAULT,
                shared_bytes: 0,
                reduction: None,
            },
            a,
            b,
            _operation: PhantomData,
        });
    }
    a_layout
        .validate_storage_len(a.len())
        .map_err(map_layout_err)?;
    b_layout
        .validate_storage_len(b.len())
        .map_err(map_layout_err)?;

    let width = BlockWidth::DEFAULT;
    let groups = grid_size(len, width)?;
    let partial_len = usize::try_from(groups).map_err(|_| HephaestusError::DispatchFailed {
        message: format!("ROCm map-reduction group count {groups} exceeds usize range"),
    })?;
    let partial = device.alloc_uninitialized::<T>(partial_len)?;
    let meta = MapReductionMeta {
        shape: map_shape(a_layout.shape())?,
        a_strides: map_strides(a_layout.strides())?,
        b_strides: map_strides(b_layout.strides())?,
        offsets: [
            u32::try_from(a_layout.offset()).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("input offset {} exceeds u32 range", a_layout.offset()),
            })?,
            u32::try_from(b_layout.offset()).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("input offset {} exceeds u32 range", b_layout.offset()),
            })?,
            0,
            u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("logical size {len} exceeds u32 range"),
            })?,
        ],
    };
    let shared_bytes = checked_shared_bytes::<T>(width)?;
    let key = PipelineKey::MapReduction {
        op: core::any::TypeId::of::<Op>(),
        scalar: core::any::TypeId::of::<T>(),
        width: width.get(),
    };
    let kernel = cached_kernel(device, key, "map_reduction_kernel", || {
        shader_source::<Op, T>(width)
    })?;
    let reduction = if groups > 1 {
        Some(PreparedReductionPlan::prepare::<Op::ReduceOp>(
            device,
            partial_len,
            width,
        )?)
    } else {
        None
    };

    Ok(PreparedMapReduction {
        resources: MapReductionResources {
            device: device.clone(),
            meta,
            partial,
            kernel: Some(kernel),
            groups,
            width,
            shared_bytes,
            reduction,
        },
        a,
        b,
        _operation: PhantomData,
    })
}

pub(super) fn map_reduction<Op, T, const N: usize>(
    device: &RocmDevice,
    a: StridedOperand<'_, T, N>,
    b: StridedOperand<'_, T, N>,
) -> Result<RocmBuffer<T>>
where
    Op: MapReductionOp,
    T: DialectScalar<HipC> + Pod + OpIdentity<Op::ReduceOp> + IdentityToken<Op::ReduceOp, HipC>,
{
    let prepared = prepare_map_reduction::<Op, T, N>(device, a, b)?;
    prepared.dispatch()?;
    Ok(prepared.into_output())
}
