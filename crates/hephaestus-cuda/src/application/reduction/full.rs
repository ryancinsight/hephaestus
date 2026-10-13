use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CombineExpr, ComputeDevice, CudaC, DeviceBuffer, DialectScalar, HephaestusError,
    IdentityToken, OpIdentity, Result, reduction_pass_count, validate_reduction_width,
};

use crate::CudaDevice;
use crate::application::linalg::to_u32;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;

use super::shaders::shader_source;

/// Run reduction on the CUDA device, returning a 1-element buffer holding the result.
pub fn reduction<Op, T>(device: &CudaDevice, input: &CudaBuffer<T>) -> Result<CudaBuffer<T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    reduction_with_width::<Op, T>(device, input, BlockWidth::DEFAULT)
}

/// Run reduction on the CUDA device with a caller-selected power-of-two block width.
pub fn reduction_with_width<Op, T>(
    device: &CudaDevice,
    input: &CudaBuffer<T>,
    width: BlockWidth,
) -> Result<CudaBuffer<T>>
where
    Op: CombineExpr<CudaC>,
    T: DialectScalar<CudaC> + Pod + OpIdentity<Op> + IdentityToken<Op, CudaC>,
{
    validate_reduction_width(width)?;

    if input.is_empty() {
        return device.upload(&[T::IDENTITY]);
    }

    if input.len() == 1 {
        let out = device.alloc_uninitialized::<T>(1)?;
        #[cfg(feature = "cuda")]
        // SAFETY: this device's context is current on this thread
        // (`alloc_uninitialized` above binds it); `input` and `out` are live
        // one-element device allocations (`input.len() == 1` checked above,
        // `out` freshly allocated with len 1), so the `size_of::<T>()`-byte
        // copy stays within both extents. The copy is asynchronous on the
        // null stream; both allocations outlive it because frees route
        // through synchronizing `cuMemFree`-family calls.
        unsafe {
            let bytes = core::mem::size_of::<T>();

            let res = (device.driver().memory.copy)(out.raw(), input.raw(), bytes);
            if res != 0 {
                return Err(HephaestusError::TransferFailed {
                    message: format!("cuMemcpyDtoD_v2 failed with code: {res}"),
                });
            }
        }
        return Ok(out);
    }

    // The kernel length argument is an unsigned 32-bit word. Validate the
    // initial count before allocation; each ceiling-division pass only shrinks it.
    let mut current_len = to_u32(input.len(), "reduction length")?;
    let mut temp_buffers: Vec<CudaBuffer<T>> =
        Vec::with_capacity(reduction_pass_count(input.len(), width));

    while current_len > 1 {
        let host_len = usize::try_from(current_len)
            .expect("u32 counts fit supported 32-bit and 64-bit host indices");
        let grid_size_val = grid_size(host_len, width)?;
        let out_len = usize::try_from(current_len.div_ceil(width.get()))
            .expect("a reduced u32 count fits the host index");
        let out_buffer = device.alloc_uninitialized::<T>(out_len)?;

        let key = PipelineKey::Reduction {
            op: std::any::TypeId::of::<Op>(),
            scalar: std::any::TypeId::of::<T>(),
            width: width.get(),
        };

        let kernel = cached_kernel(device, key, "reduction_kernel", || {
            shader_source::<Op, T>(width)
        })?;

        let source_ptr = if temp_buffers.is_empty() {
            input.raw()
        } else {
            temp_buffers
                .last()
                .expect("invariant: non-initial reduction pass has a previous buffer")
                .raw()
        };

        let mut src_val = source_ptr;
        let mut out_val = out_buffer.raw();
        let mut n_val = current_len;

        // Argument list mirrors `reduction_kernel(const T*, T*, unsigned int)`.
        let mut args: [*mut std::ffi::c_void; 3] = [
            std::ptr::from_mut(&mut src_val).cast(),
            std::ptr::from_mut(&mut out_val).cast(),
            std::ptr::from_mut(&mut n_val).cast(),
        ];

        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size_val, width),
            &mut args,
        )?;

        temp_buffers.push(out_buffer);
        current_len = current_len.div_ceil(width.get());
    }

    Ok(temp_buffers
        .pop()
        .expect("invariant: multi-element reduction allocates a final buffer"))
}
