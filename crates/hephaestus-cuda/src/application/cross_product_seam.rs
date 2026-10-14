//! CUDA implementation of [`hephaestus_core::CrossProductOps`].
//!
//! One thread per `(x, y, z)` triple evaluates the standard formula
//! directly — no strided metadata is needed since the operands are flat,
//! contiguous buffers, matching `DenseVectorOps`'s own convention.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CrossProductOps, CudaC, DeviceBuffer, DialectScalar, HephaestusError, Result,
    validate_cross_product_lengths,
};

use crate::application::linalg::to_u32;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::buffer::CudaBuffer;
use crate::infrastructure::device::CudaDevice;

fn cross_product_source<T: DialectScalar<CudaC>>() -> String {
    format!(
        r#"{prelude}extern "C" __global__ void cross_product_kernel(
    const {ty}* a,
    const {ty}* b,
    {ty}* out,
    unsigned int triples
) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= triples) {{
        return;
    }}
    unsigned int base = i * 3u;
    {ty} ax = a[base];
    {ty} ay = a[base + 1u];
    {ty} az = a[base + 2u];
    {ty} bx = b[base];
    {ty} by = b[base + 1u];
    {ty} bz = b[base + 2u];
    out[base] = ay * bz - az * by;
    out[base + 1u] = az * bx - ax * bz;
    out[base + 2u] = ax * by - ay * bx;
}}
"#,
        prelude = T::PRELUDE,
        ty = T::TYPE_TOKEN,
    )
}

/// CUDA device marker implementing [`CrossProductOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaCrossProductOps;

impl<T> CrossProductOps<CudaDevice, T> for CudaCrossProductOps
where
    T: DialectScalar<CudaC> + Pod,
{
    fn cross_into(
        &self,
        device: &CudaDevice,
        a: &CudaBuffer<T>,
        b: &CudaBuffer<T>,
        out: &CudaBuffer<T>,
    ) -> Result<()> {
        let triples = validate_cross_product_lengths(a.len(), b.len(), out.len())?;
        if a.aliases(out) || b.aliases(out) {
            return Err(HephaestusError::DispatchFailed {
                message: "cross product output must not alias either input".to_string(),
            });
        }
        if triples == 0 {
            return Ok(());
        }

        let width = BlockWidth::DEFAULT;
        let key = PipelineKey::CrossProduct {
            scalar: TypeId::of::<T>(),
            width: width.get(),
        };
        let kernel = cached_kernel(device, key, "cross_product_kernel", || {
            cross_product_source::<T>()
        })?;

        let mut a_ptr = a.raw();
        let mut b_ptr = b.raw();
        let mut out_ptr = out.raw();
        let mut triples_val = to_u32(triples, "triple count")?;

        // Argument list mirrors `cross_product_kernel(const T*, const T*,
        // T*, unsigned int)`.
        let mut args: [*mut core::ffi::c_void; 4] = [
            (&mut a_ptr as *mut u64).cast(),
            (&mut b_ptr as *mut u64).cast(),
            (&mut out_ptr as *mut u64).cast(),
            (&mut triples_val as *mut u32).cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(triples, width)?, width),
            &mut args,
        )
    }
}
