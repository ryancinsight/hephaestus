//! ROCm implementation of [`hephaestus_core::CrossProductOps`].
//!
//! One thread per `(x, y, z)` triple evaluates the standard formula
//! directly — no strided metadata is needed since the operands are flat,
//! contiguous buffers, matching `DenseVectorOps`'s own convention.

use std::any::TypeId;

use eunomia::Pod;
use hephaestus_core::{
    BlockWidth, CrossProductOps, DeviceBuffer, DialectScalar, HephaestusError, HipC, Result,
    validate_cross_product_lengths,
};

use crate::RocmDevice;
use crate::application::linalg::to_u32;
use crate::application::pipeline::{
    LaunchConfig, PipelineKey, cached_kernel, grid_size, launch_kernel,
};
use crate::infrastructure::{DevicePtr, RocmBuffer};

fn cross_product_source<T: DialectScalar<HipC>>() -> String {
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

/// ROCm device marker implementing [`CrossProductOps`].
#[derive(Clone, Copy, Debug, Default)]
pub struct RocmCrossProductOps;

impl<T> CrossProductOps<RocmDevice, T> for RocmCrossProductOps
where
    T: DialectScalar<HipC> + Pod,
{
    fn cross_into(
        &self,
        device: &RocmDevice,
        a: &RocmBuffer<T>,
        b: &RocmBuffer<T>,
        out: &RocmBuffer<T>,
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

        let mut a_ptr: DevicePtr = a.raw();
        let mut b_ptr: DevicePtr = b.raw();
        let mut out_ptr: DevicePtr = out.raw();
        let mut triples_val = to_u32(triples, "triple count")?;

        // Argument list mirrors `cross_product_kernel(const T*, const T*,
        // T*, unsigned int)`.
        let mut args: [*mut core::ffi::c_void; 4] = [
            core::ptr::addr_of_mut!(a_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(b_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(out_ptr).cast::<DevicePtr>().cast(),
            core::ptr::addr_of_mut!(triples_val).cast::<u32>().cast(),
        ];
        launch_kernel(
            device,
            &kernel,
            LaunchConfig::linear(grid_size(triples, width)?, width),
            &mut args,
        )
    }
}
